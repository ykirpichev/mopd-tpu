"""GRPO advantages and the off-policy-corrected policy loss.

Both modes use the same loss. The behavior log-probs are the ones the rollout
engine recorded while sampling, so no old-policy weights are kept anywhere.
The learner's forward pass under the current weights gives logp_cur; the
per-token importance weight exp(logp_cur - logp_behav) corrects both policy
staleness (async) and the sampler/trainer numeric mismatch (both modes).
"""
from dataclasses import dataclass

import torch


@dataclass
class LossConfig:
    kind: str = "tis"  # "tis": truncated-IS policy gradient; "ppo": PPO clip against behavior log-probs
    is_cap: float = 2.0  # truncation C for the importance weight (tis)
    clip_low: float = 0.2  # PPO clip range (ppo)
    clip_high: float = 0.28


def group_advantages(rewards, eps=1e-6, scale_by_std=True):
    """rewards: [P, G] -> advantages [P, G] relative to each prompt's group mean."""
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if scale_by_std:
        adv = adv / (rewards.std(dim=1, keepdim=True, unbiased=False) + eps)
    return adv


def policy_loss(logp_cur, logp_behav, adv, mask, cfg: LossConfig, denom=None):
    """Token-level loss summed over tokens and divided by `denom` (default: mask tokens).

    logp_cur:   [N, T] log-probs of the sampled tokens under the current weights (requires grad)
    logp_behav: [N, T] log-probs recorded by the rollout engine at sampling time
    adv:        [N]    sequence advantage, or [N, T] per-token advantage (on-policy distillation)
    mask:       [N, T] 1 for completion tokens
    """
    mask = mask.to(logp_cur.dtype)
    a = adv.to(logp_cur.dtype)
    a = a.unsqueeze(1) if a.dim() == 1 else a
    log_ratio = (logp_cur - logp_behav).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    if cfg.kind == "tis":
        w = ratio.detach().clamp(max=cfg.is_cap)
        # exp(lp - sg(lp)) == 1 in value with gradient d lp; the proximal policy is the current one.
        tok = -w * a * torch.exp(logp_cur - logp_cur.detach())
        truncated = (ratio.detach() > cfg.is_cap).to(mask.dtype)
    elif cfg.kind == "ppo":
        unclipped = ratio * a
        clipped = ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high) * a
        tok = -torch.minimum(unclipped, clipped)
        truncated = (unclipped != clipped).to(mask.dtype).detach()
    else:
        raise ValueError(cfg.kind)
    n_tok = mask.sum().clamp(min=1.0)
    d = n_tok if denom is None else denom
    loss = (tok * mask).sum() / d
    with torch.no_grad():
        lr_ = log_ratio.detach()
        stats = {
            "ratio_mean": float((ratio.detach() * mask).sum() / n_tok),
            "ratio_max": float((ratio.detach() * mask).max()),
            "truncated_frac": float((truncated * mask).sum() / n_tok),
            # k3 estimator of KL(behavior || current) on sampled tokens
            "kl_behav_cur": float(((torch.exp(lr_) - 1 - lr_) * mask).sum() / n_tok),
            "tokens": float(n_tok),
        }
    return loss, stats


def behavior_diagnostics(logp_cur, logp_behav, mask, is_cap=2.0):
    """Sampler-vs-trainer statistics that are logged but never enter the loss (no-IS ablation):
    the IS ratio the baseline would have used, how often it would be truncated, and KL(behavior||current)."""
    with torch.no_grad():
        mask = mask.to(torch.float32)
        n = mask.sum().clamp(min=1.0)
        lr_ = (logp_cur.detach().float() - logp_behav.float()).clamp(-20.0, 20.0)
        r = torch.exp(lr_)
        return {
            "behav_ratio_mean": float((r * mask).sum() / n),
            "behav_ratio_max": float((r * mask).max()),
            "behav_ratio_min": float(torch.where(mask > 0, r, torch.full_like(r, float("inf"))).min()),
            "behav_would_truncate_frac": float(((r > is_cap).float() * mask).sum() / n),
            "kl_behav_cur": float(((r - 1 - lr_) * mask).sum() / n),
        }


def recomputed_old_loss(logp_cur, logp_old, adv, mask, cfg: LossConfig, denom=None):
    """No-IS ablation: PPO clip against log-probs the trainer itself recomputed before this batch's first
    update. Sampler log-probs play no part, so staleness and sampler/trainer mismatch go uncorrected.
    With one update per batch logp_old == logp_cur in value, the ratio is exactly 1 and the clip never
    binds: the loss is the plain policy gradient that treats stale samples as on-policy."""
    loss, st = policy_loss(logp_cur, logp_old, adv, mask, LossConfig(kind="ppo", clip_low=cfg.clip_low,
                                                                      clip_high=cfg.clip_high), denom)
    return loss, {"old_ratio_mean": st["ratio_mean"], "old_ratio_max": st["ratio_max"],
                  "clip_frac": st["truncated_frac"], "tokens": st["tokens"]}


ADV_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def distill_advantage(logp_cur, logp_teacher, mask, clip=5.0):
    """MOPD / OPD per-token advantage sg[log pi_T(y_t) - log pi_S(y_t)], clipped to +-clip (MOPD paper).
    Its negative mean over the mask is the sampled estimate of the per-token reverse KL(pi_S || pi_T)."""
    with torch.no_grad():
        d = (logp_teacher.to(logp_cur.dtype) - logp_cur.detach()) * mask
        a = d.clamp(-clip, clip) * mask
        n = mask.sum().clamp(min=1.0)
        stats = {"kl_teacher": float(-d.sum() / n),
                 "adv_clip_frac": float(((d.abs() >= clip).to(mask.dtype) * mask).sum() / n)}
        for t in ADV_THRESHOLDS:  # share of tokens a tighter clip would bind (clip-sensitivity analysis)
            stats[f"adv_abs_gt{t:g}"] = float(((d.abs() > t).to(mask.dtype) * mask).sum() / n)
    return a, stats



def reverse_kl_piece(h, h_teacher, w_student, w_teacher, targets):
    """Exact per-position reverse KL(pi_S || pi_T) over the full vocabulary for one chunk of positions.

    h:          [C, H] student final hidden states (requires grad)
    h_teacher:  [C, H] teacher final hidden states (shipped, no grad)
    w_*:        [V, H] output layers (the teacher's is frozen)
    targets:    [C]    sampled tokens, for the sampled-token diagnostics
    Returns kl [C] (differentiable), and detached: student log-prob of the target, student entropy, teacher
    log-prob of the target rebuilt from the shipped hidden state."""
    ls = torch.log_softmax((h @ w_student.t()).float(), -1)
    with torch.no_grad():
        lt = torch.log_softmax((h_teacher.to(w_teacher.dtype) @ w_teacher.t()).float(), -1)
    ps = ls.exp()
    kl = (ps * (ls - lt)).sum(-1)
    with torch.no_grad():
        y = targets.unsqueeze(-1)
        lp = torch.gather(ls, -1, y).squeeze(-1)
        ent = -(ps * ls).sum(-1)
        tlp = torch.gather(lt, -1, y).squeeze(-1)
    return kl, lp.detach(), ent.detach(), tlp
