"""The bucket delete guard keeps cache/ and teachers/ out of every teardown."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tpu"))

from gcs_protect import BUCKET, Refused, check, main  # noqa: E402

B = f"gs://{BUCKET}"


@pytest.mark.parametrize("url", [
    f"{B}/plan90/", f"{B}/plan90/**", f"{B}/plan25k1/claims/pilot", f"{B}/plan65k1long/ledger/x.json",
    f"{B}/plan90/teachers/x",
])
def test_plan_prefixes_allowed(url):
    check(url)


@pytest.mark.parametrize("url", [
    B, f"{B}/", f"{B}/**", f"{B}/*", f"{B}/*/", f"{B}/plan*/x", f"{B}/cache", f"{B}/cache/",
    f"{B}/cache/vllm_root/x", f"{B}/teachers/", f"{B}/teachers/qwen3.5-2b/math-x/model.safetensors",
    f"{B}//teachers/x", f"{B}/plan90/../teachers/x", f"{B}/plan90", "/tmp/x",
])
def test_kept_and_broad_refused(url):
    with pytest.raises(Refused):
        check(url)


def test_cli_refuses_without_running_gcloud(capsys):
    assert main(["rm", "-r", f"{B}/teachers/"]) == 2
    assert main(["rm", "-r", f"{B}/plan90/", f"{B}/cache/"]) == 2
    assert main(["check", f"{B}/plan90/"]) == 0
    assert main(["hold", f"{B}/plan90/"]) == 2
