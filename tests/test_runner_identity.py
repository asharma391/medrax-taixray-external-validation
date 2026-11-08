import importlib.util
from pathlib import Path
import sys


def _runner_module():
    script = Path(__file__).parents[1] / "scripts" / "run_additional_external_benchmarks.py"
    spec = importlib.util.spec_from_file_location("external_benchmark_runner_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeAdapter:
    model_name = "densenet121-res224-all"
    input_maxval = None

    def __init__(self, device: str):
        self.device = device

    @staticmethod
    def model_file_sha256() -> str:
        return "a" * 64


def test_resume_identity_binds_device_and_local_execution_sources(monkeypatch):
    runner = _runner_module()
    monkeypatch.setattr(
        runner,
        "runtime_provenance",
        lambda: {"torchxrayvision": "1", "torch": "2", "torchvision": "3"},
    )

    cpu_identity, _ = runner._adapter_resume_identity(_FakeAdapter("cpu"))
    mps_identity, _ = runner._adapter_resume_identity(_FakeAdapter("mps"))

    assert cpu_identity["resolved_device"] == "cpu"
    assert cpu_identity["input_maxval"] == "None"
    assert len(cpu_identity["adapter_source_sha256"]) == 64
    assert len(cpu_identity["runner_source_sha256"]) == 64
    assert runner._fingerprint({"adapter": cpu_identity}) != runner._fingerprint(
        {"adapter": mps_identity}
    )
