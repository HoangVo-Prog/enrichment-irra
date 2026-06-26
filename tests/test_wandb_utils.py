import shutil
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

from utils.checkpoint import delete_output_checkpoints
from utils.wandb_utils import (
    build_wandb_config,
    get_wandb_project,
    upload_best_checkpoint_artifact,
    upload_checkpoint_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeRun:
    def __init__(self, name="run+name"):
        self.name = name
        self.summary = {}
        self.logged_artifacts = []

    def log_artifact(self, artifact, aliases=None):
        self.logged_artifacts.append((artifact, aliases))


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, _message):
        pass


def _make_tmp_output_dir(prefix):
    tmp_path = ROOT / "tests_tmp" / f"{prefix}_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    return tmp_path


def test_build_wandb_config_uses_cli_project_name():
    args = SimpleNamespace(wandb_project="custom-project", use_wandb=True)

    config = build_wandb_config(args, run_name="run-1", output_dir="logs/run-1")

    assert config["wandb_project"] == "custom-project"
    assert config["wandb_run_name"] == "run-1"
    assert config["output_dir"] == "logs/run-1"


def test_wandb_project_falls_back_for_legacy_args():
    assert get_wandb_project(SimpleNamespace()) == "enrichment"


def test_upload_best_checkpoint_artifact_logs_best_file(monkeypatch):
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run = FakeRun(name="20260529_010203_IRRA")
    tmp_path = _make_tmp_output_dir("wandb")
    try:
        checkpoint_name = "best.pth"
        (tmp_path / checkpoint_name).write_text("checkpoint", encoding="utf-8")

        artifact = upload_best_checkpoint_artifact(
            run,
            tmp_path,
            checkpoint_name=checkpoint_name,
        )

        assert artifact is not None
        assert artifact.name == "20260529_010203_IRRA-best"
        assert artifact.type == "model"
        assert artifact.files[0][1] == checkpoint_name
        assert run.logged_artifacts[0][1] == ["best", "latest"]
        assert run.summary["best_checkpoint_artifact"] == artifact.name
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_upload_best_checkpoint_artifact_uploads_config_yaml(monkeypatch):
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run = FakeRun(name="demo-run")
    tmp_path = _make_tmp_output_dir("wandb")
    try:
        (tmp_path / "best.pth").write_text("checkpoint", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("foo: bar\n", encoding="utf-8")

        artifact = upload_best_checkpoint_artifact(run, tmp_path)

        assert artifact is not None
        uploaded_names = [name for _, name in artifact.files]
        assert "best.pth" in uploaded_names
        assert "config.yaml" in uploaded_names
        assert artifact.metadata["config"] == "config.yaml"
        assert "best_checkpoint_config_path" in run.summary
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_upload_best_checkpoint_artifact_skips_missing_file():
    run = FakeRun()
    logger = FakeLogger()

    artifact = upload_best_checkpoint_artifact(
        run,
        ROOT,
        logger=logger,
        checkpoint_name="missing-best.pth",
    )

    assert artifact is None
    assert run.logged_artifacts == []
    assert "not found" in logger.warnings[0]


def test_upload_checkpoint_artifacts_logs_all_run_checkpoints(monkeypatch):
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run = FakeRun(name="demo-run")
    tmp_path = _make_tmp_output_dir("wandb")
    try:
        (tmp_path / "best.pth").write_text("best", encoding="utf-8")
        (tmp_path / "epoch_2.pth").write_text("epoch", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("not a checkpoint", encoding="utf-8")

        artifacts = upload_checkpoint_artifacts(run, tmp_path)

        assert artifacts is not None
        assert len(artifacts) == 2
        uploaded_checkpoints = [artifact.metadata["checkpoint"] for artifact in artifacts]
        assert uploaded_checkpoints == ["best.pth", "epoch_2.pth"]
        assert run.logged_artifacts[0][1] == ["best", "latest"]
        assert run.logged_artifacts[1][1] == ["epoch_2"]
        assert run.summary["checkpoint_artifact_count"] == 2
        assert run.summary["checkpoint_artifacts"] == ["demo-run-best", "demo-run-epoch_2"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_delete_output_checkpoints_removes_only_direct_pth_files():
    tmp_path = _make_tmp_output_dir("cleanup")
    try:
        nested_path = tmp_path / "nested"
        nested_path.mkdir()
        (tmp_path / "best.pth").write_text("best", encoding="utf-8")
        (tmp_path / "epoch_2.pth").write_text("epoch", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("foo: bar\n", encoding="utf-8")
        (nested_path / "nested.pth").write_text("nested", encoding="utf-8")

        deleted = delete_output_checkpoints(tmp_path)

        assert [path.name for path in deleted] == ["best.pth", "epoch_2.pth"]
        assert not (tmp_path / "best.pth").exists()
        assert not (tmp_path / "epoch_2.pth").exists()
        assert (tmp_path / "config.yaml").exists()
        assert (nested_path / "nested.pth").exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
