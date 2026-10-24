import os

import pytest

from omni2.config.settings import Settings
from omni2.core import paths, storage


@pytest.fixture
def env(tmp_path, monkeypatch):
    old, hf = tmp_path / "old", tmp_path / "hf"
    for rel, data in {"jobs/j1/audio.wav": b"a" * 100, "models/m.gguf": b"m" * 300,
                      "bin/llama.cpp/b1/llama-server.exe": b"x" * 50}.items():
        (old / rel).parent.mkdir(parents=True, exist_ok=True)
        (old / rel).write_bytes(data)
    (hf / "hub/models--Qwen--x").mkdir(parents=True)
    (hf / "hub/models--Qwen--x/w.bin").write_bytes(b"q" * 200)
    monkeypatch.setenv(paths.DATA_ENV, str(old))
    monkeypatch.setenv("OMNI2_DEFAULT_HF_HOME", str(hf))
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.setattr(storage, "save_settings", lambda s: None)
    settings = Settings()
    settings.storage.data_dir = str(old)
    settings.translate.model_path = str(old / "models/m.gguf")
    settings.translate.server_exe = str(old / "bin/llama.cpp/b1/llama-server.exe")
    settings.terms.model_path = r"D:\elsewhere\other.gguf"
    return settings, old, hf, tmp_path / "new"


def test_relocate_moves_data_and_rewrites_paths(env):
    settings, old, hf, new = env
    events = []
    storage.relocate(settings, new, hf_in_data_dir=True, progress=lambda d, t, f: events.append((d, t)))
    assert (new / "jobs/j1/audio.wav").read_bytes() == b"a" * 100
    assert (new / "models/m.gguf").exists() and not (old / "models").exists()
    assert (new / "huggingface/hub/models--Qwen--x/w.bin").exists() and not (hf / "hub").exists()
    assert settings.translate.model_path == str(new / "models/m.gguf")
    assert settings.translate.server_exe == str(new / "bin/llama.cpp/b1/llama-server.exe")
    assert settings.terms.model_path == r"D:\elsewhere\other.gguf"  # 不在資料夾裡的不動
    assert settings.storage.data_dir == str(new.resolve()) and settings.storage.hf_in_data_dir
    assert os.environ[paths.DATA_ENV] == str(new.resolve())
    assert os.environ["HF_HOME"] == str(new.resolve() / "huggingface")
    assert events and events[-1][0] == events[-1][1] == 650


def test_relocate_back_restores_default_hf(env):
    settings, old, hf, new = env
    storage.relocate(settings, new, hf_in_data_dir=True)
    storage.relocate(settings, old, hf_in_data_dir=False)
    assert (hf / "hub/models--Qwen--x/w.bin").exists()
    assert (old / "models/m.gguf").exists()
    assert os.environ["HF_HOME"] == str(hf)


def test_cannot_move_into_itself(env):
    settings, old, _, _ = env
    with pytest.raises(ValueError):
        storage.relocate(settings, old / "jobs" / "inside", hf_in_data_dir=False)


def test_usage_and_clear_jobs(env):
    settings, old, _, _ = env
    sizes = {u.label: u.size for u in storage.usage(settings)}
    assert sizes["工作檔（中間音訊、斷點）"] == 100 and sizes["翻譯與其他模型"] == 300
    assert storage.clear_jobs() == 100
    assert not any((old / "jobs").iterdir())
    assert storage.human(1536) == "1.5 KB" and storage.human(3 * 1024**3) == "3.0 GB"
