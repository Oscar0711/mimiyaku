import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from omni2.config.settings import Settings  # noqa: E402
from omni2.core import hardware, paths  # noqa: E402
from omni2.core.hardware import Gpu, Hardware  # noqa: E402
from omni2.models import catalog, download, recommend  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    return tmp_path


def gpu_hw(gb: float, cuda: float = 13.4, cc: float = 8.9, ram: float = 32) -> Hardware:
    return Hardware(gpus=[Gpu("Test GPU", int(gb * 1024), cc)], driver_cuda=cuda, ram_gb=ram)


# ---- 硬體偵測 ----

def test_parse_nvidia_smi():
    gpus = hardware.parse_gpus("NVIDIA GeForce RTX 5060 Ti, 16311, 12.0\nNVIDIA GeForce GT 710, 2048, [N/A]\n")
    assert [(g.name, g.vram_mb, g.compute_cap) for g in gpus] == [
        ("NVIDIA GeForce RTX 5060 Ti", 16311, 12.0), ("NVIDIA GeForce GT 710", 2048, 0.0)]
    assert Hardware(gpus=gpus).gpu.name.endswith("5060 Ti")
    assert hardware.parse_driver_cuda("| NVIDIA-SMI 617.14   KMD Version: 617.14   CUDA UMD Version: 13.4 |") == 13.4
    assert hardware.parse_driver_cuda("| NVIDIA-SMI 551.86   Driver Version: 551.86   CUDA Version: 12.4 |") == 12.4
    assert hardware.parse_driver_cuda("") == 0.0


# ---- 推薦 ----

@pytest.mark.parametrize("gb, translate, live_translate, file_stt", [
    (16, "hy-mt2-7b-q6", "", "qwen3-asr-1.7b"),
    (10, "hy-mt2-7b-q6", "hy-mt2-7b-q4", "qwen3-asr-1.7b"),
    (8, "hy-mt2-7b-q4", "hy-mt2-1.8b-q8", "qwen3-asr-1.7b"),
    (6, "hy-mt2-1.8b-q8", "", "qwen3-asr-0.6b"),
    (4, "hy-mt2-1.8b-q4", "", "whisper-turbo"),
])
def test_recommend_tiers(gb, translate, live_translate, file_stt):
    r = recommend.recommend(gpu_hw(gb))
    assert (r.translate_model, r.live_translate_model, r.file_stt_id) == (translate, live_translate, file_stt)
    assert r.device == "cuda" and "llama-server" in r.models()
    live_tr = r.live_translate_model or r.translate_model
    assert recommend.VRAM[r.live_stt_id] + recommend.VRAM[live_tr] < gb - 0.5  # 即時模式兩個模型放得下


def test_recommend_without_gpu_or_with_old_driver():
    cpu = recommend.recommend(Hardware(ram_gb=8))
    assert cpu.device == "cpu" and cpu.gpu_layers == 0 and cpu.file_engine == "faster-whisper"
    assert any("CPU" in w for w in cpu.warnings) and any("記憶體" in w for w in cpu.warnings)
    old = recommend.recommend(gpu_hw(12, cuda=12.4))
    assert old.file_engine == old.live_engine == "faster-whisper"
    assert "ecapa" not in old.models() and any("驅動程式" in w for w in old.warnings)


def test_apply_keeps_cloud_engine_and_reports_changes():
    s = Settings()
    s.stt.engine = "funasr"
    r = recommend.recommend(gpu_hw(8))
    changes = dict((label, new) for label, _old, new in r.changes(s))
    assert "檔案翻譯：辨識引擎" not in changes  # 雲端引擎不動
    assert changes["翻譯模型"] == "Hy-MT2-7B-Q4_K_M.gguf"
    assert changes["即時翻譯：翻譯模型"] == "Hy-MT2-1.8B-Q8_0.gguf"
    r.apply(s)
    assert s.stt.engine == "funasr" and s.live.stt_engine == "qwen3-asr"
    assert Path(s.live.translate_model_path).name == "Hy-MT2-1.8B-Q8_0.gguf"
    assert r.changes(s) == []


def test_models_in_use_and_missing(env):
    s = Settings()
    recommend.recommend(gpu_hw(16)).apply(s)
    assert recommend.models_in_use(s) >= {"qwen3-asr-1.7b", "qwen3-aligner", "qwen3-asr-0.6b", "hy-mt2-7b-q6",
                                          "llama-server", "ecapa"}
    assert "separator" not in recommend.models_in_use(s)
    missing = recommend.missing_for(s, live=False, translate=True)
    assert any("llama.cpp" in m for m in missing) and any("HY-MT2-7B-Q6_K" in m for m in missing)
    assert recommend.missing_for(s, live=False, translate=False) == []
    s.translate.endpoint = "http://localhost:11434/v1"
    assert recommend.missing_for(s, live=True, translate=True) == []


def test_reset_settings_keeps_storage_cloud_and_overlay():
    s = Settings()
    s.storage.data_dir = "D:/omni"
    s.stt.cloud.google_project = "proj"
    s.translate.server_exe = "C:/llama/llama-server.exe"
    s.live.overlay_geometry = [1, 2, 3, 4]
    s.live.font_size = 40
    s.layout.max_lines = 3
    recommend.reset_settings(s, gpu_hw(16))
    assert s.storage.data_dir == "D:/omni" and s.stt.cloud.google_project == "proj"
    assert s.translate.server_exe == "C:/llama/llama-server.exe" and s.live.overlay_geometry == [1, 2, 3, 4]
    assert s.live.font_size == 26 and s.layout.max_lines == 2 and s.first_run_done
    assert Path(s.translate.model_path).name == "HY-MT2-7B-Q6_K.gguf"


# ---- 目錄：狀態、刪除 ----

def _fake_repo(repo: str, files: dict[str, str], incomplete: bool = False) -> Path:
    root = catalog.repo_dir(repo)
    snap = root / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for name, text in files.items():
        (snap / name).write_text(text)
    (root / "blobs").mkdir()
    if incomplete:
        (root / "blobs" / "123.incomplete").write_text("x")
    return root


def test_repo_status(env):
    m = catalog.get("qwen3-asr-0.6b")
    assert not catalog.is_installed(m)
    _fake_repo(m.repo, {"config.json": "{}", "model.safetensors": "w" * 100})
    assert catalog.is_installed(m) and catalog.installed_size(m) == 102

    sharded = catalog.get("qwen3-asr-1.7b")
    index = {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
    root = _fake_repo(sharded.repo, {"model.safetensors.index.json": json.dumps(index), "model-1.safetensors": "1"})
    assert not catalog.is_installed(sharded)  # 少一份權重
    (root / "snapshots" / "abc" / "model-2.safetensors").write_text("2")
    assert catalog.is_installed(sharded)

    partial = catalog.get("whisper-turbo")
    _fake_repo(partial.repo, {"model.bin": "w"}, incomplete=True)
    assert not catalog.is_installed(partial)

    catalog.delete(m)
    assert not catalog.repo_dir(m.repo).exists()


def test_file_and_llama_status(env):
    m = catalog.get("hy-mt2-1.8b-q4")
    assert catalog.find_file("C:/x/hy-mt2-1.8b-q4_k_m.GGUF") is m
    catalog.file_path(m).write_text("gguf")
    assert catalog.is_installed(m)
    llama = catalog.get("llama-server")
    assert not catalog.is_installed(llama)
    for tag in ("b900", "b11853", "custom"):
        d = catalog.llama_root() / tag
        d.mkdir(parents=True)
        (d / "llama-server.exe").write_text(tag)
    assert [v.name for v in catalog.llama_versions()][:2] == ["b11853", "b900"]  # 依版本號，不是字母順序
    assert catalog.installed_version(llama) == "b11853"
    assert catalog.delete_old_llama() > 0
    assert [v.name for v in catalog.llama_versions()] == ["b11853"]


# ---- 下載 ----

class _RangeHandler(SimpleHTTPRequestHandler):
    """支援 Range 的靜態檔案伺服器（http.server 內建的不支援續傳）。"""

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_fetch_resumes_partial_download(server, tmp_path):
    root, base = server
    payload = bytes(range(256)) * 4096  # 1 MB
    (root / "model.gguf").write_bytes(payload)
    dest = tmp_path / "out" / "model.gguf"
    part = dest.with_name("model.gguf.part")
    part.parent.mkdir()
    part.write_bytes(payload[:300_000])  # 上次下載到一半
    seen = []
    download.fetch(f"{base}/model.gguf", dest, lambda have, total: seen.append((have, total)))
    assert dest.read_bytes() == payload and not part.exists()
    assert seen[0][0] > 300_000 and seen[-1] == (len(payload), len(payload))

    again = tmp_path / "out" / "again.gguf"
    again.with_name("again.gguf.part").write_bytes(payload)  # 其實已經下載完，只差改名
    download.fetch(f"{base}/model.gguf", again)
    assert again.read_bytes() == payload


def test_pick_llama_assets():
    def asset(name):
        return {"name": name, "size": 1, "browser_download_url": f"https://x/{name}"}

    releases = [
        {"tag_name": "b200", "assets": [asset("llama-b200-bin-macos-arm64.zip")]},  # 還沒建好 Windows 版
        {"tag_name": "b199", "assets": [asset(n) for n in (
            "llama-b199-bin-win-cuda-12.4-x64.zip", "llama-b199-bin-win-cuda-13.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip",
            "llama-b199-bin-win-cuda-13.4-arm64.zip", "llama-b199-bin-win-cpu-x64.zip")]},
    ]
    tag, picked = download.pick_llama_assets(releases, 13.4)
    assert tag == "b199" and [a["name"] for a in picked] == [
        "llama-b199-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]
    _, picked = download.pick_llama_assets(releases, 12.8)  # 舊驅動
    assert picked[0]["name"] == "llama-b199-bin-win-cuda-12.4-x64.zip"
    _, picked = download.pick_llama_assets(releases, 0.0)  # 沒有 NVIDIA
    assert [a["name"] for a in picked] == ["llama-b199-bin-win-cpu-x64.zip"]
    with pytest.raises(RuntimeError):
        download.pick_llama_assets(releases, 11.0)


# ---- 捷徑 ----

def test_create_shortcut(tmp_path):
    from omni2.core import shortcut

    path = shortcut.create(tmp_path)
    assert path.exists() and path.name == "Mimiyaku.lnk"
    assert shortcut.icon_path().exists()


# ---- 介面 ----

@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_setup_dialog_lists_missing_models(app, env):
    from omni2.ui.setup_dialog import SetupDialog, needs_setup

    s = Settings()
    rec = recommend.recommend(gpu_hw(8))
    assert needs_setup(s, rec)
    catalog.file_path(catalog.get("hy-mt2-7b-q4")).write_text("gguf")  # 這個已經下載了
    dialog = SetupDialog(s, gpu_hw(8), rec)
    chosen = dialog.chosen()
    assert "hy-mt2-7b-q4" not in chosen and {"hy-mt2-1.8b-q8", "llama-server", "qwen3-asr-1.7b"} <= set(chosen)
    assert "共約" in dialog.total.text()


def test_models_panel_apply_and_queue(app, env, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from omni2.ui import models_page as mp

    monkeypatch.setattr(mp, "save_settings", lambda s: None)
    monkeypatch.setattr(mp.ModelsPanel, "detect", lambda self: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    queued = []
    monkeypatch.setattr(mp.ModelsPanel, "download", lambda self, ids: queued.extend(ids))
    s = Settings()
    panel = mp.ModelsPanel(s)
    assert panel.table.rowCount() == len(catalog.CATALOG)
    changed = []
    panel.settings_changed.connect(lambda: changed.append(True))
    panel.set_hardware(gpu_hw(8))
    panel.apply_recommendation()
    assert Path(s.translate.model_path).name == "Hy-MT2-7B-Q4_K_M.gguf" and changed
    assert "hy-mt2-1.8b-q8" in queued and "llama-server" in queued


def test_settings_page_sections_and_translate_choice(app, env, monkeypatch):
    from omni2.ui import models_page as mp
    from omni2.ui import settings_page as sp

    for mod in (sp, mp):
        monkeypatch.setattr(mod, "save_settings", lambda s: None)
    monkeypatch.setattr(sp.SettingsPage, "refresh_usage", lambda self: None)
    monkeypatch.setattr(mp.ModelsPanel, "detect", lambda self: None)
    s = Settings()
    page = sp.SettingsPage(s)
    assert [name for name, _ in page.sections][:2] == ["硬體與模型", "辨識與翻譯"]
    page.show_section("其他")
    assert page.stack.currentIndex() == 5
    labels = [page.translate_model.itemText(i) for i in range(page.translate_model.count())]
    assert any("Hy-MT2 7B Q6_K（未下載）" in t for t in labels)
    q4 = page.live_translate_model.findData(str(catalog.file_path(catalog.get("hy-mt2-7b-q4"))))
    page.live_translate_model.setCurrentIndex(q4)
    page._pick_translate(page.live_translate_model)
    assert Path(s.live.translate_model_path).name == "Hy-MT2-7B-Q4_K_M.gguf"
    page.line_width.setValue(16)
    assert s.layout.max_line_width == 32
    page.set_busy(True)
    assert not page.reset_button.isEnabled() and page.models.busy


def test_feedback_issue_url_has_environment_but_no_paths():
    from urllib.parse import parse_qs, urlparse

    from omni2.core import feedback

    s = Settings()
    s.translate.model_path = "C:/Users/someone/secret/HY-MT2-7B-Q6_K.gguf"
    url = feedback.issue_url("bug", s, gpu_hw(16))
    query = parse_qs(urlparse(url).query)
    assert url.startswith(feedback.REPO_URL + "/issues/new?")
    assert query["template"] == ["bug_report.yml"]
    env = query["environment"][0]
    assert "Test GPU（16 GB）" in env and "HY-MT2-7B-Q6_K.gguf" in env and "someone" not in env
    assert "environment" not in feedback.issue_url("feature", s, None)


def test_progress_signals_carry_sizes_over_2gb(app):
    from omni2.ui.models_page import DownloadQueue
    from omni2.ui.tasks import Task

    big, total = 6_163_000_000, 7_000_000_000  # 超過 32 位元整數的上限
    got = []
    queue = DownloadQueue(13.4)
    queue.progress.connect(lambda *args: got.append(args))
    queue.progress.emit("hy-mt2-7b-q6", big, total, "")
    task = Task(lambda t: None)
    task.progress.connect(lambda *args: got.append(args))
    task.progress.emit(big, total, "x")
    assert got == [("hy-mt2-7b-q6", big, total, ""), (big, total, "x")]


def test_app_dir_keeps_using_old_folder(tmp_path, monkeypatch):
    from omni2.core import paths as p

    monkeypatch.setattr(p, "user_data_dir", lambda name, **kw: str(tmp_path / name))
    assert p.app_dir() == tmp_path / "Mimiyaku"  # 新安裝
    (tmp_path / "Mimiyaku").rmdir()
    (tmp_path / "OmniTranslate2").mkdir()
    assert p.app_dir() == tmp_path / "OmniTranslate2"  # 改名前就裝了：沿用，模型和設定不用搬


def test_shortcut_replaces_old_name(tmp_path):
    from omni2.core import shortcut

    (tmp_path / "OmniTranslate.lnk").write_bytes(b"old")
    shortcut.create(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Mimiyaku.lnk"]
    assert shortcut.launch_command()[0].endswith("mimiyaku-gui.exe")
