import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from omni2.config.settings import Settings  # noqa: E402
from omni2.core.types import Segment  # noqa: E402
from omni2.jobs.job import job_id  # noqa: E402
from omni2.translate.terms import TermsResult  # noqa: E402
from omni2.ui import workspace as ws  # noqa: E402
from omni2.ui.player import PlayerPanel, segment_at  # noqa: E402
from omni2.ui.terms_dialog import TermsDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(app, monkeypatch):
    monkeypatch.setattr(ws, "save_settings", lambda s: None)  # 不要動到使用者的設定檔
    return ws.FileWorkspace(Settings())


def test_terms_dialog_round_trip(app):
    result = TermsResult(summary="摘要", terms=[
        {"source": "田中", "target": "田中", "type": "人名", "note": "同事"},
        {"source": "ゲームボス", "target": "遊戲頭目", "type": "作品", "note": ""},
    ])
    dialog = TermsDialog("會議", result)
    dialog.table.item(1, 1).setText("魔王")
    dialog._append({"source": "", "target": "沒有原文的列會被丟掉"})
    dialog.summary.setPlainText("改過的摘要")
    out = dialog.result_terms()
    assert out.summary == "改過的摘要"
    assert [(t["source"], t["target"], t["type"]) for t in out.terms] == [
        ("田中", "田中", "人名"), ("ゲームボス", "魔王", "作品")]


def test_segment_at():
    segs = [Segment(1.0, 2.0, "a"), Segment(3.0, 5.0, "b")]
    starts = [s.start for s in segs]
    assert segment_at(starts, segs, 0.5) is None
    assert segment_at(starts, segs, 1.5) == 0
    assert segment_at(starts, segs, 2.5) is None
    assert segment_at(starts, segs, 4.9) == 1


def test_player_shows_translation_and_highlights_row(app):
    player = PlayerPanel()
    segs = [Segment(0.0, 1.0, "こんにちは"), Segment(1.5, 3.0, "さようなら")]
    player.load("測試", None, segs, ["你好", None])
    assert player.table.rowCount() == 2
    player._on_position(2000)
    assert player.primary.text() == "さようなら"  # 沒有譯文時顯示原文
    assert player.table.currentRow() == 1
    player._on_position(500)
    assert (player.primary.text(), player.secondary.text()) == ("你好", "こんにちは")


def test_add_inputs_skips_duplicates_and_missing(workspace, tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"x")
    assert workspace.add_inputs([str(a), str(a), str(tmp_path / "missing.wav"), "https://example.com/v"]) == 2
    assert workspace.queue.topLevelItemCount() == 2
    assert workspace.start_button.isEnabled()


def test_events_update_rows(workspace, tmp_path):
    a = tmp_path / "a.wav"
    a.write_bytes(b"x")
    workspace.add_inputs([str(a)])
    jid = job_id(str(a))
    entry = workspace.entries[jid]
    workspace._on_event({"type": "stage", "id": jid, "job": "a", "stage": "stt", "status": "progress", "value": 0.42})
    assert (entry.item.text(ws.COL_STATE), entry.bar.value()) == ("辨識", 42)
    workspace._on_event({"type": "stage", "id": jid, "job": "a", "stage": "translate", "status": "error",
                         "message": "llama-server 啟動失敗"})
    assert entry.state == "error"
    assert entry.item.text(ws.COL_MESSAGE) == "llama-server 啟動失敗"
    workspace._on_event({"type": "stage", "id": jid, "job": "a", "stage": "export", "status": "done",
                         "message": "a.zh-TW.srt"})
    assert (entry.state, entry.item.text(ws.COL_STATE), entry.bar.value()) == ("done", "完成", 100)


def test_options_become_job_spec(workspace):
    workspace.src_combo.setCurrentIndex(workspace.src_combo.findData("ko"))
    workspace.translate_check.setChecked(True)
    workspace.terms_check.setChecked(True)
    workspace.diarize_check.setChecked(True)
    workspace.speakers_spin.setValue(3)
    for name, check in workspace.format_checks.items():
        check.setChecked(name in ("srt", "md"))
    workspace._save_options()
    spec = workspace.make_spec("x.wav")
    assert (spec.src_lang, spec.translate, spec.terms_first, spec.diarize, spec.num_speakers) == (
        "ko", True, True, True, 3)
    assert spec.formats == ["srt", "md"]
    workspace.translate_check.setChecked(False)
    workspace._save_options()
    assert workspace.make_spec("x.wav").terms_first is False  # 不翻譯就不需要術語


def test_settings_page_saves_diarization_choice(app, monkeypatch):
    from omni2.ui import settings_page as sp

    saved = []
    monkeypatch.setattr(sp, "save_settings", lambda s: saved.append(s))
    monkeypatch.setattr(sp.SettingsPage, "refresh_usage", lambda self: None)
    _fake_keyring(monkeypatch)
    settings = Settings()
    page = sp.SettingsPage(settings)
    page.engine_combo.setCurrentIndex(page.engine_combo.findData("pyannote"))
    assert settings.diarization.engine == "pyannote"
    assert not page.sensitivity_combo.isEnabled()
    page.engine_combo.setCurrentIndex(page.engine_combo.findData("ecapa"))
    page.sensitivity_combo.setCurrentIndex(1)  # 較敏感
    assert settings.diarization.threshold == pytest.approx(0.6)
    assert saved
    page.set_busy(True)
    assert not page.apply_button.isEnabled() and not page.engine_combo.isEnabled()


def test_output_folder_option(workspace, tmp_path, monkeypatch):
    assert workspace.make_spec("x.wav").out_dir is None  # 預設和原檔相同
    monkeypatch.setattr(ws.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    workspace.out_combo.setCurrentIndex(1)  # 指定資料夾：沒選過就跳出對話框
    assert workspace.out_edit.text() == str(tmp_path)
    workspace._save_options()
    assert workspace.make_spec("x.wav").out_dir == str(tmp_path)
    workspace.out_combo.setCurrentIndex(0)
    workspace._save_options()
    assert workspace.make_spec("x.wav").out_dir is None



def _fake_keyring(monkeypatch):
    from omni2.core import secrets

    store = {}
    fake = type("K", (), {
        "get_password": lambda self, s, n: store.get(n),
        "set_password": lambda self, s, n, v: store.__setitem__(n, v),
        "delete_password": lambda self, s, n: store.pop(n),
    })()
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    for env in secrets.ENV_VARS.values():
        monkeypatch.delenv(env, raising=False)
    return store


def test_settings_page_saves_key_and_finds_region(app, monkeypatch):
    from omni2.stt import funasr_engine
    from omni2.ui import settings_page as sp

    monkeypatch.setattr(sp, "save_settings", lambda s: None)
    monkeypatch.setattr(sp.SettingsPage, "refresh_usage", lambda self: None)
    store = _fake_keyring(monkeypatch)
    tried = []

    def fake_check(key, region, model):
        tried.append(region)
        return (region == "cn", "ok" if region == "cn" else "InvalidApiKey")

    monkeypatch.setattr(funasr_engine, "check_key", fake_check)
    # 背景執行緒改成同步執行，測試比較單純
    monkeypatch.setattr(sp.SettingsPage, "_run", lambda self, fn, done, *a, **k: done(fn(None)))
    settings = Settings()
    page = sp.SettingsPage(settings)
    assert page.key_status["dashscope"].text() == "尚未設定"
    page.key_edits["dashscope"].setText("sk-secret")
    page._test_dashscope()
    assert store == {"dashscope": "sk-secret"}
    assert page.key_edits["dashscope"].text() == ""  # 存完就清掉，畫面上不留金鑰
    assert tried == ["intl", "cn"]
    assert settings.stt.cloud.dashscope_region == "cn"
    assert page.key_status["dashscope"].text().startswith("✓")
