"""依硬體推薦模型組合。

檔案翻譯每個階段輪流用顯示卡（辨識完才載入翻譯模型），所以挑「單一模型放得下」的最好版本；
即時翻譯要同時載入辨識和翻譯模型，兩者加起來要放得下，必要時翻譯改用較小的模型。
顯示記憶體用量：Qwen3-ASR 0.6B、Hy-MT2 7B Q6_K、1.8B Q4_K_M 是實測，其餘是依檔案大小估計的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter

from omni2.config.settings import Settings
from omni2.core.hardware import Hardware
from omni2.models import catalog
from omni2.stt.language_models import KO_MODEL

CLOUD_ENGINES = {"funasr", "qwen-realtime", "openai", "google"}
WHISPER_IDS = {"large-v3-turbo": "whisper-turbo", "small": "whisper-small"}
QWEN_IDS = {"Qwen/Qwen3-ASR-1.7B": "qwen3-asr-1.7b", "Qwen/Qwen3-ASR-0.6B": "qwen3-asr-0.6b"}
# 執行時的顯示記憶體（GB），含 KV cache
VRAM = {"qwen3-asr-0.6b": 2.0, "qwen3-asr-1.7b-ko": 4.5, "whisper-turbo": 2.3, "whisper-small": 1.0,
        "hy-mt2-7b-q6": 7.1, "hy-mt2-7b-q4": 5.7, "hy-mt2-1.8b-q8": 2.6, "hy-mt2-1.8b-q4": 1.8}


@dataclass
class Recommendation:
    tier: str
    file_engine: str = "qwen3-asr"
    file_qwen: str = "Qwen/Qwen3-ASR-1.7B"
    file_whisper: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    translate_model: str = "hy-mt2-7b-q6"
    gpu_layers: int = 99
    live_engine: str = "qwen3-asr"
    live_qwen: str = "Qwen/Qwen3-ASR-0.6B"
    live_whisper: str = "large-v3-turbo"
    live_translate_model: str = ""  # 空字串＝和檔案翻譯用同一個
    # 來源語言是韓文時改用的模型（下載了才會用）；空字串＝不換
    file_qwen_ko: str = ""
    live_qwen_ko: str = ""
    torch_ok: bool = True  # 能不能跑 PyTorch（CUDA 13）的模型：Qwen3-ASR、講者分離、人聲分離
    warnings: list[str] = field(default_factory=list)

    # ---- 需要的模型 ----

    def _stt_id(self, engine: str, qwen: str, whisper: str) -> str:
        return QWEN_IDS.get(qwen, "qwen3-asr-1.7b") if engine == "qwen3-asr" else WHISPER_IDS.get(whisper, "")

    @property
    def file_stt_id(self) -> str:
        return self._stt_id(self.file_engine, self.file_qwen, self.file_whisper)

    @property
    def live_stt_id(self) -> str:
        return self._stt_id(self.live_engine, self.live_qwen, self.live_whisper)

    def models(self) -> list[str]:
        ids = [self.file_stt_id]
        if self.file_engine == "qwen3-asr":
            ids.append("qwen3-aligner")
        ids += [self.live_stt_id, self.translate_model, self.live_translate_model, "llama-server"]
        if self.torch_ok:
            ids.append("ecapa")
        out: list[str] = []
        for i in ids:
            if i and i not in out:
                out.append(i)
        return out

    def describe(self) -> list[str]:
        name = lambda i: catalog.get(i).label  # noqa: E731
        live_tr = self.live_translate_model or self.translate_model
        live_vram = VRAM.get(self.live_stt_id, 0) + VRAM.get(live_tr, 0)
        where = "CPU" if self.device == "cpu" else "顯示卡"
        return [
            f"檔案翻譯：{name(self.file_stt_id)} 辨識＋{name(self.translate_model)} 翻譯（在{where}上輪流執行）",
            f"即時翻譯：{name(self.live_stt_id)} 辨識＋{name(live_tr)} 翻譯"
            + (f"（同時載入，約 {live_vram:.0f} GB 顯示記憶體）" if self.device != "cpu" else ""),
        ]

    # ---- 套用到設定 ----

    def apply(self, settings: Settings) -> None:
        stt, live, tr = settings.stt, settings.live, settings.translate
        if stt.engine not in CLOUD_ENGINES:  # 使用者選了雲端引擎就不動
            stt.engine = self.file_engine
        stt.qwen_model = self.file_qwen
        stt.model = self.file_whisper
        stt.device = self.device
        stt.compute_type = self.compute_type
        tr.model_path = str(catalog.file_path(catalog.get(self.translate_model)))
        tr.gpu_layers = self.gpu_layers
        if live.stt_engine not in CLOUD_ENGINES:
            live.stt_engine = self.live_engine
        live.qwen_model = self.live_qwen
        live.whisper_model = self.live_whisper
        live.translate_model_path = (str(catalog.file_path(catalog.get(self.live_translate_model)))
                                     if self.live_translate_model else "")
        stt.qwen_model_ko = self.file_qwen_ko
        live.qwen_model_ko = self.live_qwen_ko

    def changes(self, settings: Settings) -> list[tuple[str, str, str]]:
        """套用後會改變的設定：（項目, 原本, 改成）。"""
        after = settings.model_copy(deep=True)
        self.apply(after)
        out = []
        for path, label in CHANGE_FIELDS:
            old, new = attrgetter(path)(settings), attrgetter(path)(after)
            if old != new:
                empty = "（和檔案翻譯相同）" if path == "live.translate_model_path" else "（不使用）"
                out.append((label, _show(old, empty), _show(new, empty)))
        return out


CHANGE_FIELDS = [
    ("stt.engine", "檔案翻譯：辨識引擎"), ("stt.qwen_model", "檔案翻譯：Qwen3-ASR 模型"),
    ("stt.model", "檔案翻譯：Whisper 模型"), ("stt.device", "辨識裝置"), ("stt.compute_type", "辨識精度"),
    ("translate.model_path", "翻譯模型"), ("translate.gpu_layers", "翻譯模型放進顯示卡的層數"),
    ("live.stt_engine", "即時翻譯：辨識引擎"), ("live.qwen_model", "即時翻譯：Qwen3-ASR 模型"),
    ("live.whisper_model", "即時翻譯：Whisper 模型"), ("live.translate_model_path", "即時翻譯：翻譯模型"),
    ("stt.qwen_model_ko", "檔案翻譯：韓文改用的模型"), ("live.qwen_model_ko", "即時翻譯：韓文改用的模型"),
]


def _show(value, empty: str = "（和檔案翻譯相同）") -> str:
    if value == "":
        return empty
    text = str(value)
    return text.rsplit("\\", 1)[-1] if text.lower().endswith(".gguf") else text


def recommend(hw: Hardware) -> Recommendation:
    gpu = hw.gpu
    if gpu is None or hw.driver_cuda <= 0:
        r = Recommendation("沒有 NVIDIA 顯示卡：用 CPU", file_engine="faster-whisper", file_whisper="small",
                           device="cpu", compute_type="int8", translate_model="hy-mt2-1.8b-q4", gpu_layers=0,
                           live_engine="faster-whisper", live_whisper="small")
        r.warnings.append("沒有偵測到 NVIDIA 顯示卡，辨識和翻譯都會用 CPU，速度很慢。"
                          "即時翻譯建議改用雲端的阿里雲 Fun-ASR 辨識。")
        _ram_warning(hw, r)
        return r

    vram = gpu.vram_gb
    if vram >= 11.5:
        r = Recommendation(f"{vram:.0f} GB 顯示記憶體：全部用最準的模型")
    elif vram >= 9.5:
        r = Recommendation(f"{vram:.0f} GB 顯示記憶體：即時翻譯改用 Q4 翻譯模型", live_translate_model="hy-mt2-7b-q4")
    elif vram >= 7:
        r = Recommendation(f"{vram:.0f} GB 顯示記憶體：翻譯用 Q4，即時翻譯用 1.8B 翻譯模型",
                           translate_model="hy-mt2-7b-q4", live_translate_model="hy-mt2-1.8b-q8")
    elif vram >= 5:
        r = Recommendation(f"{vram:.0f} GB 顯示記憶體：辨識和翻譯都用小模型", file_qwen="Qwen/Qwen3-ASR-0.6B",
                           translate_model="hy-mt2-1.8b-q8")
    else:
        r = Recommendation(f"{vram:.0f} GB 顯示記憶體：用最小的組合", file_engine="faster-whisper",
                           compute_type="int8_float16", translate_model="hy-mt2-1.8b-q4",
                           live_engine="faster-whisper", live_whisper="small")
        r.warnings.append("顯示記憶體很少，即時翻譯只能用 Whisper small 辨識，準確度較低；"
                          "建議即時翻譯改用雲端的阿里雲 Fun-ASR 辨識。")

    # Qwen3-ASR、人聲分離、講者分離都用 PyTorch（CUDA 13 版）：驅動要支援 CUDA 13，顯示卡要 Turing（7.5）以後
    if hw.driver_cuda < 13.0:
        r.warnings.append(f"顯示卡驅動程式只支援到 CUDA {hw.driver_cuda:g}。Qwen3-ASR、人聲分離、講者分離需要支援 "
                          "CUDA 13 的驅動程式，請更新 NVIDIA 驅動程式；在那之前先改用 faster-whisper 辨識。")
    elif 0 < gpu.compute_cap < 7.5:
        r.warnings.append(f"{gpu.name} 比較舊，PyTorch 不支援，Qwen3-ASR、人聲分離、講者分離不能用；改用 faster-whisper 辨識。")
    if hw.driver_cuda < 13.0 or 0 < gpu.compute_cap < 7.5:
        r.file_engine = r.live_engine = "faster-whisper"
        r.torch_ok = False
    # 韓文微調版和 1.7B 一樣大：檔案模式用 1.7B 時就能換；即時模式要和翻譯模型同時放得下
    if r.file_engine == "qwen3-asr" and r.file_qwen == "Qwen/Qwen3-ASR-1.7B":
        r.file_qwen_ko = KO_MODEL
    live_tr = r.live_translate_model or r.translate_model
    if r.live_engine == "qwen3-asr" and VRAM["qwen3-asr-1.7b-ko"] + VRAM[live_tr] <= vram - 3:
        r.live_qwen_ko = KO_MODEL
    _ram_warning(hw, r)
    return r


def _ram_warning(hw: Hardware, r: Recommendation) -> None:
    if 0 < hw.ram_gb < 12:
        r.warnings.append(f"記憶體只有 {hw.ram_gb:.0f} GB，處理長影片時可能不夠，建議 16 GB 以上。")


def models_in_use(settings: Settings) -> set[str]:
    """目前設定會用到的模型（目錄裡的 id）。"""
    def stt_ids(engine: str, qwen: str, whisper: str, aligner: bool) -> list[str]:
        if engine == "qwen3-asr":
            return [QWEN_IDS.get(qwen, "")] + (["qwen3-aligner"] if aligner else [])
        if engine == "faster-whisper":
            return [WHISPER_IDS.get(whisper, "")]
        return []

    s = settings
    ids = stt_ids(s.stt.engine, s.stt.qwen_model, s.stt.model, True)
    ids += stt_ids(s.live.stt_engine, s.live.qwen_model, s.live.whisper_model, False)
    if not s.translate.endpoint:
        for path in (s.translate.model_path, s.live.translate_model_path, s.terms.model_path):
            m = catalog.find_file(path) if path else None
            if m:
                ids.append(m.id)
        ids.append("llama-server")
    for engine, ko in ((s.stt.engine, s.stt.qwen_model_ko), (s.live.stt_engine, s.live.qwen_model_ko)):
        m = catalog.find_repo(ko) if engine == "qwen3-asr" and ko else None
        if m and catalog.is_installed(m):  # 只有下載了才會用到
            ids.append(m.id)
    if s.diarization.engine == "ecapa":
        ids.append("ecapa")
    if s.file_mode.separate_vocals:
        ids.append("separator")
    return {i for i in ids if i}


def missing_for(settings: Settings, live: bool, translate: bool) -> list[str]:
    """開始之前檢查翻譯需要的檔案。辨識模型不用檢查，第一次使用時會自動下載。"""
    from pathlib import Path

    tr = settings.translate
    if not translate or tr.endpoint:
        return []
    out = []
    if not tr.server_exe or not Path(tr.server_exe).exists():
        out.append("還沒有安裝 llama.cpp（翻譯模型的推論程式）")
    model = (settings.live.translate_model_path if live else "") or tr.model_path
    if not model:
        out.append("還沒有選擇翻譯模型")
    elif not Path(model).exists():
        out.append(f"找不到翻譯模型：{Path(model).name}")
    return out


def reset_settings(settings: Settings, hw: Hardware) -> None:
    """還原預設值（就地修改，其他頁面拿著的是同一個物件），再依硬體套用推薦的模型。

    保留：儲存位置、雲端服務的設定（金鑰本來就不在設定檔裡）、llama.cpp 位置、懸浮字幕的位置。
    """
    fresh = Settings()
    cloud, server = settings.stt.cloud, settings.translate.server_exe
    geometry = settings.live.overlay_geometry
    for name in type(settings).model_fields:
        if name != "storage":
            setattr(settings, name, getattr(fresh, name))
    settings.stt.cloud = cloud
    settings.translate.server_exe = server
    settings.live.overlay_geometry = geometry
    settings.first_run_done = True
    recommend(hw).apply(settings)
