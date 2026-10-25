"""檔案模式的批次執行：一次處理一個階段，所有檔案做完這個階段才進下一個。

這樣每個階段只需要載入一次模型，而且上一階段的模型已經隨子行程結束而釋放，
STT 和翻譯都能用 VRAM 允許的最大模型。

階段：準備（轉檔／下載）→ 人聲分離 → STT → 講者分離 → 術語先行 → 翻譯 → 輸出。
每個階段的產物都帶指紋，改了設定只重做受影響的階段。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import traceback
from pathlib import Path
from typing import Callable

from omni2.audio.decode import decode_to_wav
from omni2.config.settings import Settings, SttSettings
from omni2.core import paths
from omni2.core.jsonio import write_text_atomic
from omni2.core.types import Transcript
from omni2.export import formats as fmt
from omni2.jobs.job import Job, JobSpec, fingerprint, is_url
from omni2.post.layout import build_cues
from omni2.stt.language_models import stt_for_language
from omni2.translate.client import ChatClient
from omni2.translate.glossary import Glossary
from omni2.translate.prompts import PROMPT_VERSION, style_name_for
from omni2.translate.server import LlamaServer
from omni2.translate.terms import TERMS_VERSION, TermsResult, extract_terms
from omni2.translate.translator import BatchTranslator
from omni2.workers.stage import Cancelled, WorkerError, run_stage

log = logging.getLogger(__name__)

EventFn = Callable[[dict], None]

# 階段代號 → 給使用者看的名稱（CLI 和介面共用）
STAGE_LABELS = {"prepare": "轉檔", "separate": "人聲分離", "stt": "辨識", "diarize": "講者分離",
                "terms": "抽術語", "translate": "翻譯", "export": "輸出"}
# 讓使用者確認抽出來的術語：回傳修改後的結果；回傳 None 表示這份檔案不用術語
ReviewFn = Callable[[Job, TermsResult], "TermsResult | None"]


class FileRunner:
    def __init__(
        self,
        specs: list[JobSpec],
        settings: Settings,
        *,
        glossary: Glossary | None = None,
        on_event: EventFn | None = None,
        review: ReviewFn | None = None,
        cancel: threading.Event | None = None,
        jobs_root: Path | None = None,
    ):
        self.settings = settings
        self.glossary = glossary or Glossary()  # 使用者的術語表，同時當 STT 提示詞
        self.on_event = on_event or (lambda ev: None)
        self.review = review
        self.cancel = cancel
        root = jobs_root or paths.jobs_dir()
        self.jobs = [Job(spec, root) for spec in specs]

    def run(self) -> list[Job]:
        self._prepare(self._alive())
        self._separate([j for j in self._alive() if j.spec.separate_vocals])
        self._stt(self._alive())
        self._diarize([j for j in self._alive() if j.spec.diarize])
        self._translate([j for j in self._alive() if j.spec.translate])
        self._export(self._alive())
        return self.jobs

    # ---- 共用 ----

    def _alive(self) -> list[Job]:
        return [j for j in self.jobs if j.error is None]

    def _emit(self, job: Job, stage: str, status: str, **extra) -> None:
        self.on_event({"type": "stage", "id": job.id, "job": job.name, "stage": stage, "status": status, **extra})

    def _fail(self, job: Job, stage: str, message: str, detail: str = "") -> None:
        job.error = f"{stage}：{message}"
        if detail:
            log.debug(detail)
        self._emit(job, stage, "error", message=message)

    def _run_worker(self, stage: str, target: str, payload: dict, todo: list[tuple[Job, str]],
                    on_done: Callable[[Job, str, dict], str]) -> None:
        """在子行程跑一個階段。on_done 記錄完成狀態並回傳要顯示的訊息。"""
        by_id = {job.id: (job, fp) for job, fp in todo}
        finished: set[str] = set()

        def on_event(ev: dict) -> None:
            kind = ev["type"]
            if kind == "log":
                logging.getLogger("omni2.worker").log(logging.getLevelName(ev["level"]), ev["message"])
            elif kind == "status":
                self.on_event(ev)
            elif ev.get("job") in by_id:
                job, fp = by_id[ev["job"]]
                if kind == "job_start":
                    self._emit(job, stage, "start")
                elif kind == "progress":
                    self._emit(job, stage, "progress", value=ev["value"])
                elif kind == "job_done":
                    finished.add(job.id)
                    self._emit(job, stage, "done", message=on_done(job, fp, ev))
                elif kind == "job_error":
                    finished.add(job.id)
                    self._fail(job, stage, ev["message"], ev.get("traceback", ""))

        try:
            run_stage(target, payload, on_event, self.cancel)
        except WorkerError as e:
            for job, _ in todo:
                if job.id not in finished:
                    self._fail(job, stage, str(e), e.detail)

    # ---- 指紋 ----

    def _fp_prepare(self, job: Job) -> str:
        return fingerprint("prepare", 1, job.id)

    def _fp_separate(self, job: Job) -> str:
        if not job.spec.separate_vocals:
            return "none"
        return fingerprint("separate", 1, self._fp_prepare(job), self.settings.separation.model)

    def _stt_settings(self, job: Job) -> SttSettings:
        """這個檔案實際用的辨識設定（韓文會換成韓文微調版，見 omni2.stt.language_models）。"""
        s = self.settings.stt
        return stt_for_language(s, job.spec.src_lang or s.language)

    def _fp_stt(self, job: Job) -> str:
        s, lay = self._stt_settings(job), self.settings.layout
        return fingerprint(
            "stt", 3, self._fp_separate(job), s.engine, s.active_model(), s.compute_type, s.batch_size > 1,
            job.spec.src_lang or s.language, s.vad_preset, s.beam_size, s.filter_hallucinations,
            self.glossary.hotwords(), lay.max_duration, lay.max_line_width * lay.max_lines,
        )

    def _fp_diarize(self, job: Job) -> str:
        d = self.settings.diarization
        detail = (d.model, d.threshold) if d.engine == "ecapa" else ()
        return fingerprint("diarize", 3, self._fp_stt(job), d.engine, *detail, job.spec.num_speakers)

    def _fp_text(self, job: Job) -> str:
        """最終文字的指紋：有講者分離就是分離後的片段，否則是辨識結果。"""
        return self._fp_diarize(job) if job.spec.diarize else self._fp_stt(job)

    def _text(self, job: Job) -> Transcript | None:
        if job.spec.diarize:
            return job.load_diarized(self._fp_diarize(job))
        return job.load_transcript(self._fp_stt(job))

    def _model_name(self) -> str:
        t = self.settings.translate
        return t.endpoint_model if t.endpoint else Path(t.model_path).name

    def _style(self) -> str:
        return style_name_for(self._model_name(), self.settings.translate.prompt_style)

    def _known_terms(self, src_lang: str | None) -> Glossary:
        """內建術語表＋使用者術語表（後者優先）。"""
        known = Glossary()
        for name in self.settings.terms.builtin:
            known = known.merged(Glossary.builtin(name, src_lang))
        return known.merged(self.glossary)

    def _fp_terms(self, job: Job, transcript: Transcript) -> str:
        t = self.settings
        model = Path(t.terms.model_path or t.translate.model_path).name
        return fingerprint("terms", TERMS_VERSION, self._fp_text(job), job.spec.tgt_lang, model,
                           t.terms.chunk_chars, self._known_terms(transcript.language).digest())

    def _job_terms(self, job: Job, transcript: Transcript) -> TermsResult:
        if not job.spec.terms_first:
            return TermsResult()
        data = job.load_terms(self._fp_terms(job, transcript))
        return TermsResult.from_dict(data) if data else TermsResult()

    def _fp_translate(self, job: Job, transcript: Transcript) -> str:
        t = self.settings.translate
        model = t.endpoint + "|" + t.endpoint_model if t.endpoint else self._model_name()
        terms = self._job_terms(job, transcript)
        glossary = self._known_terms(transcript.language).merged(terms.glossary())
        summary = hashlib.sha1(terms.summary.encode("utf-8")).hexdigest()[:12]
        return fingerprint(
            "translate", PROMPT_VERSION, self._fp_text(job), job.spec.tgt_lang, model, self._style(),
            t.batch_lines, t.context_lines, t.temperature, t.opencc, glossary.digest(), summary,
        )

    # ---- 階段 1：下載、轉成 16k wav ----

    def _prepare(self, jobs: list[Job]) -> None:
        for job in jobs:
            fp = self._fp_prepare(job)
            if job.stage_done("prepare", fp) and job.audio_path.exists():
                self._emit(job, "prepare", "skip")
                continue
            self._emit(job, "prepare", "start")
            try:
                if is_url(job.spec.input):
                    from omni2.audio.download import download_audio

                    source, title = download_audio(
                        job.spec.input, job.dir,
                        progress=lambda p, job=job: self._emit(job, "prepare", "progress", value=p),
                    )
                    job.state.update(source=str(source), title=title)
                    job.save()
                duration = decode_to_wav(job.source_path, job.audio_path)
            except Exception as e:  # noqa: BLE001
                self._fail(job, "prepare", f"{type(e).__name__}: {e}")
                continue
            job.set_stage("prepare", fp, "done", duration=round(duration, 2))
            self._emit(job, "prepare", "done", message=f"{duration / 60:.1f} 分鐘")

    # ---- 階段 2：人聲分離（子行程） ----

    def _separate(self, jobs: list[Job]) -> None:
        todo = []
        for job in jobs:
            fp = self._fp_separate(job)
            if job.stage_done("separate", fp) and job.vocals_path.exists():
                self._emit(job, "separate", "skip")
            else:
                todo.append((job, fp))
        if not todo:
            return
        payload = {
            "model": self.settings.separation.model,
            "items": [{"job": job.id, "input": str(job.source_path), "out": str(job.vocals_path)}
                      for job, _ in todo],
        }

        def on_done(job: Job, fp: str, ev: dict) -> str:
            job.set_stage("separate", fp, "done")
            return ""

        self._run_worker("separate", "omni2.post.separate:run", payload, todo, on_done)

    # ---- 階段 3：STT（子行程） ----

    def _stt(self, jobs: list[Job]) -> None:
        todo = []
        for job in jobs:
            fp = self._fp_stt(job)
            if job.load_transcript(fp) is not None:
                self._emit(job, "stt", "skip")
            else:
                todo.append((job, fp))
        if not todo:
            return
        # 用不同模型的檔案（例如韓文用韓文微調版）分開跑，各自載入一次模型
        by_model: dict[str, list[tuple[Job, str]]] = {}
        for job, fp in todo:
            by_model.setdefault(self._stt_settings(job).active_model(), []).append((job, fp))
        for group in by_model.values():
            self._stt_group(group)

    def _stt_group(self, todo: list[tuple[Job, str]]) -> None:
        s, lay = self._stt_settings(todo[0][0]), self.settings.layout
        payload = {
            "stt": s.model_dump(),
            "items": [
                {
                    "job": job.id,
                    "fp": fp,
                    "audio": str(job.stt_audio_path),
                    "duration": job.stage("prepare").get("duration", 0.0),
                    "out": str(job.transcript_path),
                    "language": job.spec.src_lang or s.language,
                    "vad": s.vad_preset,
                    "hotwords": self.glossary.hotwords(),
                    "max_duration": lay.max_duration,
                    "max_width": lay.max_line_width * lay.max_lines,
                }
                for job, fp in todo
            ],
        }

        def on_done(job: Job, fp: str, ev: dict) -> str:
            job.set_stage("stt", fp, "done", language=ev["language"], segments=ev["segments"])
            message = f"{ev['segments']} 句，語言 {ev['language']}"
            if ev["dropped"]:
                message += f"，過濾 {ev['dropped']} 句幻覺"
            return message

        self._run_worker("stt", "omni2.jobs.stt_worker:run", payload, todo, on_done)

    # ---- 階段 4：講者分離（子行程） ----

    def _diarize(self, jobs: list[Job]) -> None:
        todo = []
        for job in jobs:
            fp = self._fp_diarize(job)
            if job.load_diarized(fp) is not None:
                self._emit(job, "diarize", "skip")
            else:
                todo.append((job, fp))
        if not todo:
            return
        d = self.settings.diarization
        payload = {
            "engine": d.engine,
            "model": d.model,
            "threshold": d.threshold,
            "items": [{"job": job.id, "fp": fp, "audio": str(job.stt_audio_path),
                       "transcript": str(job.transcript_path), "out": str(job.diarized_path),
                       "num_speakers": job.spec.num_speakers} for job, fp in todo],
        }

        def on_done(job: Job, fp: str, ev: dict) -> str:
            job.set_stage("diarize", fp, "done", speakers=ev["speakers"])
            return f"{ev['speakers']} 位講者，{ev['segments']} 句"

        self._run_worker("diarize", "omni2.post.diarize:run", payload, todo, on_done)

    # ---- 階段 5、6：術語先行、翻譯（llama-server） ----

    def _translate(self, jobs: list[Job]) -> None:
        loaded: list[tuple[Job, Transcript]] = []
        for job in jobs:
            transcript = self._text(job)
            if transcript is None:
                self._fail(job, "translate", "找不到辨識結果")
            else:
                loaded.append((job, transcript))

        # 術語還沒確認的檔案一定要重翻（術語會改變譯文）；其餘看翻譯斷點是否完整
        need_terms = [(j, t) for j, t in loaded
                      if j.spec.terms_first and not (j.load_terms(self._fp_terms(j, t)) or {}).get("reviewed")]
        terms_ids = {j.id for j, _ in need_terms}
        need_translate = [(j, t) for j, t in loaded
                          if j.id in terms_ids
                          or len(j.load_translations(self._fp_translate(j, t))) < len(t.segments)]
        translate_ids = {j.id for j, _ in need_translate}
        for job, _ in loaded:
            if job.id not in translate_ids:
                if job.spec.terms_first:
                    self._emit(job, "terms", "skip")
                self._emit(job, "translate", "skip")
        if not need_translate:
            return

        t = self.settings.translate
        terms_model = self.settings.terms.model_path
        server = None
        try:
            if need_terms and terms_model and terms_model != t.model_path and not t.endpoint:
                # 抽術語指定了不同的模型：先用它抽完，再換翻譯模型
                with LlamaServer(t.server_exe, terms_model, ctx_size=self.settings.terms.ctx_size,
                                 gpu_layers=t.gpu_layers) as terms_server:
                    client = ChatClient(terms_server.base_url)
                    try:
                        self._extract_terms(client, need_terms)
                    finally:
                        client.close()
                need_terms = []
            if t.endpoint:
                client = ChatClient(t.endpoint, model=t.endpoint_model)
            else:
                self.on_event({"type": "status", "message": f"啟動翻譯模型 {Path(t.model_path).name}"})
                ctx = max(t.ctx_size, self.settings.terms.ctx_size) if need_terms else t.ctx_size
                server = LlamaServer(t.server_exe, t.model_path, ctx_size=ctx,
                                     gpu_layers=t.gpu_layers, extra_args=t.server_args)
                client = ChatClient(server.start())
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            for job, _ in need_translate:
                self._fail(job, "translate", str(e))
            return

        try:
            self._extract_terms(client, need_terms)
            self._review_terms([(j, tr) for j, tr in need_translate if j.error is None and j.spec.terms_first])
            for job, transcript in need_translate:
                if job.error is None:
                    self._translate_one(client, job, transcript)
        finally:
            client.close()
            if server:
                server.stop()

    def _extract_terms(self, client: ChatClient, jobs: list[tuple[Job, Transcript]]) -> None:
        for job, transcript in jobs:
            fp = self._fp_terms(job, transcript)
            if job.load_terms(fp) is not None:
                continue  # 抽過了，只是還沒確認
            self._emit(job, "terms", "start")
            try:
                result = extract_terms(
                    client, [s.text for s in transcript.segments],
                    src_lang=job.spec.src_lang or transcript.language, tgt_lang=job.spec.tgt_lang,
                    known=self._known_terms(transcript.language),
                    chunk_chars=self.settings.terms.chunk_chars,
                    progress=lambda p, job=job: self._emit(job, "terms", "progress", value=p),
                )
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                self._fail(job, "terms", f"{type(e).__name__}: {e}", traceback.format_exc())
                continue
            job.save_terms(fp, result.to_dict(), reviewed=False)
            self._emit(job, "terms", "done", message=f"{len(result.terms)} 個術語")

    def _review_terms(self, jobs: list[tuple[Job, Transcript]]) -> None:
        for job, transcript in jobs:
            fp = self._fp_terms(job, transcript)
            data = job.load_terms(fp)
            if data is None or data.get("reviewed"):
                continue
            result = TermsResult.from_dict(data)
            if self.review is not None:
                reviewed = self.review(job, result)
                result = reviewed if reviewed is not None else TermsResult()
            job.save_terms(fp, result.to_dict(), reviewed=True)

    def _translate_one(self, client: ChatClient, job: Job, transcript: Transcript) -> None:
        t = self.settings.translate
        fp = self._fp_translate(job, transcript)
        done = job.load_translations(fp)
        if len(done) >= len(transcript.segments):
            self._emit(job, "translate", "skip")
            return
        terms = self._job_terms(job, transcript)
        self._emit(job, "translate", "start")
        translator = BatchTranslator(
            client,
            src_lang=job.spec.src_lang or transcript.language,
            tgt_lang=job.spec.tgt_lang,
            glossary=self._known_terms(transcript.language).merged(terms.glossary()),
            summary=terms.summary,
            batch_lines=t.batch_lines,
            context_lines=t.context_lines,
            temperature=t.temperature,
            max_retries=t.max_retries,
            opencc=t.opencc,
            style=self._style(),
        )
        try:
            translator.translate_all(
                [s.text for s in transcript.segments],
                done=done,
                on_result=job.append_translations,
                progress=lambda p: self._emit(job, "translate", "progress", value=p),
                cancel=self.cancel,
            )
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            self._fail(job, "translate", f"{type(e).__name__}: {e}", traceback.format_exc())
            return
        job.set_stage("translate", fp, "done", issues=len(translator.issues))
        message = f"{len(translator.issues)} 句未通過品質檢查" if translator.issues else ""
        self._emit(job, "translate", "done", message=message)

    # ---- 階段 7：排版、輸出 ----

    def load_result(self, job: Job) -> tuple[Transcript, list[str | None] | None] | None:
        """目前設定下這個 job 的辨識結果（含講者）和譯文；還沒辨識完就回傳 None。"""
        transcript = self._text(job)
        if transcript is None:
            return None
        translations = None
        if job.spec.translate:
            done = job.load_translations(self._fp_translate(job, transcript))
            translations = [done.get(i) for i in range(len(transcript.segments))]
        return transcript, translations

    def _export(self, jobs: list[Job]) -> None:
        out_cfg, layout = self.settings.output, self.settings.layout
        for job in jobs:
            result = self.load_result(job)
            if result is None:
                self._fail(job, "export", "找不到辨識結果")
                continue
            transcript, translations = result
            segments = transcript.segments
            tag = job.spec.tgt_lang if job.spec.translate else (transcript.language or "src")

            if is_url(job.spec.input):
                out_dir = Path(job.spec.out_dir or out_cfg.dir or paths.url_output_dir())
            else:
                out_dir = Path(job.spec.out_dir or out_cfg.dir or Path(job.spec.input).parent)
            cues = None
            job.outputs = []
            for kind in job.spec.formats:
                if kind in fmt.SUBTITLE_FORMATS:
                    if cues is None:
                        cues = build_cues(segments, translations, layout, bilingual=job.spec.bilingual)
                    text = {"srt": fmt.to_srt, "vtt": fmt.to_vtt, "ass": fmt.to_ass}[kind](cues, layout)
                elif kind == "txt":
                    text = fmt.to_txt(segments, translations, timestamps=out_cfg.timestamps)
                elif kind == "md":
                    text = fmt.to_md(segments, translations, title=job.name, timestamps=out_cfg.timestamps)
                else:
                    self._fail(job, "export", f"不支援的格式：{kind}")
                    break
                path = out_dir / f"{job.name}.{tag}.{kind}"
                write_text_atomic(path, text)
                job.outputs.append(path)
            if job.error is None:
                job.state["outputs"] = [str(p) for p in job.outputs]
                job.save()
                self._emit(job, "export", "done", message=", ".join(p.name for p in job.outputs))
