"""
runpod_worker.py — FastAPI server that runs INSIDE the RunPod GPU pod.

Model is loaded in a background thread at startup so the server responds to
/health immediately. The client polls until health.ready == true before sending
audio for transcription.

Endpoints:
  GET  /health      → {"status":"ok|loading|error", "ready":bool, "gpu":"...", "model":"..."}
  POST /transcribe  → multipart audio upload → Whisper-compatible JSON

Run with:
  uvicorn runpod_worker:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = FastAPI(title="SurSync RunPod Worker", version="1.1.0")

# ── Model state (written from background thread, read from request handlers) ──
_model        = None          # WhisperModel once loaded
_model_ready  = False         # True when model is loaded and ready
_model_error: Optional[str]  = None
WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "large-v3")


def _load_model_background() -> None:
    """Load the Whisper model on CUDA in a background thread."""
    global _model, _model_ready, _model_error
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]

        log.info("[worker] Loading faster-whisper '%s' on CUDA …", WHISPER_MODEL)
        m = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type="int8_float16",
        )
        _model       = m
        _model_ready = True
        log.info("[worker] Model '%s' ready ✓", WHISPER_MODEL)

    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
        log.error("[worker] Model load FAILED: %s", exc, exc_info=True)


@app.on_event("startup")
async def startup_event() -> None:
    """Kick off model loading in a daemon thread — server accepts requests immediately."""
    # Log key env vars so we can see them in pod logs
    log.info("[worker] WHISPER_MODEL=%s", WHISPER_MODEL)
    log.info("[worker] HF_HOME=%s", os.environ.get("HF_HOME", "(not set)"))

    # Ensure HF cache directory is writable.
    # If the RunPod Network Volume isn't mounted, /runpod-volume won't exist —
    # fall back to /tmp/hf-cache so the server still starts (model re-downloads,
    # but that is better than a hard crash).
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        try:
            from pathlib import Path as _P
            _P(hf_home).mkdir(parents=True, exist_ok=True)
            log.info("[worker] HF cache dir ready: %s", hf_home)
        except Exception as exc:
            fallback = "/tmp/hf-cache"
            os.environ["HF_HOME"] = fallback
            try:
                from pathlib import Path as _P
                _P(fallback).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            log.warning(
                "[worker] HF cache dir %s not writable (%s) — falling back to %s",
                hf_home, exc, fallback,
            )

    t = threading.Thread(target=_load_model_background, daemon=True, name="model-loader")
    t.start()


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    """
    Return worker status.  The local client polls this until ready == true.

    Status values:
      "loading" — model is being downloaded / loaded
      "ok"      — model loaded, ready for transcription
      "error"   — model failed to load
    """
    gpu_name = "unknown"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            gpu_name = proc.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass

    if _model_ready:
        status = "ok"
    elif _model_error:
        status = "error"
    else:
        status = "loading"

    return {
        "status": status,
        "ready":  _model_ready,
        "gpu":    gpu_name,
        "model":  WHISPER_MODEL,
        "error":  _model_error,
    }


# ── Transcription ─────────────────────────────────────────────────────────────


@app.post("/transcribe")
async def transcribe_audio(
    file:     UploadFile     = File(...),
    language: Optional[str] = Query(default=None),
) -> JSONResponse:
    """
    Accept an audio file and return a Whisper-compatible transcript JSON.

    Waits until model is ready (returns 503 if still loading).
    Response format matches openai-whisper so processor.py works unchanged.
    """
    if not _model_ready:
        if _model_error:
            raise HTTPException(500, f"Model failed to load: {_model_error}")
        raise HTTPException(503, "Model is still loading — retry in a few seconds")

    suffix   = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path = Path(tempfile.mktemp(suffix=suffix))

    try:
        with open(tmp_path, "wb") as dst:
            shutil.copyfileobj(file.file, dst)

        size_mb = tmp_path.stat().st_size / 1e6
        log.info("[worker] Transcribing '%s' (%.1f MB), lang=%s",
                 file.filename, size_mb, language or "auto")

        segments_iter, info = _model.transcribe(
            str(tmp_path),
            language=language or None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

        segments_out:    list[dict] = []
        full_text_parts: list[str]  = []
        seg_id = 0

        for seg in segments_iter:
            words_out: list[dict] = []
            for w in (seg.words or []):
                words_out.append({
                    "word":        w.word,
                    "start":       round(float(w.start), 3),
                    "end":         round(float(w.end),   3),
                    "probability": round(float(w.probability), 4),
                })
            seg_text = seg.text.strip()
            full_text_parts.append(seg_text)
            segments_out.append({
                "id":    seg_id,
                "start": round(float(seg.start), 3),
                "end":   round(float(seg.end),   3),
                "text":  seg_text,
                "words": words_out,
            })
            seg_id += 1

        detected_lang = info.language or language or "en"
        result = {
            "language": detected_lang,
            "text":     " ".join(full_text_parts),
            "segments": segments_out,
        }

        total_words = sum(len(s["words"]) for s in segments_out)
        log.info("[worker] Done — %d segments, %d words, lang=%s",
                 len(segments_out), total_words, detected_lang)
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("[worker] Transcription error: %s", exc)
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
