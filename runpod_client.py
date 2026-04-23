"""
runpod_client.py — RunPod pod lifecycle + Whisper transcription client.

Responsibilities:
  is_configured()            → True if API key + pod ID are set in env / .env
  resume_pod(pod_id, key)    → start pod via RunPod SDK, wait until RUNNING
  get_pod_api_url(pod_id)    → "https://{pod_id}-8000.proxy.runpod.net"
  wait_for_api(base_url)     → poll GET /health until 200
  transcribe(base_url, path) → POST multipart WAV → Whisper-format JSON
  stop_pod(pod_id, key)      → stop pod via RunPod SDK
  full_pipeline(path, lang)  → orchestrate all steps; always stops pod in finally
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────


def _load_config() -> tuple[str, str]:
    """
    Return (RUNPOD_API_KEY, RUNPOD_POD_ID) from environment or the project .env file.
    Empty strings are returned if either value is not found.
    """
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    pod_id  = os.environ.get("RUNPOD_POD_ID",  "")

    if api_key and pod_id:
        return api_key, pod_id

    # Fallback: parse .env two levels up from this file (project root)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == "RUNPOD_API_KEY" and not api_key:
                api_key = v
            elif k == "RUNPOD_POD_ID" and not pod_id:
                pod_id = v

    return api_key, pod_id


def is_configured() -> bool:
    """Return True when both RUNPOD_API_KEY and RUNPOD_POD_ID are set and non-empty."""
    api_key, pod_id = _load_config()
    return bool(api_key and pod_id)


# ── Pod lifecycle ──────────────────────────────────────────────────────────────


def resume_pod(pod_id: str, api_key: str, *, timeout_s: int = 120) -> None:
    """
    Resume (start) a stopped RunPod pod and wait until it reports RUNNING.

    Raises:
        RuntimeError: if the pod does not reach RUNNING within timeout_s seconds.
        ImportError:  if the `runpod` SDK is not installed.
    """
    import runpod as _rp  # type: ignore[import]

    _rp.api_key = api_key
    log.info("[runpod] Resuming pod %s …", pod_id)
    _rp.resume_pod(pod_id=pod_id, gpu_count=1)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            pod = _rp.get_pod(pod_id)
            if pod:
                # RunPod SDK may use either field depending on version
                current  = pod.get("currentStatus", "")
                desired  = pod.get("desiredStatus", "")
                runtime  = pod.get("runtime")         # non-None once GPU is allocated
                status   = current or desired
                log.debug("[runpod] Pod status: current=%s desired=%s runtime=%s",
                          current, desired, bool(runtime))
                if status == "RUNNING" or runtime is not None:
                    log.info("[runpod] Pod %s is RUNNING", pod_id)
                    return
        except Exception as exc:  # noqa: BLE001
            log.debug("[runpod] get_pod error (will retry): %s", exc)
        time.sleep(5)

    raise RuntimeError(
        f"RunPod pod {pod_id} did not reach RUNNING state within {timeout_s}s"
    )


def stop_pod(pod_id: str, api_key: str) -> None:
    """
    Stop a RunPod pod.  Logs a warning on failure instead of raising — this is
    called in a finally block where we must not mask the original exception.
    """
    try:
        import runpod as _rp  # type: ignore[import]

        _rp.api_key = api_key
        _rp.stop_pod(pod_id)
        log.info("[runpod] Pod %s stopped", pod_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("[runpod] Failed to stop pod %s: %s", pod_id, exc)


def get_pod_api_url(pod_id: str) -> str:
    """Return the RunPod HTTPS proxy URL for port 8000 on the given pod."""
    return f"https://{pod_id}-8000.proxy.runpod.net"


def wait_for_api(
    base_url: str,
    *,
    timeout_s: int = 480,
    interval_s: int = 8,
) -> None:
    """
    Poll GET {base_url}/health until the worker reports ready == true.

    Two phases happen while we wait:
      1. Pod/proxy booting  → connection refused  (no HTTP yet)
      2. Model loading      → HTTP 200, ready: false  (server up, model downloading)

    timeout_s is 480 s (8 min) to cover cold-start model download (~3 GB).
    With a RunPod Network Volume the model is already cached and this finishes
    in ~60-90 s.

    Raises:
        RuntimeError: if the worker is not ready within timeout_s seconds.
    """
    health_url = f"{base_url}/health"
    log.info("[runpod] Waiting for worker at %s (timeout %ds) …", health_url, timeout_s)
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == 200:
                data      = resp.json()
                status    = data.get("status", "?")
                ready     = data.get("ready", True)   # default True = older worker without ready flag
                error_msg = data.get("error")

                if ready:
                    log.info("[runpod] Worker ready ✓ — gpu=%s model=%s",
                             data.get("gpu", "?"), data.get("model", "?"))
                    return

                # Worker returned an explicit error — fail immediately, don't burn 480s
                if status == "error":
                    raise RuntimeError(
                        f"RunPod worker model load failed: {error_msg or '(no detail)'}"
                    )

                # Server up but model still loading
                if error_msg:
                    log.info("[runpod] Worker status: %s — %s", status, error_msg)
                else:
                    log.info("[runpod] Worker status: %s (model loading, waiting…)", status)
            else:
                log.debug("[runpod] /health → HTTP %s (body: %s)",
                          resp.status_code, resp.text[:200])

        except requests.RequestException as exc:
            log.debug("[runpod] /health not yet reachable: %s", exc)

        time.sleep(interval_s)

    raise RuntimeError(
        f"RunPod worker at {base_url} did not become ready within {timeout_s}s"
    )


# ── Transcription ──────────────────────────────────────────────────────────────


def _maybe_compress_to_mp3(audio_path: Path) -> Path:
    """
    If audio_path is a WAV, convert it to MP3 (192 kbps) next to the original
    and return the MP3 path.  WAV → MP3 reduces size ~7× (7 MB → ~1 MB) and
    avoids SSL EOF errors on the RunPod proxy when uploading large files.

    Returns audio_path unchanged if it is already MP3 or ffmpeg is unavailable.
    """
    if audio_path.suffix.lower() != ".wav":
        return audio_path

    mp3_path = audio_path.with_suffix(".upload.mp3")
    try:
        import subprocess as _sp
        result = _sp.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-b:a", "192k", str(mp3_path)],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and mp3_path.exists() and mp3_path.stat().st_size > 1000:
            log.info(
                "[runpod] Compressed WAV → MP3: %.1f MB → %.1f MB",
                audio_path.stat().st_size / 1e6,
                mp3_path.stat().st_size / 1e6,
            )
            return mp3_path
    except Exception as exc:  # noqa: BLE001
        log.warning("[runpod] WAV→MP3 compression failed (%s) — uploading raw WAV", exc)

    if mp3_path.exists():
        mp3_path.unlink(missing_ok=True)
    return audio_path


def transcribe(
    base_url: str,
    audio_path: Path,
    language: Optional[str] = None,
    *,
    request_timeout_s: int = 600,
) -> dict:
    """
    POST an audio file to the RunPod worker /transcribe endpoint.

    The worker runs faster-whisper and returns a Whisper-compatible result dict:
    ::

        {
            "language": "hi",
            "text": "Jai ho jai ho …",
            "segments": [
                {
                    "id": 0,
                    "start": 12.3,
                    "end": 16.8,
                    "text": "Jai ho jai ho",
                    "words": [
                        {"word": "Jai", "start": 12.3, "end": 12.7, "probability": 0.95},
                        …
                    ]
                }
            ]
        }

    Args:
        base_url:          Worker base URL, e.g. "https://{pod_id}-8000.proxy.runpod.net"
        audio_path:        Local path to the audio file to transcribe.
        language:          ISO 639-1 language hint (e.g. "hi", "te") or None for auto-detect.
        request_timeout_s: HTTP timeout in seconds (long songs can take a few minutes).

    Returns:
        Whisper-compatible result dict.

    Raises:
        requests.HTTPError: on non-2xx responses.
    """
    url = f"{base_url}/transcribe"

    # Convert WAV → MP3 before uploading to reduce file size 7x and avoid
    # SSL EOF errors on the RunPod proxy which drops large uploads.
    upload_path = _maybe_compress_to_mp3(audio_path)
    try:
        size = upload_path.stat().st_size
        log.info(
            "[runpod] Uploading %s (%.1f MB) for transcription …",
            upload_path.name,
            size / 1e6,
        )

        params: dict = {}
        if language:
            params["language"] = language

        # Retry up to 3 times — SSL EOF on the RunPod proxy is transient
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(1, 4):
            try:
                with open(upload_path, "rb") as fh:
                    mime = "audio/wav" if upload_path.suffix.lower() == ".wav" else "audio/mpeg"
                    resp = requests.post(
                        url,
                        files={"file": (upload_path.name, fh, mime)},
                        params=params,
                        timeout=request_timeout_s,
                    )
                resp.raise_for_status()
                break   # success — exit retry loop
            except (requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if attempt < 3:
                    wait = attempt * 5
                    log.warning(
                        "[runpod] Upload attempt %d failed (%s) — retrying in %ds …",
                        attempt, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise last_exc from exc
        result = resp.json()
    finally:
        # Remove the compressed temp file if we created one
        if upload_path != audio_path:
            upload_path.unlink(missing_ok=True)

    total_words = sum(len(seg.get("words", [])) for seg in result.get("segments", []))
    log.info(
        "[runpod] Transcript received — %d segments, %d words, lang=%s",
        len(result.get("segments", [])),
        total_words,
        result.get("language", "?"),
    )
    return result


# ── Full pipeline ──────────────────────────────────────────────────────────────


def full_pipeline(
    audio_path: Path,
    language: Optional[str] = None,
) -> Optional[dict]:
    """
    Run the complete RunPod transcription pipeline:

    1. Load API key + pod ID from config.
    2. Resume pod → wait until RUNNING.
    3. Wait for worker API → GET /health returns 200.
    4. POST audio file → receive Whisper-format transcript.
    5. Stop pod (always, even on failure).

    Args:
        audio_path: Local path to the audio file (WAV or MP3).
        language:   ISO 639-1 language hint or None for auto-detect.

    Returns:
        Whisper-compatible result dict on success, or None on any failure.
        Failures are logged as errors; the caller should fall back gracefully.
    """
    api_key, pod_id = _load_config()
    if not api_key or not pod_id:
        log.warning(
            "[runpod] RUNPOD_API_KEY or RUNPOD_POD_ID not configured — skipping"
        )
        return None

    start_time  = time.monotonic()
    pod_started = False

    try:
        resume_pod(pod_id, api_key)
        pod_started = True

        base_url = get_pod_api_url(pod_id)
        wait_for_api(base_url)

        result = transcribe(base_url, audio_path, language)

        elapsed_min = (time.monotonic() - start_time) / 60
        cost_est    = elapsed_min / 60 * 0.34          # RTX 4090 ≈ $0.34/hr
        log.info(
            "[runpod] Pipeline complete in %.1f min (est. cost ≈ $%.4f @ RTX 4090)",
            elapsed_min,
            cost_est,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        log.error("[runpod] Pipeline failed: %s", exc, exc_info=True)
        return None

    finally:
        if pod_started:
            stop_pod(pod_id, api_key)
