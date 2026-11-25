import os
import re
import time
import shutil
import tempfile
import subprocess
import requests
import numpy as np
import soundfile as sf
import torch
import whisper
import stable_whisper
from demucs.pretrained import get_model as get_demucs_model
from demucs.apply import apply_model
import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "turbo")

# Keep both models warm in GPU VRAM between serverless requests
_DEMUCS = None
_WHISPER = None


def get_demucs():
    global _DEMUCS
    if _DEMUCS is None:
        m = get_demucs_model("htdemucs")
        m.to(DEVICE)
        m.eval()
        _DEMUCS = m
    return _DEMUCS


def get_whisper():
    global _WHISPER
    if _WHISPER is None:
        _WHISPER = stable_whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)
    return _WHISPER


def run_ffmpeg(args):
    p = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.strip()[:300]}")


def isolate_vocals_16k(input_path: str, work_dir: str) -> tuple[str, float]:
    """Separate clean vocals using HTDemucs on GPU and output 16 kHz mono WAV."""
    wav44 = os.path.join(work_dir, "in_44k.wav")
    run_ffmpeg(["-i", input_path, "-vn", "-ac", "2", "-ar", "44100", "-f", "wav", wav44])

    audio_np, sr = sf.read(wav44, dtype="float32", always_2d=True)
    duration_sec = float(audio_np.shape[0]) / float(sr)

    # Shape: [batch=1, channels=2, samples]
    wav_t = torch.from_numpy(audio_np.T).unsqueeze(0).to(DEVICE)
    demucs = get_demucs()
    vocals_idx = demucs.sources.index("vocals")

    with torch.no_grad():
        stems = apply_model(
            demucs,
            wav_t,
            device=DEVICE,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )
        vocals_t = stems[0, vocals_idx].detach().cpu().numpy().T  # [samples, 2]

    del wav_t, stems
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    voc_44 = os.path.join(work_dir, "vocals_44k.wav")
    voc_16 = os.path.join(work_dir, "vocals_16k.wav")
    sf.write(voc_44, np.clip(vocals_t, -1.0, 1.0), sr, subtype="PCM_16")
    run_ffmpeg(["-i", voc_44, "-ac", "1", "-ar", "16000", "-f", "wav", voc_16])
    return voc_16, duration_sec


def detect_vocal_language(model, voc_16k_path: str, fallback: str = "id") -> str:
    """Detect language from the most active 30-second vocal window."""
    try:
        audio = whisper.load_audio(voc_16k_path)
        n = len(audio)
        win = 30 * 16000
        if n > win:
            # Find 30s window with highest vocal RMS energy (skips long DJ/instrumental intros)
            step = 5 * 16000
            best_pos = 0
            best_rms = -1.0
            for pos in range(0, max(1, n - win), step):
                chunk = audio[pos : pos + win]
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                if rms > best_rms:
                    best_rms = rms
                    best_pos = pos
            seg = audio[best_pos : best_pos + win]
        else:
            seg = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(seg, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        if probs.get(lang, 0.0) >= 0.35:
            return lang
    except Exception:
        pass
    return fallback


def fmt_lrc_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"[{m:02d}:{s:05.2f}]"


TS_RE = re.compile(r"^\s*(?:\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]\s*)+")


def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    audio_url = inp.get("audio_url")
    raw_lines = inp.get("lines")
    raw_text = inp.get("lyrics") or ""
    req_lang = str(inp.get("language") or "").strip().lower()

    if not audio_url:
        return {"error": "Missing audio_url"}

    if isinstance(raw_lines, list) and len(raw_lines) > 0:
        lines = [TS_RE.sub("", str(x)).strip() for x in raw_lines]
    else:
        lines = [TS_RE.sub("", l).strip() for l in str(raw_text).splitlines()]
    lines = [l for l in lines if l and not (l.startswith("[") and l.endswith("]"))][:250]

    if not lines:
        return {"error": "No valid lyric lines provided"}

    work_dir = tempfile.mkdtemp(prefix="scai_lyr_")
    try:
        # 1. Download audio from SC AI backend
        in_path = os.path.join(work_dir, "input_audio")
        r = requests.get(audio_url, stream=True, timeout=120)
        if not r.ok:
            return {"error": f"Failed to download input audio ({r.status_code})"}
        with open(in_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        # 2. Isolate clean singing vocals on GPU (HTDemucs)
        t_sep = time.time()
        voc_16k, duration_sec = isolate_vocals_16k(in_path, work_dir)
        sep_sec = round(time.time() - t_sep, 2)

        # 3. Forced-align lyrics to clean vocals using stable-ts (Whisper large-v3-turbo)
        t_aln = time.time()
        model = get_whisper()
        lang = req_lang if req_lang and req_lang != "auto" else detect_vocal_language(model, voc_16k, fallback="id")

        joined_lyrics = "\n".join(lines)
        result = model.align(
            voc_16k,
            joined_lyrics,
            language=lang,
            original_split=True,
            vad=True,
            vad_threshold=0.32,
            verbose=None,
        )
        aln_sec = round(time.time() - t_aln, 2)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # 4. Build strictly-ordered line & word timestamps
        raw_segs = list(result.segments) if result and getattr(result, "segments", None) else []
        segments_out = []
        lrc_lines = []

        prev_start = 0.0
        for idx, line_txt in enumerate(lines):
            seg = raw_segs[idx] if idx < len(raw_segs) else None
            if seg is not None:
                s = float(getattr(seg, "start", prev_start) or prev_start)
                e = float(getattr(seg, "end", s + 2.5) or (s + 2.5))
                words_out = []
                for w in (getattr(seg, "words", None) or []):
                    wt = str(getattr(w, "word", "") or "").strip()
                    if not wt:
                        continue
                    ws = float(getattr(w, "start", s) or s)
                    we = float(getattr(w, "end", ws + 0.2) or (ws + 0.2))
                    words_out.append({
                        "start": round(max(0.0, ws), 3),
                        "end": round(max(ws + 0.05, we), 3),
                        "word": wt,
                    })
            else:
                s = min(duration_sec - 0.5, prev_start + 2.5)
                e = min(duration_sec, s + 2.5)
                words_out = []

            if idx > 0 and s < prev_start + 0.15:
                s = min(duration_sec - 0.2, prev_start + 0.15)
            e = max(s + 0.35, min(duration_sec, e))
            prev_start = s

            segments_out.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "text": line_txt,
                "words": words_out,
            })
            lrc_lines.append(f"{fmt_lrc_time(s)}{line_txt}")

        return {
            "status": "ok",
            "model": f"HTDemucs + Whisper {WHISPER_MODEL_NAME}",
            "language": lang,
            "durationSec": round(duration_sec, 2),
            "sepSec": sep_sec,
            "alignSec": aln_sec,
            "elapsedSec": round(time.time() - t0, 2),
            "lrc": "\n".join(lrc_lines),
            "segments": segments_out,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
