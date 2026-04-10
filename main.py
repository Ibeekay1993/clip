from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import uvicorn
import whisper
import yt_dlp
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="X-CLIP AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
ACTIVE_STATUSES = {"pending", "downloading", "transcribing", "analyzing", "generating"}
HOOK_PHRASES = [
    "wait for it",
    "you won't believe",
    "here's why",
    "secret",
    "watch this",
    "top",
    "best",
    "hack",
    "insane",
    "wild",
    "how to",
]

print("Loading Whisper model...")
model = whisper.load_model("base")


def ffprobe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def parse_json_request(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return {}
    return {}


def slugify_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"source_{digest}"


def find_source_file(video_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
        candidate = UPLOAD_DIR / f"{video_id}{ext}".replace("//", "/")
        if candidate.exists():
            return candidate
        candidate = UPLOAD_DIR / f"{video_id}.{ext.lstrip('.')}"
        if candidate.exists():
            return candidate
    for file in UPLOAD_DIR.iterdir():
      if file.is_file() and file.stem == video_id and file.suffix.lower() in VIDEO_EXTENSIONS:
            return file
    return None


def is_direct_video_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.hostname and "supabase.co" in parsed.hostname:
            return True
        if re.search(r"\.(mp4|mov|webm|m4v|mkv|avi)(\?|#|$)", parsed.path, re.I):
            return True
        if "/storage/v1/object/public/" in parsed.path or "/storage/v1/object/sign/" in parsed.path:
            return True
    except Exception:
        return False
    return False


def download_direct_file(url: str, destination: Path) -> Path:
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return destination


def download_video_source(video_url: str, source_id: Optional[str] = None) -> Path:
    source_id = source_id or slugify_url(video_url)
    destination_prefix = UPLOAD_DIR / source_id

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(destination_prefix) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            candidate = Path(ydl.prepare_filename(info))
            if candidate.exists():
                return candidate
            merged = destination_prefix.with_suffix(".mp4")
            if merged.exists():
                return merged
    except Exception:
        pass

    if is_direct_video_url(video_url):
        suffix = Path(urlparse(video_url).path).suffix or ".mp4"
        destination = destination_prefix.with_suffix(suffix if suffix in VIDEO_EXTENSIONS else ".mp4")
        return download_direct_file(video_url, destination)

    raise HTTPException(status_code=400, detail="Unable to download the provided video URL")


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@dataclass
class VideoProcessor:
    path: Path

    def __post_init__(self) -> None:
        self.duration = ffprobe_duration(self.path)
        if self.duration <= 0:
            self.duration = 0.0

    def transcribe(self) -> list[dict[str, Any]]:
        result = model.transcribe(str(self.path), word_timestamps=True)
        words: list[dict[str, Any]] = []

        for segment in result.get("segments", []):
            if segment.get("words"):
                for word in segment["words"]:
                    token = str(word.get("word", "")).strip()
                    if not token:
                        continue
                    words.append(
                        {
                            "start": float(word.get("start", 0.0)),
                            "end": float(word.get("end", word.get("start", 0.0) + 0.5)),
                            "text": token,
                            "confidence": float(word.get("probability", 0.9)),
                        }
                    )
            else:
                text = str(segment.get("text", "")).strip()
                if text:
                    words.append(
                        {
                            "start": float(segment.get("start", 0.0)),
                            "end": float(segment.get("end", segment.get("start", 0.0) + 2.0)),
                            "text": text,
                            "confidence": 0.85,
                        }
                    )

        return words

    def detect_scenes(self, sample_interval: float = 1.0, threshold: float = 24.0) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = self.duration or (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0)
        scenes: list[dict[str, Any]] = []
        prev_hist = None

        for second in np.arange(0, max(1, duration), sample_interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

            if prev_hist is not None:
                diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if diff > threshold / 100:
                    scenes.append(
                        {
                            "time": float(second),
                            "confidence": min(1.0, diff * 2),
                            "type": "scene_change",
                        }
                    )

            prev_hist = hist

        cap.release()
        return scenes

    def _score_window(self, text: str, start: float, end: float, scene_hits: int) -> float:
        lower = text.lower()
        hook_hits = sum(1 for phrase in HOOK_PHRASES if phrase in lower)
        question_boost = 1 if "?" in text else 0
        emotion_boost = 1 if re.search(r"\b(wow|crazy|insane|amazing|wild|brutal|holy)\b", lower) else 0
        duration = max(1.0, end - start)

        score = 40 + hook_hits * 14 + question_boost * 10 + emotion_boost * 8 + min(12, len(text.split()))
        if 12 <= duration <= 60:
            score += 10
        if scene_hits:
            score += min(12, scene_hits * 4)
        return float(max(35, min(100, score)))

    def suggest_clips(self, transcription: list[dict[str, Any]], scenes: list[dict[str, Any]], clip_count: int = 10) -> list[dict[str, Any]]:
        if not transcription:
            return []

        candidates: list[dict[str, Any]] = []
        scene_times = [scene["time"] for scene in scenes]

        for index, word in enumerate(transcription):
            text = str(word["text"])
            lower = text.lower()
            if any(phrase in lower for phrase in HOOK_PHRASES) or "?" in text:
                start = max(0.0, float(word["start"]) - 2.0)
                end = min(self.duration or float(word["end"]) + 20.0, float(word["end"]) + 24.0)
                scene_hits = sum(1 for scene_time in scene_times if start <= scene_time <= end)
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "hook_score": self._score_window(text, start, end, scene_hits),
                        "reason": "Hook phrase detected" if "?" not in text else "Question-driven engagement moment",
                        "transcript": text,
                    }
                )

            if index % 9 == 0:
                start = max(0.0, float(word["start"]) - 1.5)
                end = min(self.duration or float(word["end"]) + 18.0, start + 28.0)
                scene_hits = sum(1 for scene_time in scene_times if start <= scene_time <= end)
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "hook_score": self._score_window(text, start, end, scene_hits) - 6,
                        "reason": "Highlighted conversational segment",
                        "transcript": text,
                    }
                )

        for scene in scenes:
            start = max(0.0, scene["time"] - 2.0)
            end = min(self.duration or scene["time"] + 20.0, start + 30.0)
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "hook_score": min(100.0, 55.0 + scene.get("confidence", 0.5) * 40.0),
                    "reason": "Scene transition moment",
                    "transcript": "",
                }
            )

        candidates.sort(key=lambda clip: clip["hook_score"], reverse=True)
        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(filtered) >= max(1, clip_count):
                break
            overlap = any(not (candidate["end"] <= existing["start"] + 3 or candidate["start"] >= existing["end"] - 3) for existing in filtered)
            if not overlap:
                candidate["start"] = round(max(0.0, candidate["start"]), 2)
                candidate["end"] = round(min(self.duration or candidate["end"], max(candidate["start"] + 12.0, min(candidate["end"], candidate["start"] + 60.0))), 2)
                candidate["duration"] = round(candidate["end"] - candidate["start"], 2)
                filtered.append(candidate)

        return filtered

    def build_srt(self, transcription: list[dict[str, Any]], clip_start: float, clip_end: float, srt_path: Path) -> None:
        relevant = [item for item in transcription if clip_start <= float(item["start"]) <= clip_end]
        with open(srt_path, "w", encoding="utf-8") as handle:
            for index, segment in enumerate(relevant, start=1):
                start = float(segment["start"]) - clip_start
                end = min(clip_end - clip_start, float(segment["end"]) - clip_start)
                text = str(segment["text"]).strip()
                if not text:
                    continue
                handle.write(f"{index}\n{format_srt_time(start)} --> {format_srt_time(max(start + 0.2, end))}\n{text}\n\n")

    def cut_clip(self, start: float, end: float, output_path: Path, captions: bool = False, transcription: Optional[list[dict[str, Any]]] = None) -> Path:
        duration = max(1.0, end - start)
        temp_output = output_path
        if captions and transcription:
            temp_output = output_path.with_suffix(".tmp.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(self.path),
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if captions and transcription:
            srt_path = output_path.with_suffix(".srt")
            self.build_srt(transcription, start, end, srt_path)
            captioned_output = output_path
            subtitle_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_output),
                "-vf",
                f"subtitles={srt_path}:force_style='FontSize=22,PrimaryColour=&H00FFFFFF,Outline=2,BorderStyle=1'",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(captioned_output),
            ]
            subprocess.run(subtitle_cmd, check=True, capture_output=True)
            if temp_output != output_path and temp_output.exists():
                temp_output.unlink(missing_ok=True)
            srt_path.unlink(missing_ok=True)

        return output_path


def get_json_body(request: Request) -> dict[str, Any]:
    return request._body if hasattr(request, "_body") else {}


async def request_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        return dict(form)
    return {}


def load_analysis_metadata(video_id: str) -> Optional[dict[str, Any]]:
    analysis_path = UPLOAD_DIR / f"{video_id}.json"
    if analysis_path.exists():
        with open(analysis_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def save_analysis_metadata(video_id: str, analysis: dict[str, Any]) -> None:
    with open(UPLOAD_DIR / f"{video_id}.json", "w", encoding="utf-8") as handle:
        json.dump(analysis, handle)


def get_local_video_path(video_id: str) -> Optional[Path]:
    return find_source_file(video_id)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"success": True, "status": "ok"}


@app.post("/process-url")
async def process_url(request: Request):
    try:
        payload = await request_payload(request)
        url = payload.get("url") or payload.get("videoUrl") or payload.get("video_url")
        if not url:
            raise HTTPException(status_code=400, detail="url is required")

        video_id = str(uuid.uuid4())
        local_path = download_video_source(str(url), video_id)
        processor = VideoProcessor(local_path)
        transcription = processor.transcribe()
        scenes = processor.detect_scenes()
        clips = processor.suggest_clips(transcription, scenes, clip_count=10)
        analysis = {
            "video_id": video_id,
            "duration": processor.duration,
            "transcription": transcription,
            "scenes": scenes,
            "clips": clips,
            "source_url": url,
        }
        save_analysis_metadata(video_id, analysis)

        return JSONResponse(
            content={
                "success": True,
                "video_id": video_id,
                "duration": processor.duration,
                "clips": clips,
                "transcription": transcription,
                "source_url": url,
            }
        )
    except Exception as error:
        return JSONResponse(status_code=500, content={"success": False, "error": str(error)})


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_id = str(uuid.uuid4())
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            ext = ".mp4"
        path = UPLOAD_DIR / f"{video_id}{ext}"
        with open(path, "wb") as handle:
            shutil.copyfileobj(file.file, handle)
        duration = ffprobe_duration(path)
        return {"success": True, "video_id": video_id, "duration": duration}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/analyze/{video_id}")
async def analyze_video(video_id: str):
    try:
        path = get_local_video_path(video_id)
        if not path:
            raise HTTPException(status_code=404, detail="Video not found")

        processor = VideoProcessor(path)
        transcription = processor.transcribe()
        scenes = processor.detect_scenes()
        clips = processor.suggest_clips(transcription, scenes, clip_count=10)
        analysis = {
            "video_id": video_id,
            "duration": processor.duration,
            "transcription": transcription,
            "scenes": scenes,
            "clips": clips,
        }
        save_analysis_metadata(video_id, analysis)
        return analysis
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/status/{video_id}")
async def job_status(video_id: str):
    analysis = load_analysis_metadata(video_id)
    if not analysis:
        return {"success": True, "status": "pending", "video_id": video_id}
    return {
        "success": True,
        "status": "complete",
        "video_id": video_id,
        "duration": analysis.get("duration", 0),
        "clip_count": len(analysis.get("clips", [])),
    }


@app.post("/export")
async def export_clip(request: Request):
    try:
        payload = await request_payload(request)

        video_id = payload.get("video_id") or payload.get("videoId")
        source_url = payload.get("videoUrl") or payload.get("video_url") or payload.get("url") or payload.get("source_url")
        start = float(payload.get("start") or payload.get("start_time") or 0)
        end = float(payload.get("end") or payload.get("end_time") or 0)
        captions = str(payload.get("captions") or payload.get("add_subtitles") or "false").lower() in {"1", "true", "yes", "on"}

        if end <= start:
            raise HTTPException(status_code=400, detail="end must be greater than start")

        if video_id:
            source_path = get_local_video_path(str(video_id))
            if not source_path and source_url:
                source_path = download_video_source(str(source_url), str(video_id))
        elif source_url:
            source_id = slugify_url(str(source_url))
            source_path = download_video_source(str(source_url), source_id)
            video_id = source_id
        else:
            raise HTTPException(status_code=400, detail="video_id or videoUrl is required")

        if not source_path:
            raise HTTPException(status_code=404, detail="Video source not found")

        output_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{output_id}.mp4"

        transcription = None
        if captions and video_id:
            analysis = load_analysis_metadata(str(video_id))
            transcription = analysis.get("transcription") if analysis else None
            if not transcription:
                transcription = VideoProcessor(source_path).transcribe()

        processor = VideoProcessor(source_path)
        final_path = processor.cut_clip(start, end, output_path, captions=captions, transcription=transcription)

        return FileResponse(
            path=final_path,
            media_type="video/mp4",
            filename=f"clip_{int(start)}_{int(end)}.mp4",
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
