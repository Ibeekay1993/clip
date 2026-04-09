from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import uuid
import shutil
import subprocess
import json
from pathlib import Path
import yt_dlp
import whisper
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip

app = FastAPI(title="X-CLIP AI API")

# CORS for Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Lovable domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load AI model once
print("Loading Whisper model...")
model = whisper.load_model("base")

class VideoProcessor:
    def __init__(self, video_path):
        self.path = video_path
        self.clip = VideoFileClip(video_path)
        
    def analyze(self):
        """Full AI analysis pipeline"""
        # Transcribe
        result = model.transcribe(str(self.path), word_timestamps=True)
        transcription = []
        for seg in result["segments"]:
            for word in seg.get("words", []):
                transcription.append({
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"]
                })
        
        # Detect scenes
        scenes = self._detect_scenes()
        
        # Find viral clips
        clips = self._find_viral_clips(transcription, scenes)
        
        return {
            "transcription": transcription,
            "scenes": scenes,
            "clips": clips,
            "duration": self.clip.duration
        }
    
    def _detect_scenes(self):
        cap = cv2.VideoCapture(str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        scenes = []
        prev = None
        
        for i in range(int(self.clip.duration)):
            cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = cv2.absdiff(prev, gray)
                if np.mean(diff) > 25:
                    scenes.append({"time": i, "type": "scene_change"})
            prev = gray
            
        cap.release()
        return scenes
    
    def _find_viral_clips(self, trans, scenes):
        """AI logic to find best moments"""
        clips = []
        text = " ".join([t["text"].lower() for t in trans])
        
        # Hook keywords
        hooks = ["wait for it", "watch till", "you won't believe", "here's why"]
        
        for i, word in enumerate(trans):
            for hook in hooks:
                if hook in word["text"].lower():
                    start = max(0, word["start"] - 3)
                    end = min(start + 60, self.clip.duration)
                    clips.append({
                        "start": start,
                        "end": end,
                        "reason": f"Viral hook: '{hook}'",
                        "score": 0.95
                    })
        
        # If no hooks found, create segments
        if not clips:
            for i in range(0, int(self.clip.duration), 60):
                clips.append({
                    "start": i,
                    "end": min(i + 60, self.clip.duration),
                    "reason": "AI segment",
                    "score": 0.7
                })
        
        return clips[:10]
    
    def cut_clip(self, start, end, output_path, add_captions=False, transcription=None):
        """Precise FFmpeg cut"""
        cmd = [
            "ffmpeg", "-y", "-i", str(self.path),
            "-ss", str(start), "-to", str(end),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
        return output_path

processor = None

@app.post("/process-url")
async def process_url(url: str = Form(...)):
    """Download and analyze video from URL"""
    try:
        video_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{video_id}.mp4"
        
        # Download
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': str(output_path),
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
        
        # Analyze
        proc = VideoProcessor(output_path)
        analysis = proc.analyze()
        
        return JSONResponse(content={
            "success": True,
            "video_id": video_id,
            "duration": duration,
            "clips": analysis["clips"],
            "transcription": analysis["transcription"]
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle direct file upload"""
    try:
        video_id = str(uuid.uuid4())
        ext = file.filename.split(".")[-1]
        path = UPLOAD_DIR / f"{video_id}.{ext}"
        
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Get duration
        clip = VideoFileClip(str(path))
        duration = clip.duration
        clip.close()
        
        return {
            "success": True,
            "video_id": video_id,
            "duration": duration
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/analyze/{video_id}")
async def analyze_video(video_id: str):
    """Run AI analysis on uploaded video"""
    try:
        # Find file
        path = None
        for ext in ["mp4", "mov", "webm"]:
            p = UPLOAD_DIR / f"{video_id}.{ext}"
            if p.exists():
                path = p
                break
        
        if not path:
            raise HTTPException(404, "Video not found")
        
        proc = VideoProcessor(path)
        analysis = proc.analyze()
        
        # Save metadata
        with open(UPLOAD_DIR / f"{video_id}.json", "w") as f:
            json.dump(analysis, f)
        
        return {
            "video_id": video_id,
            "clips": analysis["clips"],
            "duration": analysis["duration"],
            "transcription": analysis["transcription"]
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/export")
async def export_clip(
    video_id: str = Form(...),
    start: float = Form(...),
    end: float = Form(...),
    captions: bool = Form(False)
):
    """Cut and return video clip"""
    try:
        # Find source
        path = None
        for ext in ["mp4", "mov", "webm"]:
            p = UPLOAD_DIR / f"{video_id}.{ext}"
            if p.exists():
                path = p
                break
        
        if not path:
            raise HTTPException(404, "Video not found")
        
        # Load transcription if captions requested
        trans = None
        meta_path = UPLOAD_DIR / f"{video_id}.json"
        if meta_path.exists() and captions:
            with open(meta_path) as f:
                data = json.load(f)
                trans = data.get("transcription", [])
        
        # Cut
        output_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{output_id}.mp4"
        
        proc = VideoProcessor(path)
        proc.cut_clip(start, end, output_path, captions, trans)
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"clip_{int(start)}_{int(end)}.mp4"
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
