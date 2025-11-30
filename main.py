import os

# Set Cache Path
os.environ['HF_HOME'] = "D:/1.tta pro/lecture-ai/ai_cache"

import uvicorn
import whisper
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
asr_model = None

@app.on_event("startup")
def load_models():
    global asr_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Whisper (Listening Model) on {device}...")
    
    # We ONLY load Whisper. This saves huge amounts of RAM/CPU.
    asr_model = whisper.load_model("base", device=device, download_root="D:/1.tta pro/lecture-ai/ai_cache")
    print("✅ Whisper Model Loaded! (Skipping heavy summarizers for speed)")

@app.post("/process-audio")
def process_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # 1. Save File
        with open(temp_filename, "wb") as buffer:
            buffer.write(file.file.read())

        # 2. Transcribe (REAL AI)
        print("🎤 AI is listening (Transcribing)...")
        transcript_result = asr_model.transcribe(temp_filename, fp16=False)
        transcript_text = transcript_result["text"].strip()
        
        print("✅ Transcription complete!")

        # 3. Summary (FAST TRICK)
        # Instead of crashing the CPU with another AI, we take the key sentences.
        print("📝 Generating Fast Summary...")
        if len(transcript_text) > 0:
            # Simple logic: Take the first few sentences as the 'Lead'
            summary_text = " ".join(transcript_text.split('.')[:5]) + "."
        else:
            summary_text = "No speech detected."

        # 4. Quiz (FAST TRICK)
        print("❓ Generating Fast Quiz...")
        quiz_text = (
            "1. What was the main topic of this audio?\n"
            "(A) " + transcript_text[:30] + "...\n"
            "(B) Something else\n"
            "Answer: A\n\n"
            "2. Did the AI transcribe this correctly?\n"
            "(A) Yes\n(B) No\n"
            "Answer: A"
        )

        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        print("🚀 Sending response to browser...")
        return JSONResponse(content={
            "transcript": transcript_text,
            "summary": "⚡ FAST SUMMARY (CPU Optimized):\n" + summary_text,
            "quiz": "⚡ FAST QUIZ (CPU Optimized):\n" + quiz_text
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    