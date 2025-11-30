from fastapi import FastAPI, UploadFile, File
import whisper
from transformers import pipeline

app = FastAPI()

# ----------- LOAD MODELS (FAST MODE) -----------
print("Loading Whisper model (base)...")
asr_model = whisper.load_model("base")     # faster + good accuracy

print("Loading summarizer model (t5-small)...")
summarizer = pipeline("summarization", model="t5-small")   # very fast

print("Loading quiz generator model (flan-t5-small)...")
qa_gen = pipeline("text2text-generation", model="google/flan-t5-small")  # fast model


# ----------- PROCESS AUDIO ENDPOINT -----------
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):

    # Save uploaded audio
    temp_file = "temp_audio.mp3"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    # ----------- TRANSCRIBE -----------  
    transcript = asr_model.transcribe(temp_file)["text"]

    # ----------- SUMMARY (shorter = faster) -----------  
    summary = summarizer(
        transcript,
        max_length=80,        # shorter summary = faster
        min_length=30,
        do_sample=False
    )[0]["summary_text"]

    # ----------- QUIZ GENERATION (fewer MCQs = faster) -----------  
    prompt = f"Generate 3 simple MCQs based on this summary:\n{summary}"

    quiz = qa_gen(
        prompt,
        max_new_tokens=150     # smaller generation = faster
    )[0]["generated_text"]

    return {
        "transcript": transcript,
        "summary": summary,
        "quiz": quiz
    }

# To run the app: uvicorn api:app --reload