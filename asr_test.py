import whisper

print("Loading Whisper model...")
model = whisper.load_model("small")

audio_path = "sample.mp3"   # your test audio

print("Transcribing...")
result = model.transcribe(audio_path)

print("\n===== TRANSCRIPT =====\n")
print(result["text"])
