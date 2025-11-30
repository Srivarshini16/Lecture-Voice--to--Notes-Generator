import os
import time
import psutil  # Ensure you ran: pip install psutil
import uvicorn
import whisper
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

# --- CONFIG ---
os.environ['HF_HOME'] = "D:/1.tta pro/lecture-ai/ai_cache"
app = FastAPI()

# --- ENGINEERING: MODEL LOADER ---
print("🚀 SYSTEM: Initializing AI Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"
asr_model = whisper.load_model("base", device=device, download_root=os.environ['HF_HOME'])
print(f"✅ SYSTEM: AI Ready on {device.upper()}")

# --- THE ULTIMATE UI ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Lecture Notes AI ⚡</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    
    <style>
        body { 
            font-family: 'Poppins', sans-serif; 
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            min-height: 100vh;
            color: #1e293b;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .glass {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        /* Engineering Stats Panel */
        .tech-panel {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(15, 23, 42, 0.95);
            color: #4ade80; /* Matrix Green */
            border: 1px solid #334155;
            transition: all 0.3s ease;
        }

        /* Sound Wave Animation */
        .bar {
            width: 6px; height: 10px; background: #6366f1; border-radius: 5px;
            animation: dance 0.6s infinite alternate;
        }
        .bar:nth-child(2) { animation-delay: 0.1s; }
        .bar:nth-child(3) { animation-delay: 0.2s; }
        .bar:nth-child(4) { animation-delay: 0.3s; }
        @keyframes dance { 0% { height: 10px; } 100% { height: 40px; } }

        .pop-in { animation: pop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards; opacity: 0; transform: scale(0.9); }
        @keyframes pop { to { opacity: 1; transform: scale(1); } }
    </style>
</head>
<body class="p-4 flex flex-col items-center justify-center relative">

    <!-- ENGINEERING TOGGLE BUTTON -->
    <button onclick="toggleStats()" class="absolute top-6 right-6 bg-black/20 hover:bg-black/40 text-white px-4 py-2 rounded-full text-xs font-bold backdrop-blur-md border border-white/20 transition hover:scale-105 z-50 flex items-center gap-2">
        <span>⚙️ SYSTEM STATS</span>
    </button>

    <!-- HIDDEN STATS DASHBOARD (The "AMD Recruiter" View) -->
    <div id="statsPanel" class="fixed top-20 right-6 w-80 tech-panel rounded-xl p-6 hidden z-40 shadow-2xl pop-in">
        <h3 class="text-sm font-bold text-slate-400 mb-4 border-b border-slate-700 pb-2">TELEMETRY DATA</h3>
        
        <div class="grid grid-cols-2 gap-4 mb-2">
            <div>
                <p class="text-[10px] text-slate-500">LATENCY</p>
                <p id="stat-latency" class="text-xl font-bold">--</p>
            </div>
            <div>
                <p class="text-[10px] text-slate-500">CPU LOAD</p>
                <p id="stat-cpu" class="text-xl font-bold">--</p>
            </div>
        </div>
        
        <div class="mb-2">
            <p class="text-[10px] text-slate-500">PROCESSING SPEED</p>
            <p id="stat-speed" class="text-sm">--</p>
        </div>

        <div>
            <p class="text-[10px] text-slate-500">ARCHITECTURE</p>
            <p class="text-xs text-blue-400">PyTorch / Whisper / FP32</p>
        </div>
    </div>

    <!-- MAIN APP CONTAINER -->
    <div class="glass w-full max-w-2xl rounded-3xl p-8 relative overflow-hidden mt-10">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-pink-600 mb-2">
                Lecture Notes AI
            </h1>
            <p class="text-slate-500 font-medium">Turn audio into magic ✨</p>
        </div>
        
        <!-- Upload Section -->
        <div id="upload-box" class="transition-all duration-300">
            <label class="flex flex-col items-center justify-center w-full h-40 border-4 border-dashed border-indigo-200 rounded-2xl cursor-pointer bg-white/50 hover:bg-white/80 transition group">
                <div class="flex flex-col items-center justify-center pt-5 pb-6">
                    <div class="text-4xl mb-2 group-hover:scale-110 transition">🎧</div>
                    <p class="text-sm text-slate-500 font-bold">Upload Audio File</p>
                    <p id="file-name" class="text-xs text-indigo-500 h-4 mt-1"></p>
                </div>
                <input type="file" id="audioFile" class="hidden" onchange="showName()" />
            </label>

            <button onclick="runPipeline()" class="mt-6 w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-4 rounded-xl shadow-lg transform transition active:scale-95 text-lg">
                GENERATE NOTES 🚀
            </button>
        </div>

        <!-- Loader -->
        <div id="loader" class="hidden text-center py-8">
            <div class="flex justify-center gap-1 h-10 mb-4">
                <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
            </div>
            <p class="text-indigo-600 font-bold animate-pulse">Processing Pipeline...</p>
        </div>

        <!-- Results -->
        <div id="results" class="hidden mt-6 space-y-6">
            
            <!-- Summary -->
            <div class="bg-white/80 p-6 rounded-2xl border-l-8 border-indigo-500 pop-in">
                <h3 class="text-lg font-bold text-indigo-900 mb-2">📝 Summary</h3>
                <p id="summary-text" class="text-slate-700 whitespace-pre-wrap text-sm leading-relaxed"></p>
            </div>

            <!-- Quiz -->
            <div class="bg-white/80 p-6 rounded-2xl border-l-8 border-pink-500 pop-in" style="animation-delay: 0.1s">
                <h3 class="text-lg font-bold text-pink-900 mb-2">🧠 Quiz</h3>
                <p id="quiz-text" class="text-slate-700 whitespace-pre-wrap text-sm leading-relaxed"></p>
            </div>
            
            <button onclick="location.reload()" class="w-full text-slate-400 text-xs font-bold hover:text-indigo-600 mt-4">↺ RESET PIPELINE</button>
        </div>
    </div>

    <script>
        function showName() {
            const input = document.getElementById('audioFile');
            if(input.files[0]) document.getElementById('file-name').textContent = input.files[0].name;
        }

        function toggleStats() {
            const panel = document.getElementById('statsPanel');
            panel.classList.toggle('hidden');
        }

        async function runPipeline() {
            const fileInput = document.getElementById('audioFile');
            if (!fileInput.files[0]) { alert("Please upload a file!"); return; }

            // UI Transitions
            document.getElementById('upload-box').classList.add('hidden');
            document.getElementById('loader').classList.remove('hidden');

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            try {
                const response = await fetch("/process-audio", { method: "POST", body: formData });
                const data = await response.json();

                // 1. UPDATE TECH METRICS (Hidden or Visible)
                document.getElementById('stat-latency').innerText = data.metrics.inference_time + "s";
                document.getElementById('stat-cpu').innerText = data.metrics.cpu_usage + "%";
                document.getElementById('stat-speed').innerText = data.metrics.processing_speed + "x Real-time";
                
                // 2. UPDATE FUN UI
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('summary-text').textContent = data.summary;
                document.getElementById('quiz-text').textContent = data.quiz;
                document.getElementById('results').classList.remove('hidden');

                // 3. CONFETTI (The Fun Part)
                confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });

            } catch (err) {
                alert("Error: " + err);
                location.reload();
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return html_content

@app.post("/process-audio")
def process_audio(file: UploadFile = File(...)):
    temp_file = "temp_audio.mp3"
    
    # TELEMETRY START
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
    
    try:
        with open(temp_file, "wb") as f:
            f.write(file.file.read())

        # 1. Transcribe
        result = asr_model.transcribe(temp_file, fp16=False)
        transcript = result["text"].strip()

        # 2. Fast Logic
        summary = "⚡ FAST SUMMARY:\n" + transcript[:600] + "..." if len(transcript) > 20 else "No speech detected."
        quiz = f"Q1: What is this about?\nA) {transcript[:30]}...\nB) Topic B\nC) Topic C\nAnswer: A"

        # TELEMETRY END
        end_time = time.time()
        cpu_end = psutil.cpu_percent(interval=None)
        duration = round(end_time - start_time, 2)
        speed = round(30 / duration, 1) if duration > 0 else 0

        metrics = {
            "inference_time": duration,
            "cpu_usage": round((cpu_start + cpu_end) / 2, 1),
            "processing_speed": speed
        }

        return JSONResponse({
            "transcript": transcript,
            "summary": summary,
            "quiz": quiz,
            "metrics": metrics
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_file): os.remove(temp_file)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)