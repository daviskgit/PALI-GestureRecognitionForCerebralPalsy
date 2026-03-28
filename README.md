# GestureVoice — Face Gesture TTS

Real-time facial gesture detection → ElevenLabs TTS, running entirely in the browser + a FastAPI backend.

## Architecture

```
Browser (MediaPipe WASM)
  └─ WebSocket (/ws) ──▶ FastAPI backend
                              ├─ Feature engineering (MAR / EAR ratios)
                              ├─ KNN classifier (scikit-learn)
                              └─ ElevenLabs TTS → binary audio back to browser
```

## Quick Start

### 1. Install Python deps

```bash
cd gesture_tts
pip install -r requirements.txt
```

### 2. Set your ElevenLabs key

```bash
cp .env.example .env
# edit .env and paste your key
```

### 3. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the app

Visit **http://localhost:8000** in Chrome or Edge (WebGL + WebAssembly required).

---

## How to Use

1. **Record Neutral** — Click "Record Neutral", sit still with a relaxed face for ~50 frames (~2 sec), click stop.
2. **Add Gesture** — Click "Add Gesture", type a phrase (e.g. *"Play the next song"*), click Record, hold the gesture for ~50 frames, stop.
3. **Repeat** for as many gestures as you want (eyebrows raised, wide smile, mouth open, etc.).
4. **Train & Start Inference** — Click the green button. The KNN model trains in milliseconds.
5. Hold a trained gesture — after 10 consecutive matching frames, ElevenLabs speaks your phrase.

---

## Landmark Indices Used

| Feature | MediaPipe Index |
|---|---|
| Top lip | 13 |
| Bottom lip | 14 |
| Left mouth corner | 61 |
| Right mouth corner | 291 |
| Left eye top | 159 |
| Left eye bottom | 145 |
| Right eye top | 386 |
| Right eye bottom | 374 |
| Left eyebrow | 70 |
| Right eyebrow | 300 |

Features fed to KNN: **MAR** (Mouth Aspect Ratio), **left EAR**, **right EAR**, **left brow lift**, **right brow lift** — all normalised by mouth width so scale-invariant.

---

## File Structure

```
gesture_tts/
├── main.py            # FastAPI backend
├── requirements.txt
├── .env.example
└── static/
    └── index.html     # Everything frontend (MediaPipe + WebSocket + UI)
```
