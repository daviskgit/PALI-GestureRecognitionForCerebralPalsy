# Pali — Face & Hand Gesture TTS

Real-time facial expression and hand gesture detection → ElevenLabs TTS, running entirely in the browser + a FastAPI backend.

## Architecture

```
Browser (MediaPipe WASM)
  └─ WebSocket (/ws) ──▶ FastAPI backend
                              ├─ Feature engineering (34D vector)
                              │    ├─ 8 face features  (mouth, jaw, eyes, brows)
                              │    ├─ 12 left hand features  (lifts, curls, spreads)
                              │    ├─ 12 right hand features (lifts, curls, spreads)
                              │    └─ 2 hand-presence flags
                              ├─ SVM classifier (scikit-learn RBF, StandardScaler)
                              │    ├─ Per-gesture region masking (ignores irrelevant features)
                              │    └─ Personal neutral baseline subtracted before classification
                              └─ ElevenLabs TTS → binary audio back to browser
```

---

## Setup

### Prerequisites

- **Python 3.10+**
- **Chrome or Edge** (WebGL + WebAssembly required for MediaPipe)
- An **ElevenLabs account** — free tier works ([elevenlabs.io](https://elevenlabs.io))

---

### 1. Clone the repo

```bash
git clone <repo-url>
cd gesture_tts
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following to `.env`:

```
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here
```

- **`ELEVENLABS_API_KEY`** — Found in your ElevenLabs dashboard under Profile → API Key.
- **`ELEVENLABS_VOICE_ID`** — The ID of the voice you want to use. Find it in the ElevenLabs Voice Library. The default if omitted is Rachel (`21m00Tcm4TlvDq8ikWAM`).

### 5. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the app

Visit **http://localhost:8000** in Chrome or Edge.

---

## How to Use

### Step 1 — Record Neutral

Click **Record Neutral** and sit still with a relaxed face and hands at rest for about 5 seconds (150 frames are collected automatically). This establishes your personal baseline — all gesture features are computed relative to it.

### Step 2 — Add a Gesture

1. Click **Add Gesture** and type the phrase you want spoken (e.g. *"Play the next song"*).
2. You'll see 4 session slots. Click **Record** on the first slot, hold your gesture for ~2 seconds (50 frames), then release.
3. Repeat for each session slot. You need at least 3 complete sessions to train.
4. Supported gestures include any combination of:
   - Face: mouth open, jaw drop, eye widening/narrowing, eyebrow raises, brow furrow, winks
   - Hands: individual finger raises, curls, spreads, fist, open hand, pointing, etc.

### Step 3 — Train & Start Inference

Click the **Train & Start** button. The SVM trains in milliseconds and inference begins immediately.

### Step 4 — Trigger gestures

Hold a trained gesture steadily. After the debounce queue fills (~0.67 seconds of consistent detection at 65%+ confidence), ElevenLabs speaks your phrase. You must fully return to neutral before the same gesture can trigger again.

---

## Inference Pipeline Details

| Parameter | Value | Description |
|---|---|---|
| Neutral frames | 150 | Frames collected for personal baseline |
| Sessions per gesture | 4 (min 3) | Recording sessions required per gesture |
| Frames per session | 50 (~1.7s) | Frames collected per session |
| Debounce window | 20 frames | Consecutive frames needed to trigger |
| Confidence threshold | 0.65 | Minimum avg SVM probability to fire |
| Smoothing window | 5 frames | Temporal vote smoothing |
| Neutral reset | 10 frames | Consecutive neutral frames required before re-arming |
| Hold-off | 8 frames | Ignored frames after latch releases |

**Edge trigger latch:** Once a gesture fires, a latch engages that blocks re-triggering until the model sees 10 consecutive neutral frames, followed by an 8-frame hold-off. This prevents double-firing from sustained holds.

---

## Feature Engineering

### Face (8 features, indices 0–7)

All features are normalized by face scale so they are position and distance invariant.

| Feature | Description |
|---|---|
| Mouth open | Vertical lip gap / face height |
| Jaw drop | Jaw–lip distance / face height |
| Right eye EAR | Eye aspect ratio (eye open vs. closed) |
| Left eye EAR | Eye aspect ratio (independent — enables wink detection) |
| Brow outer raise | Outer brow tier height relative to eye midpoint |
| Brow peak raise | Arch peak height — primary raise/furrow indicator |
| Brow inner raise | Inner brow height — concern/surprise signal |
| Brow furrow | Horizontal gap between inner brow peaks |

### Hands (12 features per hand, indices 8–31)

Each hand contributes 5 lifts + 5 curls + 2 spreads, normalized by palm dimensions.

| Block | Features |
|---|---|
| Lifts [0–4] | Per-finger 3D elevation (y + z axis from MCP) — thumb, index, middle, ring, pinky |
| Curls [5–9] | Tip-to-MCP Euclidean distance — distinguishes extended vs. bent raised fingers |
| Spreads [10–11] | Thumb–index spread; index–middle spread |

### Presence flags (indices 32–33)

Binary flags indicating whether the left and right hands are visible in the frame.

### Region masking

At train time, a binary mask is computed per gesture that identifies which body regions (face, left hand, right hand) contributed meaningfully to that gesture's signal. At inference, features from irrelevant regions are zeroed out before classification — so a face gesture won't misfire based on incidental hand position changes.

---

## File Structure

```
gesture_tts/
├── main.py            # FastAPI backend — feature engineering, SVM, WebSocket, TTS
├── requirements.txt
├── .env               # Your API keys (not committed)
└── static/
    └── index.html     # Full frontend — MediaPipe WASM, WebSocket client, UI
```

---

## Troubleshooting

**Camera not starting** — Make sure you're using Chrome or Edge and have granted camera permissions. Safari does not support the required WebAssembly features.

**"Train failed"** — You need at least 150 neutral frames and at least one gesture with 3+ complete sessions (50 frames each).

**Gesture fires too easily / false positives** — Re-record neutral with more variety of your resting expression. More neutral data = fewer false positives.

**No audio / ElevenLabs error** — Check that `ELEVENLABS_API_KEY` is set correctly in `.env` and that the server was restarted after editing the file.

**Port already in use** — Change the port: `uvicorn main:app --port 8001`
