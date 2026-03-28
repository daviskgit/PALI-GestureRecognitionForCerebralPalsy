import asyncio
import json
import os
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sklearn.neighbors import KNeighborsClassifier
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ── Constants ─────────────────────────────────────────────────────────────────
SESSIONS_REQUIRED = 5        # sessions per gesture
FRAMES_PER_SESSION = 50      # frames per session
MIN_SESSIONS_TO_TRAIN = 3    # minimum sessions needed to allow training
DEBOUNCE_LEN = 10

# ── State ─────────────────────────────────────────────────────────────────────
MODE = "NEUTRAL"
model: KNeighborsClassifier | None = None

# neutral_data: flat list of feature vectors
neutral_data: list[list[float]] = []

# gesture_data: gesture_id -> list of feature vectors (across all sessions)
gesture_data: dict[str, list[list[float]]] = {}

# session tracking: gesture_id -> list of session frame counts
gesture_sessions: dict[str, list[int]] = {}

active_gesture_id: str | None = None
active_session_idx: int | None = None   # which session slot is currently recording

pred_queue: deque = deque(maxlen=DEBOUNCE_LEN)
gesture_phrases: dict[str, str] = {}
active_ws: WebSocket | None = None


# ── Feature Engineering ───────────────────────────────────────────────────────

def euclidean(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def extract_face_features(face: dict) -> list[float] | None:
    """5 scale-invariant face features."""
    try:
        mouth_h = euclidean(face["top_lip"], face["bottom_lip"])
        mouth_w = euclidean(face["left_mouth"], face["right_mouth"])
        denom = mouth_w + 1e-6
        mar = mouth_h / denom
        left_ear  = euclidean(face["left_eye_top"],   face["left_eye_bottom"])  / denom
        right_ear = euclidean(face["right_eye_top"],  face["right_eye_bottom"]) / denom
        left_brow = euclidean(face["left_eyebrow"],   face["left_eye_top"])     / denom
        right_brow= euclidean(face["right_eyebrow"],  face["right_eye_top"])    / denom
        return [mar, left_ear, right_ear, left_brow, right_brow]
    except (KeyError, TypeError):
        return None


def extract_hand_features(hands: list[dict]) -> list[float]:
    """
    10 scale-invariant hand features per hand slot (left, right).
    Each hand: 5 fingertip curl ratios normalised by palm width.
    If a hand is absent, zeros are used so the vector length stays fixed.
    """
    # fingertip indices in MediaPipe Hand landmark set (21 points)
    # We encode as keys passed from JS: wrist, thumb_tip, index_tip,
    # middle_tip, ring_tip, pinky_tip, index_mcp, pinky_mcp, thumb_ip
    def hand_features(h: dict | None) -> list[float]:
        if h is None:
            return [0.0] * 5
        try:
            palm_w = euclidean(h["index_mcp"], h["pinky_mcp"]) + 1e-6
            thumb_curl  = euclidean(h["thumb_tip"],  h["wrist"]) / palm_w
            index_curl  = euclidean(h["index_tip"],  h["wrist"]) / palm_w
            middle_curl = euclidean(h["middle_tip"], h["wrist"]) / palm_w
            ring_curl   = euclidean(h["ring_tip"],   h["wrist"]) / palm_w
            pinky_curl  = euclidean(h["pinky_tip"],  h["wrist"]) / palm_w
            return [thumb_curl, index_curl, middle_curl, ring_curl, pinky_curl]
        except (KeyError, TypeError):
            return [0.0] * 5

    left  = next((h for h in hands if h.get("handedness") == "Left"),  None)
    right = next((h for h in hands if h.get("handedness") == "Right"), None)
    return hand_features(left) + hand_features(right)


def build_feature_vector(face: dict | None, hands: list[dict]) -> list[float] | None:
    """Combined 15-dim feature vector. Returns None only if face is missing."""
    face_feats = extract_face_features(face) if face else None
    hand_feats = extract_hand_features(hands)
    if face_feats is None:
        # Still usable if we have at least one hand
        if any(h != 0.0 for h in hand_feats):
            face_feats = [0.0] * 5
        else:
            return None
    return face_feats + hand_feats


# ── ML ────────────────────────────────────────────────────────────────────────

def train_model() -> bool:
    global model
    if not neutral_data or not gesture_data:
        return False
    X, y = [], []
    for feat in neutral_data:
        X.append(feat)
        y.append("neutral")
    for gid, frames in gesture_data.items():
        for feat in frames:
            X.append(feat)
            y.append(gid)
    if len(set(y)) < 2 or len(X) < 10:
        return False
    clf = KNeighborsClassifier(n_neighbors=min(5, len(X)))
    clf.fit(X, y)
    model = clf
    return True


# ── Audio Delivery ─────────────────────────────────────────────────────────────

async def fire_trigger(gesture_id: str, ws: WebSocket):
    phrase = gesture_phrases.get(gesture_id, f"Gesture {gesture_id} detected")
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", ""))
        audio_bytes = b"".join(
            client.text_to_speech.convert(
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
                text=phrase,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
        )
        await ws.send_bytes(audio_bytes)
    except Exception as e:
        await ws.send_text(json.dumps({"error": f"ElevenLabs error: {e}"}))


# ── WebSocket Endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global MODE, neutral_data, gesture_data, gesture_sessions
    global active_gesture_id, active_session_idx, pred_queue, active_ws, model

    await websocket.accept()
    active_ws = websocket
    pred_queue.clear()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # ── Commands ──────────────────────────────────────────────────────
            if "command" in msg:
                cmd = msg["command"]

                if cmd == "set_mode":
                    MODE = msg["mode"]
                    active_gesture_id = msg.get("gesture_id")
                    active_session_idx = msg.get("session_idx")
                    pred_queue.clear()
                    await websocket.send_text(json.dumps({
                        "status": "mode_set", "mode": MODE,
                        "gesture_id": active_gesture_id,
                        "session_idx": active_session_idx,
                    }))

                elif cmd == "set_phrase":
                    gesture_phrases[msg["gesture_id"]] = msg["phrase"]

                elif cmd == "clear_session":
                    gid = msg["gesture_id"]
                    sidx = msg["session_idx"]
                    # Remove frames for this session slot
                    # We track per-session counts so we can remove them
                    if gid in gesture_sessions and sidx < len(gesture_sessions[gid]):
                        frames_in_slot = gesture_sessions[gid][sidx]
                        # Remove the last N frames added for this session
                        # (sessions are appended sequentially)
                        offset = sum(gesture_sessions[gid][:sidx])
                        del gesture_data[gid][offset:offset + frames_in_slot]
                        gesture_sessions[gid][sidx] = 0

                elif cmd == "clear_gesture":
                    gid = msg["gesture_id"]
                    gesture_data.pop(gid, None)
                    gesture_sessions.pop(gid, None)
                    gesture_phrases.pop(gid, None)

                elif cmd == "clear_all":
                    neutral_data.clear()
                    gesture_data.clear()
                    gesture_sessions.clear()
                    gesture_phrases.clear()
                    model = None
                    pred_queue.clear()
                    MODE = "NEUTRAL"
                    active_gesture_id = None
                    active_session_idx = None

                elif cmd == "train":
                    success = train_model()
                    if success:
                        MODE = "INFERENCE"
                        pred_queue.clear()
                        await websocket.send_text(json.dumps({
                            "status": "trained",
                            "neutral_frames": len(neutral_data),
                            "gestures": {k: len(v) for k, v in gesture_data.items()},
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "status": "train_failed",
                            "reason": "Not enough data — need neutral + at least one gesture with 3+ sessions",
                        }))

                elif cmd == "get_status":
                    await websocket.send_text(json.dumps({
                        "status": "ok",
                        "mode": MODE,
                        "neutral_frames": len(neutral_data),
                        "gesture_frames": {k: len(v) for k, v in gesture_data.items()},
                        "gesture_sessions": {k: v for k, v in gesture_sessions.items()},
                        "model_ready": model is not None,
                        "phrases": gesture_phrases,
                    }))
                continue

            # ── Landmark frame ────────────────────────────────────────────────
            face  = msg.get("face")
            hands = msg.get("hands", [])

            features = build_feature_vector(face, hands)
            if features is None:
                continue

            if MODE == "TRAIN_NEUTRAL":
                neutral_data.append(features)
                if len(neutral_data) % 10 == 0:
                    await websocket.send_text(json.dumps({
                        "status": "collecting",
                        "label": "neutral",
                        "count": len(neutral_data),
                    }))

            elif MODE == "TRAIN_GESTURE" and active_gesture_id is not None and active_session_idx is not None:
                gid  = active_gesture_id
                sidx = active_session_idx

                gesture_data.setdefault(gid, []).append(features)
                if gid not in gesture_sessions:
                    gesture_sessions[gid] = [0] * SESSIONS_REQUIRED
                gesture_sessions[gid][sidx] = gesture_sessions[gid][sidx] + 1
                count = gesture_sessions[gid][sidx]

                await websocket.send_text(json.dumps({
                    "status": "collecting",
                    "label": gid,
                    "session_idx": sidx,
                    "count": count,
                    "done": count >= FRAMES_PER_SESSION,
                }))

                # Auto-stop when session is full
                if count >= FRAMES_PER_SESSION:
                    MODE = "NEUTRAL"
                    active_session_idx = None
                    await websocket.send_text(json.dumps({
                        "status": "session_complete",
                        "gesture_id": gid,
                        "session_idx": sidx,
                        "total_frames": len(gesture_data[gid]),
                        "sessions": gesture_sessions[gid],
                    }))

            elif MODE == "INFERENCE" and model is not None:
                pred = model.predict([features])[0]
                pred_queue.append(pred)

                if len(pred_queue) == DEBOUNCE_LEN and pred != "neutral":
                    counts: dict[str, int] = {}
                    for p in pred_queue:
                        counts[p] = counts.get(p, 0) + 1
                    top_label, top_count = max(counts.items(), key=lambda x: x[1])
                    if top_count == DEBOUNCE_LEN and top_label != "neutral":
                        pred_queue.clear()
                        asyncio.create_task(fire_trigger(top_label, websocket))
                        await websocket.send_text(json.dumps({
                            "status": "triggered",
                            "gesture": top_label,
                        }))

                await websocket.send_text(json.dumps({
                    "status": "prediction",
                    "label": pred,
                    "queue_fill": len(pred_queue),
                }))

    except WebSocketDisconnect:
        active_ws = None


# ── Static files ───────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
