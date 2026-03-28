import asyncio
import json
import os
import time
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Ensure you have a .env file with ELEVENLABS_API_KEY=your_key_here
load_dotenv()

app = FastAPI()

# ── Constants ─────────────────────────────────────────────────────────────────
SESSIONS_REQUIRED    = 4
FRAMES_PER_SESSION   = 50
MIN_SESSIONS_TO_TRAIN = 3
DEBOUNCE_LEN         = 20   # frames in primary debounce queue (~0.67s at 30fps)
SMOOTH_WINDOW        = 5    # temporal smoothing window (voted labels)
CONFIDENCE_THRESH    = 0.65 # min avg probability to trigger
PROB_WINDOW          = 10   # frames to average predict_proba over for disambiguation

# ── EMA Smoother (Shock Absorber for Jitter) ──────────────────────────────────
class EMASmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed_state = None

    def update(self, current_features: list[float] | None) -> list[float] | None:
        if current_features is None:
            self.smoothed_state = None
            return None
            
        if self.smoothed_state is None:
            self.smoothed_state = list(current_features)
        else:
            for i in range(len(current_features)):
                self.smoothed_state[i] = (self.alpha * current_features[i]) + \
                                         ((1 - self.alpha) * self.smoothed_state[i])
        
        return list(self.smoothed_state)

feature_smoother = EMASmoother(alpha=0.3)

# ── State ─────────────────────────────────────────────────────────────────────
MODE = "NEUTRAL"
model: Pipeline | None = None

neutral_data:     list[list[float]] = []
gesture_data:     dict[str, list[list[float]]] = {}
gesture_sessions: dict[str, list[int]] = {}

active_gesture_id:   str | None = None
active_session_idx:  int | None = None

pred_queue:    deque = deque(maxlen=DEBOUNCE_LEN)
smooth_queue:  deque = deque(maxlen=SMOOTH_WINDOW)   
feature_queue: deque = deque(maxlen=PROB_WINDOW)     

# The Edge Trigger Latch variables
require_neutral_reset: bool = False
neutral_reset_count: int = 0

active_ws: WebSocket | None = None

gesture_phrases: dict[str, str] = {
    "yes":  "Yes",
    "no":   "No",
    "help": "Help me",
}

gesture_descriptions: dict[str, str] = {}  

# ── Raw Movement Descriptor ────────────────────────────────────────────────────
_FEAT_DESC: list[tuple[str, str, str]] = [
    # Face [0-4]
    ("mouth",                 "opened",               "closed"),
    ("left eye",              "widened",              "narrowed"),
    ("right eye",             "widened",              "narrowed"),
    ("left eyebrow",          "raised",               "lowered"),
    ("right eyebrow",         "raised",               "lowered"),
    # Left vertical extensions [5-9]
    ("left thumb",            "raised",               "lowered"),
    ("left index finger",     "raised",               "lowered"),
    ("left middle finger",    "raised",               "lowered"),
    ("left ring finger",      "raised",               "lowered"),
    ("left pinky",            "raised",               "lowered"),
    # Left curls [10-14]
    ("left thumb",            "extended",             "curled"),
    ("left index finger",     "extended",             "curled"),
    ("left middle finger",    "extended",             "curled"),
    ("left ring finger",      "extended",             "curled"),
    ("left pinky",            "extended",             "curled"),
    # Left spreads [15-16]
    ("left thumb & index",    "spread apart",         "pinched"),
    ("left index & middle",   "spread apart",         "closed"),
    # Right vertical extensions [17-21]
    ("right thumb",           "raised",               "lowered"),
    ("right index finger",    "raised",               "lowered"),
    ("right middle finger",   "raised",               "lowered"),
    ("right ring finger",     "raised",               "lowered"),
    ("right pinky",           "raised",               "lowered"),
    # Right curls [22-26]
    ("right thumb",           "extended",             "curled"),
    ("right index finger",    "extended",             "curled"),
    ("right middle finger",   "extended",             "curled"),
    ("right ring finger",     "extended",             "curled"),
    ("right pinky",           "extended",             "curled"),
    # Right spreads [27-28]
    ("right thumb & index",   "spread apart",         "pinched"),
    ("right index & middle",  "spread apart",         "closed"),
]

def describe_gesture(gesture_id: str) -> str:
    if not neutral_data or gesture_id not in gesture_data:
        return "movement detected"
    neutral_mean  = np.mean(neutral_data, axis=0)
    gesture_mean  = np.mean(gesture_data[gesture_id], axis=0)
    delta         = gesture_mean - neutral_mean
    top_idx       = np.argsort(np.abs(delta))[::-1][:3]
    parts: list[str] = []
    for i in top_idx:
        if abs(delta[i]) < 0.06:
            break
        noun, pos, neg = _FEAT_DESC[i]
        parts.append(f"{noun} {pos if delta[i] > 0 else neg}")
    if not parts:
        return "subtle movement"
    result = parts[0].capitalize()
    if len(parts) > 1:
        result += "; " + "; ".join(parts[1:])
    return result

# ── Feature Engineering ───────────────────────────────────────────────────────
def euclidean(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def extract_face_features(face: dict) -> list[float] | None:
    try:
        mouth_h  = euclidean(face["top_lip"],      face["bottom_lip"])
        mouth_w  = euclidean(face["left_mouth"],   face["right_mouth"])
        denom    = mouth_w + 1e-6
        mar       = mouth_h / denom
        left_ear  = euclidean(face["left_eye_top"],   face["left_eye_bottom"])  / denom
        right_ear = euclidean(face["right_eye_top"],  face["right_eye_bottom"]) / denom
        left_brow = euclidean(face["left_eyebrow"],   face["left_eye_top"])     / denom
        right_brow= euclidean(face["right_eyebrow"],  face["right_eye_top"])    / denom
        return [mar, left_ear, right_ear, left_brow, right_brow]
    except (KeyError, TypeError):
        return None

def extract_hand_features(hands: list[dict]) -> list[float]:
    def hand_features(h: dict | None) -> list[float]:
        if h is None:
            return [1.0] * 10 + [0.2, 0.2]
            
        try:
            palm_w = euclidean(h["index_mcp"], h["pinky_mcp"]) + 1e-6
            palm_h = euclidean(h["middle_mcp"], h["wrist"]) + 1e-6

            def extension(tip_key: str) -> float:
                return euclidean(h[tip_key], h["wrist"]) / palm_h

            def curl(tip_key: str, mcp_key: str) -> float:
                return euclidean(h[tip_key], h[mcp_key]) / palm_w

            return [
                extension("thumb_tip"), extension("index_tip"), extension("middle_tip"), 
                extension("ring_tip"), extension("pinky_tip"),
                curl("thumb_tip", "thumb_mcp"), curl("index_tip", "index_mcp"), 
                curl("middle_tip", "middle_mcp"), curl("ring_tip", "ring_mcp"), 
                curl("pinky_tip", "pinky_mcp"),
                euclidean(h["thumb_tip"], h["index_tip"]) / palm_w,
                euclidean(h["index_tip"], h["middle_tip"]) / palm_w
            ]
        except (KeyError, TypeError):
            return [1.0] * 10 + [0.2, 0.2]

    left  = next((h for h in hands if h.get("handedness") == "Left"),  None)
    right = next((h for h in hands if h.get("handedness") == "Right"), None)
    return hand_features(left) + hand_features(right)

def build_feature_vector(face: dict | None, hands: list[dict]) -> list[float] | None:
    face_feats = extract_face_features(face) if face else None
    hand_feats = extract_hand_features(hands)
    if face_feats is None:
        if any(h != 0.0 for h in hand_feats):
            face_feats = [0.0] * 5
        else:
            return None
    return face_feats + hand_feats

# ── ML ────────────────────────────────────────────────────────────────────────
def train_model() -> bool:
    global model, gesture_descriptions
    if not neutral_data or not gesture_data:
        return False
    X, y = [], []
    for feat in neutral_data:
        X.append(feat); y.append("neutral")
    for gid, frames in gesture_data.items():
        for feat in frames:
            X.append(feat); y.append(gid)
    if len(set(y)) < 2 or len(X) < 10:
        return False
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=min(5, len(X)), weights='distance')),
    ])
    clf.fit(X, y)
    model = clf
    gesture_descriptions = {gid: describe_gesture(gid) for gid in gesture_data}
    return True

def smoothed_predict(raw_label: str) -> str:
    smooth_queue.append(raw_label)
    counts = Counter(smooth_queue)
    return counts.most_common(1)[0][0]

# ── Audio Delivery ─────────────────────────────────────────────────────────────
async def fire_trigger(gesture_id: str, ws: WebSocket):
    phrase = gesture_phrases.get(gesture_id, f"Gesture {gesture_id} detected")
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY", ""))
        audio_bytes = b"".join(
            client.text_to_speech.convert(
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"), # Default Rachel voice
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
    global active_gesture_id, active_session_idx
    global pred_queue, smooth_queue, feature_queue, active_ws, model
    global require_neutral_reset, neutral_reset_count, gesture_descriptions

    await websocket.accept()
    active_ws = websocket
    pred_queue.clear()
    smooth_queue.clear()
    feature_queue.clear()
    feature_smoother.smoothed_state = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # ── Commands ──────────────────────────────────────────────────────
            if "command" in msg:
                cmd = msg["command"]

                if cmd == "set_mode":
                    MODE = msg["mode"]
                    active_gesture_id  = msg.get("gesture_id")
                    active_session_idx = msg.get("session_idx")
                    pred_queue.clear()
                    smooth_queue.clear()
                    feature_queue.clear()
                    await websocket.send_text(json.dumps({
                        "status": "mode_set", "mode": MODE,
                        "gesture_id": active_gesture_id,
                        "session_idx": active_session_idx,
                    }))

                elif cmd == "set_phrase":
                    gesture_phrases[msg["gesture_id"]] = msg["phrase"]

                elif cmd == "clear_session":
                    gid  = msg["gesture_id"]
                    sidx = msg["session_idx"]
                    if gid in gesture_sessions and sidx < len(gesture_sessions[gid]):
                        frames_in_slot = gesture_sessions[gid][sidx]
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
                    gesture_descriptions.clear()
                    model = None
                    pred_queue.clear()
                    smooth_queue.clear()
                    feature_queue.clear()
                    feature_smoother.smoothed_state = None
                    MODE = "NEUTRAL"
                    active_gesture_id  = None
                    active_session_idx = None
                    require_neutral_reset = False
                    neutral_reset_count = 0

                elif cmd == "train":
                    success = train_model()
                    if success:
                        MODE = "INFERENCE"
                        pred_queue.clear()
                        smooth_queue.clear()
                        feature_queue.clear()
                        require_neutral_reset = False
                        neutral_reset_count = 0
                        await websocket.send_text(json.dumps({
                            "status": "trained",
                            "neutral_frames": len(neutral_data),
                            "gestures": {k: len(v) for k, v in gesture_data.items()},
                            "descriptions": gesture_descriptions,
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "status": "train_failed",
                            "reason": "Need neutral + at least one gesture with 3+ complete sessions",
                        }))

                elif cmd == "test_gesture":
                    gid = msg.get("gesture_id")
                    if gid:
                        asyncio.create_task(fire_trigger(gid, websocket))
                        await websocket.send_text(json.dumps({
                            "status": "test_triggered",
                            "gesture_id": gid,
                            "phrase": gesture_phrases.get(gid, f"Gesture {gid}"),
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
                        "descriptions": gesture_descriptions,
                    }))
                continue

            # ── Landmark frame ────────────────────────────────────────────────
            face  = msg.get("face")
            hands = msg.get("hands", [])

            raw_features = build_feature_vector(face, hands)
            features = feature_smoother.update(raw_features)
            
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

            elif MODE == "TRAIN_GESTURE" and active_gesture_id and active_session_idx is not None:
                gid  = active_gesture_id
                sidx = active_session_idx

                gesture_data.setdefault(gid, []).append(features)
                if gid not in gesture_sessions:
                    gesture_sessions[gid] = [0] * SESSIONS_REQUIRED
                gesture_sessions[gid][sidx] += 1
                count = gesture_sessions[gid][sidx]

                await websocket.send_text(json.dumps({
                    "status": "collecting",
                    "label": gid,
                    "session_idx": sidx,
                    "count": count,
                    "done": count >= FRAMES_PER_SESSION,
                }))

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
                raw_pred = model.predict([features])[0]
                smoothed = smoothed_predict(raw_pred)

                # ── 1. THE LATCH: Block triggers until fully returned to Neutral ──
                if require_neutral_reset:
                    # FIX: Use 'smoothed' to prevent micro-jitters from restarting the counter
                    if smoothed == "neutral":
                        neutral_reset_count += 1
                    else:
                        neutral_reset_count = max(0, neutral_reset_count - 2)

                    # FIX: Lowered from 15 to 10 frames (~0.3 seconds of solid resting)
                    if neutral_reset_count >= 10:
                        require_neutral_reset = False
                        neutral_reset_count = 0
                        pred_queue.clear()
                        smooth_queue.clear()
                        feature_queue.clear()
                    
                    await websocket.send_text(json.dumps({
                        "status": "prediction",
                        "label": "Returning to Neutral...",
                        "raw_label": raw_pred,
                        "queue_fill": neutral_reset_count, 
                    }))
                    continue

                # ── 2. Normal Accumulation ("Going Up") ───────────────────────
                feature_queue.append(features)
                pred_queue.append(raw_pred)

                # ── 3. The Trigger Gate ───────────────────────────────────────
                if len(pred_queue) == DEBOUNCE_LEN and smoothed != "neutral":
                    
                    if pred_queue.count(smoothed) >= DEBOUNCE_LEN - 8: 
                        classes      = model.named_steps["knn"].classes_
                        proba_matrix = np.array(model.predict_proba(list(feature_queue)))
                        avg_proba    = proba_matrix.mean(axis=0)
                        class_proba  = dict(zip(classes, avg_proba))

                        if smoothed in class_proba and class_proba[smoothed] >= CONFIDENCE_THRESH:
                            # 🎯 FIRE TRIGGER & ENGAGE THE LATCH
                            require_neutral_reset = True 
                            
                            asyncio.create_task(fire_trigger(smoothed, websocket))
                            await websocket.send_text(json.dumps({
                                "status": "triggered",
                                "gesture":     smoothed,
                                "confidence":  round(class_proba[smoothed], 2),
                                "description": gesture_descriptions.get(smoothed, "movement detected"),
                            }))
                            continue

                await websocket.send_text(json.dumps({
                    "status": "prediction",
                    "label": smoothed,
                    "raw_label": raw_pred,
                    "queue_fill": len(pred_queue),
                }))

    except WebSocketDisconnect:
        active_ws = None

app.mount("/", StaticFiles(directory="static", html=True), name="static")
