import asyncio
import json
import os
import time
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dotenv import load_dotenv

# Ensure you have a .env file with ELEVENLABS_API_KEY=your_key_here
load_dotenv()

app = FastAPI()

# ── Constants ─────────────────────────────────────────────────────────────────
SESSIONS_REQUIRED      = 4
FRAMES_PER_SESSION     = 50
NEUTRAL_FRAMES         = 150  # more neutral variety = fewer false positives
MIN_SESSIONS_TO_TRAIN  = 3
DEBOUNCE_LEN         = 20   # frames in primary debounce queue (~0.67s at 30fps)
SMOOTH_WINDOW        = 5    # temporal smoothing window (voted labels)
CONFIDENCE_THRESH    = 0.65 # min avg probability to trigger
PROB_WINDOW          = 10   # frames to average predict_proba over for disambiguation

# ── State ─────────────────────────────────────────────────────────────────────
MODE = "NEUTRAL"
model: Pipeline | None = None

neutral_data:     list[list[float]] = []
gesture_data:     dict[str, list[list[float]]] = {}
gesture_sessions: dict[str, list[int]] = {}
neutral_mean_vec: list[float] | None = None  # personal baseline for relative normalization

# Feature region slices within the 34D vector
# Keep in sync with build_feature_vector: [0:8]=face, [8:20]=left, [20:32]=right, [32:34]=flags
_FACE_SLICE  = slice(0,  8)
_LEFT_SLICE  = slice(8,  20)
_RIGHT_SLICE = slice(20, 32)
_FLAG_SLICE  = slice(32, 34)

gesture_masks: dict[str, np.ndarray] = {}  # per-gesture binary feature mask

active_gesture_id:   str | None = None
active_session_idx:  int | None = None

pred_queue:    deque = deque(maxlen=DEBOUNCE_LEN)
smooth_queue:  deque = deque(maxlen=SMOOTH_WINDOW)   
feature_queue: deque = deque(maxlen=PROB_WINDOW)     

# The Edge Trigger Latch variables
require_neutral_reset: bool = False
neutral_reset_count:   int  = 0
holdoff_count:         int  = 0   # frames to ignore after latch releases
HOLDOFF_FRAMES               = 8  # ~0.27s of confirmed rest before re-accumulating

active_ws: WebSocket | None = None

gesture_phrases: dict[str, str] = {
    "yes":  "Yes",
    "no":   "No",
    "help": "Help me",
}

gesture_descriptions: dict[str, str] = {}  

# ── Raw Movement Descriptor ────────────────────────────────────────────────────
_FEAT_DESC: list[tuple[str, str, str]] = [
    # Face [0-7]  (eyes separate for wink; brows grouped bilaterally)
    ("mouth",               "opened",       "closed"),
    ("jaw",                 "dropped",      "raised"),
    ("right eye",           "widened",      "narrowed"),
    ("left eye",            "widened",      "narrowed"),
    ("eyebrows outer",      "raised",       "lowered"),
    ("eyebrows peak",       "raised",       "lowered"),
    ("eyebrows inner",      "raised",       "lowered"),
    ("brows",               "relaxed",      "furrowed"),
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
    # Hand presence flags [29-30]
    ("left hand",             "present",              "absent"),
    ("right hand",            "present",              "absent"),
]

_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Hand region layout within the full feature vector (must match build_feature_vector)
_HAND_REGIONS = [
    ("left",  8,  20),  # (side, start_idx, end_idx)
    ("right", 20, 32),
]
# Within each 12-feature hand block: lifts[0:5], curls[5:10], spreads[10:12]
_HAND_LIFT_SLICE  = slice(0, 5)
_HAND_CURL_SLICE  = slice(5, 10)

def _finger_phrase(side: str, indices: list[int], verb: str) -> str:
    """Turn a list of active finger indices into a natural phrase.
    e.g. side='right', indices=[1,2], verb='raised' → 'right index and middle fingers raised'
    """
    names = [_FINGER_NAMES[i] for i in indices]
    if len(names) == 5:
        finger_str = "all fingers"
    elif len(names) == 1:
        finger_str = f"{names[0]} finger" if names[0] != "thumb" else "thumb"
    else:
        finger_str = ", ".join(names[:-1]) + f" and {names[-1]} fingers"
        # thumb + others: keep "thumb" without "finger" suffix
        if "thumb" in names:
            parts = [n if n == "thumb" else f"{n}" for n in names]
            finger_str = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return f"{side} {finger_str} {verb}"


def describe_gesture(gesture_id: str) -> str:
    """Describe the dominant movement of a gesture, grouping multiple active fingers
    of the same type into a single natural phrase rather than listing individually."""
    if not neutral_data or gesture_id not in gesture_data:
        return "movement detected"

    neutral_mean = np.mean(neutral_data, axis=0)
    gesture_mean = np.mean(gesture_data[gesture_id], axis=0)
    delta        = gesture_mean - neutral_mean
    peak         = np.max(np.abs(delta))

    if peak < 0.04:
        return "subtle movement"

    # Threshold: include any feature within 60% of the peak delta
    # This ensures thumb (often slightly smaller delta than longer fingers) is included
    active_threshold = peak * 0.40

    parts: list[tuple[float, str]] = []  # (score, phrase) — sorted by score descending

    # ── Face features (indices 0-7): report individually, only the strongest ──
    face_deltas = [(i, delta[i]) for i in range(8) if abs(delta[i]) >= active_threshold]
    if face_deltas:
        best_i, best_d = max(face_deltas, key=lambda x: abs(x[1]))
        noun, pos, neg = _FEAT_DESC[best_i]
        parts.append((abs(best_d), f"{noun} {pos if best_d > 0 else neg}"))

    # ── Hand features: group fingers by movement type ──
    for side, start, end in _HAND_REGIONS:
        block = delta[start:end]  # 12-element slice for this hand

        # Lifts — which fingers are raised/lowered together
        lift_deltas = [(i, block[i]) for i in range(5) if abs(block[i]) >= active_threshold]
        if lift_deltas:
            positive = [i for i, d in lift_deltas if d > 0]
            negative = [i for i, d in lift_deltas if d < 0]
            score = max(abs(d) for _, d in lift_deltas)
            if positive:
                parts.append((score, _finger_phrase(side, positive, "raised")))
            if negative:
                parts.append((score, _finger_phrase(side, negative, "lowered")))

        # Curls — which fingers are extended/curled together
        curl_deltas = [(i, block[5 + i]) for i in range(5) if abs(block[5 + i]) >= active_threshold]
        if curl_deltas:
            positive = [i for i, d in curl_deltas if d > 0]
            negative = [i for i, d in curl_deltas if d < 0]
            score = max(abs(d) for _, d in curl_deltas)
            if positive:
                parts.append((score, _finger_phrase(side, positive, "extended")))
            if negative:
                parts.append((score, _finger_phrase(side, negative, "curled")))

    if not parts:
        return "subtle movement"

    # Sort by score, take top 2 distinct body regions
    parts.sort(key=lambda x: x[0], reverse=True)
    seen_sides: set[str] = set()
    result_phrases: list[str] = []
    for score, phrase in parts:
        side_key = phrase.split()[0]  # "left", "right", "mouth", "jaw", etc.
        if side_key not in seen_sides:
            seen_sides.add(side_key)
            result_phrases.append(phrase)
        if len(result_phrases) == 2:
            break

    result = result_phrases[0].capitalize()
    if len(result_phrases) > 1:
        result += "; " + result_phrases[1]
    return result

# ── Feature Engineering ───────────────────────────────────────────────────────
def euclidean(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def extract_face_features(face: dict) -> list[float] | None:
    """8 expression features — all position/scale invariant.

    Eyes are tracked individually (needed for wink detection).
    Brows are tracked as bilateral groups — people raise/furrow both brows together,
    and the SVM should learn the combined movement as a single signal.

    Three brow points per side (outer, peak, inner) are averaged so the peak of
    the arch — which moves most — contributes fully without creating separate
    left/right features that would split the signal.
    """
    try:
        face_h   = abs(face["jaw"][1] - face["left_eye_top"][1]) + 1e-6
        eye_span = abs(face["right_eye_top"][0] - face["left_eye_top"][0]) + 1e-6

        left_eye_w  = abs(face["left_eye_inner"][0]  - face["left_eye_outer"][0])  + 1e-6
        right_eye_w = abs(face["right_eye_inner"][0] - face["right_eye_outer"][0]) + 1e-6

        mouth_open = (face["bottom_lip"][1] - face["top_lip"][1])    / face_h
        jaw_drop   = (face["jaw"][1]        - face["bottom_lip"][1]) / face_h

        # EAR per eye — kept separate so winks register as one eye closing
        # (mp-left = user's right due to mirrored display)
        right_ear = (face["left_eye_bottom"][1]  - face["left_eye_top"][1])  / left_eye_w
        left_ear  = (face["right_eye_bottom"][1] - face["right_eye_top"][1]) / right_eye_w

        # Bilateral brow raise using the eye midpoint as reference
        eye_mid_y = (face["left_eye_top"][1] + face["right_eye_top"][1]) / 2

        # Average all three brow points per side → single raise signal per tier
        # Outer tier: follows full brow lift
        brow_outer_raise = (eye_mid_y - (
            face["left_eyebrow_outer"][1] + face["right_eyebrow_outer"][1]
        ) / 2) / eye_span

        # Peak tier: arch peak moves most — primary raise/furrow indicator
        brow_peak_raise = (eye_mid_y - (
            face["left_eyebrow_peak"][1] + face["right_eyebrow_peak"][1]
        ) / 2) / eye_span

        # Inner tier: inner brow rises during concern/surprise independently
        brow_inner_raise = (eye_mid_y - (
            face["left_eyebrow_inner"][1] + face["right_eyebrow_inner"][1]
        ) / 2) / eye_span

        # Furrow: horizontal gap between inner brow peaks — shrinks when brows pull together
        furrow = (face["right_eyebrow_inner"][0] - face["left_eyebrow_inner"][0]) / eye_span

        return [mouth_open, jaw_drop, right_ear, left_ear,
                brow_outer_raise, brow_peak_raise, brow_inner_raise, furrow]
    except (KeyError, TypeError):
        return None

def extract_hand_features(hands: list[dict]) -> list[float]:
    """12 features per hand (24D total): 5 3D lifts + 5 curls + 2 spreads.

    Lift combines y-axis (gravity direction) and z-axis (depth toward camera):
    both increase when a finger lifts off a flat surface. Using 3D euclidean
    with the mcp as origin captures finger elevation regardless of hand rotation.
    """
    def hand_features(h: dict | None) -> list[float]:
        if h is None:
            return [0.0] * 12
        try:
            palm_w = euclidean(h["index_mcp"], h["pinky_mcp"]) + 1e-6
            palm_scale = euclidean(h["middle_mcp"], h["wrist"]) + 1e-6

            def lift(tip_key: str, mcp_key: str) -> float:
                # y lift (up in image = negative y direction in MediaPipe)
                y_lift = (h[mcp_key][1] - h[tip_key][1]) / palm_scale
                # z lift (toward camera = more negative z in MediaPipe hand coords)
                z_lift = (h[mcp_key][2] - h[tip_key][2]) / palm_scale
                # Combine: both signals reinforce when finger genuinely raised
                return y_lift + z_lift

            def curl(tip_key: str, mcp_key: str) -> float:
                return euclidean(h[tip_key], h[mcp_key]) / palm_w

            # Vertical lifts (primary signal for surface-resting hand gestures)
            thumb_lift  = lift("thumb_tip",   "thumb_mcp")
            index_lift  = lift("index_tip",   "index_mcp")
            middle_lift = lift("middle_tip",  "middle_mcp")
            ring_lift   = lift("ring_tip",    "ring_mcp")
            pinky_lift  = lift("pinky_tip",   "pinky_mcp")

            # Curls (distinguish extended vs. bent for raised fingers)
            thumb_curl  = curl("thumb_tip",   "thumb_mcp")
            index_curl  = curl("index_tip",   "index_mcp")
            middle_curl = curl("middle_tip",  "middle_mcp")
            ring_curl   = curl("ring_tip",    "ring_mcp")
            pinky_curl  = curl("pinky_tip",   "pinky_mcp")

            # Spreads (thumb abduction, index-middle separation)
            thumb_spread = euclidean(h["thumb_tip"],  h["index_tip"])  / palm_w
            idx_spread   = euclidean(h["index_tip"],  h["middle_tip"]) / palm_w

            return [
                thumb_lift, index_lift, middle_lift, ring_lift, pinky_lift,
                thumb_curl, index_curl, middle_curl, ring_curl, pinky_curl,
                thumb_spread, idx_spread,
            ]
        except (KeyError, TypeError):
            return [0.0] * 12

    left  = next((h for h in hands if h.get("handedness") == "Left"),  None)
    right = next((h for h in hands if h.get("handedness") == "Right"), None)
    return hand_features(left) + hand_features(right)

def build_feature_vector(face: dict | None, hands: list[dict]) -> list[float] | None:
    """34D feature vector: 8 face + 12 left hand + 12 right hand + 2 hand-presence flags."""
    face_feats = extract_face_features(face) if face else None
    hand_feats = extract_hand_features(hands)
    left_present  = 1.0 if any(h.get("handedness") == "Left"  for h in hands) else 0.0
    right_present = 1.0 if any(h.get("handedness") == "Right" for h in hands) else 0.0
    presence = [left_present, right_present]
    if face_feats is None:
        if left_present or right_present:
            face_feats = [0.0] * 8
        else:
            return None
    return face_feats + hand_feats + presence

# ── ML ────────────────────────────────────────────────────────────────────────
def center(vec: list[float]) -> list[float]:
    """Subtract the personal neutral mean so all features are relative to the user's rest pose."""
    if neutral_mean_vec is None:
        return vec
    return list(np.array(vec) - np.array(neutral_mean_vec))


def compute_gesture_mask(gid: str, neutral_mean: np.ndarray) -> np.ndarray:
    """Return a binary mask (34D) that is 1.0 for regions this gesture actually uses
    and 0.0 for regions that are irrelevant noise.

    A region is considered active if its mean absolute delta from neutral exceeds
    a threshold. Presence flags are always kept. This way a face gesture trained
    while the hand is in some position will have hand features zeroed out, so
    different hand positions at inference time don't affect recognition.
    """
    gesture_mean = np.mean(gesture_data[gid], axis=0)
    delta = np.abs(gesture_mean - neutral_mean)
    mask = np.zeros(len(neutral_mean))
    mask[_FLAG_SLICE] = 1.0  # always keep presence flags

    region_threshold = 0.04  # mean abs delta required for a region to be considered active
    for sl in (_FACE_SLICE, _LEFT_SLICE, _RIGHT_SLICE):
        if delta[sl].mean() > region_threshold:
            mask[sl] = 1.0

    # Fallback: if nothing clears the threshold, use everything
    if mask[_FACE_SLICE.start:_RIGHT_SLICE.stop].sum() == 0:
        mask[:] = 1.0
    return mask


def apply_mask(vec: list[float], gid: str) -> list[float]:
    """Apply the gesture's region mask to a centered feature vector."""
    if gid not in gesture_masks:
        return vec
    return list(np.array(vec) * gesture_masks[gid])


def train_model() -> bool:
    global model, gesture_descriptions, neutral_mean_vec, gesture_masks
    if not neutral_data or not gesture_data:
        return False
    # Compute personal neutral baseline
    neutral_mean_vec = np.mean(neutral_data, axis=0).tolist()
    neutral_arr = np.array(neutral_mean_vec)

    # Compute per-gesture masks before building training data
    gesture_masks = {gid: compute_gesture_mask(gid, neutral_arr) for gid in gesture_data}

    X, y = [], []
    for feat in neutral_data:
        X.append(center(feat)); y.append("neutral")
    for gid, frames in gesture_data.items():
        for feat in frames:
            # Apply mask so the SVM never sees irrelevant region noise during training
            X.append(apply_mask(center(feat), gid)); y.append(gid)
    if len(set(y)) < 2 or len(X) < 10:
        return False
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=5, gamma='scale', probability=True, class_weight='balanced')),
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
    global require_neutral_reset, neutral_reset_count, holdoff_count, gesture_descriptions
    global neutral_mean_vec, gesture_masks

    await websocket.accept()
    active_ws = websocket
    pred_queue.clear()
    smooth_queue.clear()
    feature_queue.clear()

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
                    neutral_mean_vec = None
                    gesture_masks.clear()
                    gesture_descriptions.clear()
                    model = None
                    pred_queue.clear()
                    smooth_queue.clear()
                    feature_queue.clear()
                    MODE = "NEUTRAL"
                    active_gesture_id  = None
                    active_session_idx = None
                    require_neutral_reset = False
                    neutral_reset_count   = 0
                    holdoff_count         = 0

                elif cmd == "train":
                    success = train_model()
                    if success:
                        MODE = "INFERENCE"
                        pred_queue.clear()
                        smooth_queue.clear()
                        feature_queue.clear()
                        require_neutral_reset = False
                        neutral_reset_count   = 0
                        holdoff_count         = 0
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
                elif cmd == "get_feature_stats":
                    if not neutral_data or not gesture_data or neutral_mean_vec is None:
                        await websocket.send_text(json.dumps({"status": "feature_stats", "ready": False}))
                    else:
                        neutral_arr = np.array(neutral_mean_vec)
                        gestures_stats = {}
                        for gid in gesture_data:
                            g_mean = np.mean(gesture_data[gid], axis=0)
                            delta = (g_mean - neutral_arr).tolist()
                            mask = gesture_masks.get(gid, np.ones(len(neutral_mean_vec))).tolist()
                            gestures_stats[gid] = {
                                "frames": len(gesture_data[gid]),
                                "delta": delta,
                                "mask": mask,
                                "description": gesture_descriptions.get(gid, "movement detected"),
                            }
                        await websocket.send_text(json.dumps({
                            "status": "feature_stats",
                            "ready": True,
                            "neutral_frames": len(neutral_data),
                            "gestures": gestures_stats,
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
                await websocket.send_text(json.dumps({
                    "status": "collecting",
                    "label": "neutral",
                    "count": len(neutral_data),
                    "done": len(neutral_data) >= NEUTRAL_FRAMES,
                }))
                if len(neutral_data) >= NEUTRAL_FRAMES:
                    MODE = "NEUTRAL"

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
                features = center(features)
                raw_pred = model.predict([features])[0]
                # Apply the predicted gesture's region mask and re-predict.
                # This zeroes out irrelevant regions (e.g. hand position for a face gesture)
                # so the final classification only depends on the features that matter.
                if raw_pred != "neutral":
                    masked = apply_mask(features, raw_pred)
                    raw_pred = model.predict([masked])[0]
                    features = masked
                smoothed = smoothed_predict(raw_pred)

                # ── Hold-off: brief pause after latch releases ────────────────
                if holdoff_count > 0:
                    holdoff_count -= 1
                    await websocket.send_text(json.dumps({
                        "status": "prediction",
                        "label": "neutral", "raw_label": "neutral", "queue_fill": 0,
                    }))
                    continue

                # ── KNN gate: don't start accumulating while at rest ───────────
                if len(pred_queue) == 0 and raw_pred == "neutral":
                    await websocket.send_text(json.dumps({
                        "status": "prediction",
                        "label": "neutral", "raw_label": "neutral", "queue_fill": 0,
                    }))
                    continue

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
                        holdoff_count = HOLDOFF_FRAMES
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
                        classes      = model.named_steps["svm"].classes_
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
