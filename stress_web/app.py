"""
app.py — Flask backend
- Local Mistral (no API key)
- LMF/cross_modal stress model
- Local machine (localhost:5000)
"""

import os, json, uuid, logging, sys
from datetime import datetime
from flask import (Flask, request, jsonify, render_template,
                   session, send_from_directory)
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Initialize the logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# Add this right after all imports, before load_dotenv()
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)
log.info(".env loaded from: %s", _env_path)

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── ML framework path ────────────────────────────────────────────────────────
ML_PATH = os.getenv("ML_FRAMEWORK_PATH",
                     "E:/StressDetect/stress_framework")
if os.path.exists(ML_PATH):
    sys.path.insert(0, ML_PATH)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", "stress-detect-secret-key-2024")
CORS(app)

# ── Database ──────────────────────────────────────────────────────────────────
from rag_memory import init_db, purge_old_sessions
init_db()
purge_old_sessions()

# ── Static data ───────────────────────────────────────────────────────────────
with open("data/psychologists.json") as f: PSYCHOLOGISTS = json.load(f)
with open("data/resources.json", encoding="utf-8") as f: RESOURCES = json.load(f)

# ── Load ML model (LMF / cross_modal) ────────────────────────────────────────
ML_AVAILABLE  = False
stress_model  = scaler = stress_calc_fn = None
MODEL_NAME    = "SelfMM"
MODEL_FUSION  = "late"

def try_load_model():
    global ML_AVAILABLE, stress_model, scaler, stress_calc_fn

    ml_path     = os.getenv("ML_FRAMEWORK_PATH", "E:/StressDetect/stress_framework")
    ckpt_dir    = os.getenv("CHECKPOINT_DIR",    "E:/StressDetect/checkpoints")
    results_dir = os.getenv("RESULTS_DIR",       "E:/StressDetect/results")
    ckpt_file   = os.path.join(ckpt_dir, "SelfMM_late_best.keras")

    # add ML path to sys.path
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)

    try:
        import keras
        keras.config.enable_unsafe_deserialization()
        from models        import CUSTOM_OBJECTS
        from preprocessing import ModalityScaler
        from stress_score  import stress_calc

        stress_model   = keras.models.load_model(
            ckpt_file, custom_objects=CUSTOM_OBJECTS)
        stress_calc_fn = stress_calc

        sc = ModalityScaler()
        sp = os.path.join(results_dir, "scaler.pkl")
        if os.path.exists(sp):
            sc.load(sp)
        scaler       = sc
        ML_AVAILABLE = True
        log.info("✅ ML LIVE — SelfMM/late loaded from %s", ckpt_file)

    except Exception as e:
        import traceback
        log.error("ML load failed: %s", e)
        traceback.print_exc()


try_load_model()

# ── Booking agent (local Mistral) ─────────────────────────────────────────────
from agent import BookingAgent
booking_agent = BookingAgent()


# =============================================================================
# Helpers
# =============================================================================

def get_sid():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def _demo_result(modality):
    import random
    score = round(random.uniform(0.25, 0.72), 3)
    label = "Severe" if score > 0.67 else "Moderate" if score > 0.33 else "Low"
    emo   = {
        "anger":    round(random.uniform(0, 2.0), 2),
        "fear":     round(random.uniform(0, 1.8), 2),
        "happy":    round(random.uniform(0, 2.0), 2),
        "sad":      round(random.uniform(0, 1.5), 2),
        "surprise": round(random.uniform(0, 1.2), 2),
        "disgust":  round(random.uniform(0, 1.0), 2),
    }
    dominant = max(emo, key=emo.get)
    return {
        "emotions":          emo,
        "stress_score":      score,
        "stress_label":      label,
        "dominant_emotion":  dominant,
        "valence":           round(random.uniform(-0.5, 0.5), 3),
        "arousal":           round(random.uniform(0.2, 0.8), 3),
        "modality":          modality,
        "ml_model":          "demo",
        "note": ("⚠️  Running in demo mode — ML model path not configured. "
                 "Update ML_FRAMEWORK_PATH in .env to enable live predictions."),
    }

# Add this function ABOVE _ml_inference in app.py

def _check_input_quality(result: dict, modality: str = "text") -> dict:
    """
    Detects empty/random input with no human content.
    Returns clear warning instead of misleading prediction.
    """
    emotions = result.get("emotions", {})
    if not emotions:
        return result

    values   = list(emotions.values())
    max_val  = max(values)
    mean_val = sum(values) / len(values)

    # Rule 1: All emotions near zero = no human content in audio/video
    if modality in ("audio", "video") and max_val < 0.25:
        return {
            "emotions":         {e: 0.0 for e in emotions},
            "stress_score":     0.0,
            "stress_label":     "Unknown",
            "dominant_emotion": "none",
            "valence":          0.0,
            "arousal":          0.0,
            "modality":         modality,
            "ml_model":         result.get("ml_model",""),
            "is_valid":         False,
            "warning": (
                "⚠️ No human content detected. "
                "Please upload a video or audio with a real person "
                "speaking or showing facial expressions."
            )
        }

    # Rule 2: All emotions nearly equal = random noise input
    spread = max(values) - min(values)
    if spread < 0.08 and modality in ("audio", "video"):
        return {
            "emotions":         emotions,
            "stress_score":     0.0,
            "stress_label":     "Unknown",
            "dominant_emotion": "none",
            "valence":          0.0,
            "arousal":          0.0,
            "modality":         modality,
            "ml_model":         result.get("ml_model",""),
            "is_valid":         False,
            "warning": (
                "⚠️ Could not detect clear emotional signals. "
                "Try a clearer recording with a person speaking."
            )
        }

    result["is_valid"] = True
    return result


def _ml_inference(**kw):
    from main import (extract_text_features, extract_audio_features,
                      extract_visual_features, split_video)
    import numpy as np, tensorflow as tf
    from config import TEXT_DIM, AUDIO_DIM, VISUAL_DIM, EMOTION_NAMES

    t_vec = np.zeros(TEXT_DIM,   dtype=np.float32)
    a_vec = np.zeros(AUDIO_DIM,  dtype=np.float32)
    v_vec = np.zeros(VISUAL_DIM, dtype=np.float32)
    has_audio = True; audio_note = ""

    if kw.get("text"):        t_vec = extract_text_features(kw["text"])
    if kw.get("audio_path"):  a_vec = extract_audio_features(kw["audio_path"])
    if kw.get("video_path"):
        sp = split_video(kw["video_path"])
        v_vec = sp["visual_vec"]; has_audio = sp["has_audio"]
        audio_note = sp["audio_note"]
        if has_audio: a_vec = sp["audio_vec"]

    if scaler:
        t_vec, a_vec, v_vec = scaler.transform_single(t_vec, a_vec, v_vec)

    inp  = {"text":   tf.expand_dims(t_vec, 0),
            "audio":  tf.expand_dims(a_vec, 0),
            "visual": tf.expand_dims(v_vec, 0)}
    pred = stress_model(inp, training=False)

    # Handle SelfMM dict output
    if isinstance(pred, dict): pred = pred["main"]
    pred = pred.numpy()[0]   # shape (6,)

    emo_dict     = {n: round(float(pred[i]), 3)
                    for i, n in enumerate(EMOTION_NAMES)}
    stress_result = stress_calc_fn.compute(pred)

    return {
        "emotions":          emo_dict,
        "stress_score":      round(stress_result["stress_score"], 3),
        "stress_label":      stress_result["stress_label"],
        "dominant_emotion":  stress_result["dominant_emotion"],
        "valence":           round(stress_result["valence"], 3),
        "arousal":           round(stress_result["arousal"], 3),
        "has_audio":         has_audio,
        "audio_note":        audio_note,
        "modality":          kw.get("modality","text"),
        "ml_model":          "SelfMM/late",
        "is_valid":          True,
    }


def _derive_emotions(stress_score: float, t_vec, a_vec) -> dict:
    """
    Derives plausible emotion intensities from stress score.
    Since stress_direct only outputs a single score,
    we map it to emotions using Russell's Circumplex logic.
    """
    import numpy as np

    s = stress_score   # 0 to 1

    if s >= 0.67:      # Severe
        return {"anger":   round(s * 2.5, 2),
                "fear":    round(s * 2.0, 2),
                "happy":   round((1-s) * 0.8, 2),
                "sad":     round(s * 1.5, 2),
                "surprise":round(s * 0.8, 2),
                "disgust": round(s * 1.2, 2)}
    elif s >= 0.33:    # Moderate
        return {"anger":   round(s * 1.5, 2),
                "fear":    round(s * 1.2, 2),
                "happy":   round((1-s) * 1.2, 2),
                "sad":     round(s * 1.0, 2),
                "surprise":round(s * 0.9, 2),
                "disgust": round(s * 0.7, 2)}
    else:              # Low
        return {"anger":   round(s * 0.8, 2),
                "fear":    round(s * 0.5, 2),
                "happy":   round((1-s) * 2.2, 2),
                "sad":     round(s * 0.6, 2),
                "surprise":round(s * 0.4, 2),
                "disgust": round(s * 0.3, 2)}

def _save_stress(sid, result, modality):
    from rag_memory import save_stress_record
    save_stress_record(sid, result["stress_score"], result["stress_label"],
                       result["dominant_emotion"], modality)


# =============================================================================
# Page Routes
# =============================================================================

@app.route("/")
def index(): return render_template("index.html")

@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html")

@app.route("/psychologists")
def psychologists_page(): return render_template("psychologists.html")

@app.route("/resources")
def resources_page(): return render_template("resources.html")

@app.route("/chat")
def chat_page(): return render_template("chat.html")


# =============================================================================
# API — Session
# =============================================================================

@app.route("/api/session", methods=["POST"])
def api_session():
    sid  = get_sid()
    name = (request.json or {}).get("name", "User")
    from rag_memory import create_session
    create_session(sid, name)
    return jsonify({"session_id": sid, "name": name,
                    "ml_available": ML_AVAILABLE,
                    "model": f"{MODEL_NAME}/{MODEL_FUSION}"})

@app.route("/api/session/info")
def api_session_info():
    sid = get_sid()
    from rag_memory import get_stress_history, get_bookings
    return jsonify({
        "session_id":     sid,
        "stress_history": get_stress_history(sid, 200),
        "bookings":       get_bookings(sid),
        "ml_available":   ML_AVAILABLE,
        "model":          f"{MODEL_NAME}/{MODEL_FUSION}",
    })


# =============================================================================
# API — Stress Analysis
# =============================================================================

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    sid  = get_sid()
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    # Minimum length check
    if len(text.split()) < 4:
        return jsonify({"error": "Please write at least one full sentence."}), 400
    result = _ml_inference(text=text, modality="text") if ML_AVAILABLE else _demo_result("text")
    result = _check_input_quality(result)
    _save_stress(sid, result, "text")
    return jsonify(result)

@app.route("/api/analyze/audio", methods=["POST"])
def analyze_audio():
    sid = get_sid()
    if "file" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    request.files["file"].save(path)
    try:
        result = _ml_inference(audio_path=path, modality="audio") if ML_AVAILABLE else _demo_result("audio")
        result = _check_input_quality(result)
    except Exception as e:
        log.error("Audio error: %s", e)
        result = _demo_result("audio")
    finally:
        if os.path.exists(path): os.remove(path)
    _save_stress(sid, result, "audio")
    return jsonify(result)

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    sid = get_sid()
    if "file" not in request.files:
        return jsonify({"error": "No video file"}), 400
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
    request.files["file"].save(path)
    try:
        result = _ml_inference(video_path=path, modality="video") if ML_AVAILABLE else _demo_result("video")
        result = _check_input_quality(result)
    except Exception as e:
        log.error("Video error: %s", e)
        result = _demo_result("video")
    finally:
        if os.path.exists(path): os.remove(path)
    _save_stress(sid, result, "video")
    return jsonify(result)


# =============================================================================
# API — Chat (RAG + local Mistral)
# =============================================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    sid = get_sid()
    msg = (request.json or {}).get("message", "").strip()
    if not msg: return jsonify({"error": "No message"}), 400

    from rag_memory import (save_message, get_recent_messages,
                             build_context_summary, create_session)
    create_session(sid)
    save_message(sid, "user", msg)

    context = build_context_summary(sid)
    resp    = booking_agent.process_booking_request(msg, sid, context)
    reply   = resp.get("message", "I'm here to help.")

    save_message(sid, "assistant", reply)
    return jsonify({
        "reply":   reply,
        "action":  resp.get("action", "text"),
        "data":    {k: v for k, v in resp.items()
                    if k not in ("message", "action")},
        "history": get_recent_messages(sid, 20),
    })


# =============================================================================
# API — Psychologists & Booking
# =============================================================================

@app.route("/api/psychologists")
def api_psychologists():
    city   = request.args.get("city")
    fee    = request.args.get("max_fee", type=int)
    spec   = request.args.get("specialization")
    result = PSYCHOLOGISTS
    if city:  result = [p for p in result if p["city"].lower()==city.lower()]
    if fee:   result = [p for p in result if p["fee_pkr"]<=fee]
    if spec:  result = [p for p in result if spec.lower() in p["specialization"].lower()]
    result = sorted(result, key=lambda x: x["rating"], reverse=True)
    return jsonify({"psychologists": result, "total": len(result)})

@app.route("/api/psychologists/<psych_id>/slots")
def api_slots(psych_id):
    psych = next((p for p in PSYCHOLOGISTS if p["id"]==psych_id), None)
    if not psych: return jsonify({"error": "Not found"}), 404
    slots = booking_agent.get_available_slots(psych_id)
    return jsonify({"slots": slots, "psychologist": psych})

@app.route("/api/book", methods=["POST"])
def api_book():
    sid  = get_sid()
    data = request.json or {}
    pid, date, time_ = data.get("psychologist_id"), data.get("date"), data.get("time")
    if not all([pid, date, time_]):
        return jsonify({"error": "Missing fields"}), 400
    result = booking_agent.confirm_booking(sid, pid, date, time_,
                                            data.get("notes",""))
    return jsonify(result)

@app.route("/api/bookings")
def api_bookings():
    from rag_memory import get_bookings
    return jsonify({"bookings": get_bookings(get_sid())})

@app.route("/api/bookings/<int:bid>/cancel", methods=["POST"])
def api_cancel(bid):
    from rag_memory import cancel_booking
    cancel_booking(bid, get_sid())
    return jsonify({"success": True})


# =============================================================================
# API — Resources
# =============================================================================

@app.route("/api/resources")
def api_resources():
    return jsonify(RESOURCES)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    print(f"\n{'='*55}")
    print(f"  StressDetect running at http://localhost:{port}")
    print(f"  Model  : SelfMM/late")
    print(f"  ML live: {ML_AVAILABLE}")
    print(f"{'='*55}\n")
    app.run(debug=debug, port=port, host="127.0.0.1")
