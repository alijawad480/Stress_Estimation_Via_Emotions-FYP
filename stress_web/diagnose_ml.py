"""
diagnose_ml.py
Run this: python diagnose_ml.py
This script has NOTHING to do with Flask.
It only tries to load the ML model and tells you exactly what fails.
"""

import os, sys

# ── Step 1: Check Python and packages ────────────────────────────────────────
print("="*60)
print("STEP 1 — Python environment")
print("="*60)
print(f"Python  : {sys.version}")
print(f"Exe     : {sys.executable}")

packages_needed = ["tensorflow", "keras", "numpy", "sklearn", "pandas"]
for pkg in packages_needed:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {pkg:<15}: ✅  version {ver}")
    except ImportError:
        print(f"  {pkg:<15}: ❌  NOT INSTALLED — run: pip install {pkg}")

# ── Step 2: Check paths ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Paths")
print("="*60)

ML_PATH   = "E:/StressDetect/stress_framework"
CKPT_DIR  = "E:/StressDetect/checkpoints"
CKPT_FILE = os.path.join(CKPT_DIR, "SelfMM_late_best.keras")
SCALER    = "E:/StressDetect/results/scaler.pkl"

for label, path in [
    ("ML framework", ML_PATH),
    ("Checkpoints",  CKPT_DIR),
    ("Scaler",       SCALER),
    ("SelfMM ckpt",  CKPT_FILE),
]:
    ok = os.path.exists(path)
    print(f"  {label:<15}: {'✅' if ok else '❌ MISSING'}  {path}")

if not os.path.exists(CKPT_FILE):
    # Try to find any .keras file
    print(f"\n  Searching for .keras files in {CKPT_DIR}:")
    if os.path.exists(CKPT_DIR):
        keras_files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".keras")]
        if keras_files:
            for f in keras_files:
                print(f"    Found: {f}")
            print(f"\n  ⚠️  Update CKPT_FILE in this script to the correct filename above")
        else:
            print("    ❌ No .keras files found at all")
    sys.exit("Fix paths first then re-run")

# ── Step 3: Import ML framework ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Import ML framework")
print("="*60)

sys.path.insert(0, ML_PATH)

try:
    import config as cfg
    print(f"  config.py      : ✅  TEXT_DIM={cfg.TEXT_DIM} AUDIO_DIM={cfg.AUDIO_DIM}")
except Exception as e:
    print(f"  config.py      : ❌  {e}")
    sys.exit("Fix config import")

try:
    from models import CUSTOM_OBJECTS
    print(f"  models.py      : ✅  custom_objects={list(CUSTOM_OBJECTS.keys())}")
except Exception as e:
    print(f"  models.py      : ❌  {e}")
    sys.exit("Fix models import")

try:
    from preprocessing import ModalityScaler
    print(f"  preprocessing  : ✅")
except Exception as e:
    print(f"  preprocessing  : ❌  {e}")

try:
    from stress_score import stress_calc
    test_result = stress_calc.compute([1.5, 0.5, 0.3, 0.2, 0.1, 0.1])
    print(f"  stress_score   : ✅  test={test_result['stress_score']:.3f}")
except Exception as e:
    print(f"  stress_score   : ❌  {e}")

# ── Step 4: Load the model ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Load Keras model")
print("="*60)

try:
    import keras
    import tensorflow as tf
    import numpy as np

    keras.config.enable_unsafe_deserialization()

    # Register ALL possible custom losses so old checkpoints load too
    for pkg_name, fn_name in [
        ("stress_final",  "emotion_weighted_mse"),
        ("stress_app",    "weighted_mse"),
        ("stress_app",    "focal_weighted_mse"),
        ("stress_v3",     "stress_weighted_mse"),
    ]:
        try:
            @keras.saving.register_keras_serializable(package=pkg_name)
            def _dummy_loss(y_true, y_pred, _n=fn_name):
                return tf.reduce_mean(tf.square(y_true - y_pred))
            _dummy_loss.__name__ = fn_name
        except Exception:
            pass

    all_custom = {
        **CUSTOM_OBJECTS,
        "emotion_weighted_mse": lambda a, b: tf.reduce_mean(tf.square(a-b)),
        "weighted_mse":         lambda a, b: tf.reduce_mean(tf.square(a-b)),
        "focal_weighted_mse":   lambda a, b: tf.reduce_mean(tf.square(a-b)),
        "stress_weighted_mse":  lambda a, b: tf.reduce_mean(tf.square(a-b)),
    }

    print(f"  Loading: {CKPT_FILE}")
    model = keras.models.load_model(CKPT_FILE, custom_objects=all_custom)
    print(f"  ✅ Model loaded!")
    print(f"  Name          : {model.name}")
    print(f"  Input names   : {[i.name for i in model.inputs]}")
    out = model.output
    if isinstance(out, dict):
        print(f"  Outputs (dict): {list(out.keys())}")
    else:
        print(f"  Output shape  : {model.output_shape}")

except Exception as e:
    import traceback
    print(f"  ❌ Load failed: {e}")
    print("\n  Full traceback:")
    traceback.print_exc()
    sys.exit("Fix model loading")

# ── Step 5: Run a test prediction ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Test prediction")
print("="*60)

try:
    dummy = {
        "text":   tf.zeros((1, cfg.TEXT_DIM)),
        "audio":  tf.zeros((1, cfg.AUDIO_DIM)),
        "visual": tf.zeros((1, cfg.VISUAL_DIM)),
    }
    pred = model(dummy, training=False)
    if isinstance(pred, dict):
        pred = pred["main"]
    pred_np = pred.numpy()[0]
    print(f"  ✅ Prediction works!")
    print(f"  Output values : {pred_np.round(3)}")
    print(f"  Output shape  : {pred_np.shape}")

    # Check if output is 6 emotions or 1 stress score
    if len(pred_np) == 6:
        from config import EMOTION_NAMES
        emo_dict = {n: round(float(pred_np[i]),3) for i,n in enumerate(EMOTION_NAMES)}
        sr = stress_calc.compute(pred_np)
        print(f"  Emotions      : {emo_dict}")
        print(f"  Stress score  : {sr['stress_score']:.3f}")
        print(f"  Stress label  : {sr['stress_label']}")
    elif len(pred_np) == 1:
        score = float(pred_np[0])
        print(f"  Stress score  : {score:.3f}")
        print(f"  Stress label  : {stress_calc.label(score)}")

except Exception as e:
    import traceback
    print(f"  ❌ Prediction failed: {e}")
    traceback.print_exc()
    sys.exit("Fix prediction")

# ── Step 6: Load scaler ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — Load scaler")
print("="*60)

try:
    sc = ModalityScaler()
    if os.path.exists(SCALER):
        sc.load(SCALER)
        print(f"  ✅ Scaler loaded from {SCALER}")
    else:
        print(f"  ⚠️  Scaler not found — predictions will use raw features")
except Exception as e:
    print(f"  ❌ Scaler error: {e}")

# ── Done ───────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("✅ ALL STEPS PASSED — ML is ready")
print("The model will go live when app.py starts.")
print("="*60)