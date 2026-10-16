# test_ml.py — run: python test_ml.py
import os, sys
from dotenv import load_dotenv

# Force load .env from same directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
print(f".env path : {env_path}")
print(f".env exists: {os.path.exists(env_path)}")
load_dotenv(env_path, override=True)

ml_path  = os.getenv("ML_FRAMEWORK_PATH", "NOT_SET")
ckpt_dir = os.getenv("CHECKPOINT_DIR",    "NOT_SET")
res_dir  = os.getenv("RESULTS_DIR",       "NOT_SET")

print(f"\nML_FRAMEWORK_PATH = {ml_path}  exists={os.path.exists(ml_path)}")
print(f"CHECKPOINT_DIR    = {ckpt_dir}  exists={os.path.exists(ckpt_dir)}")
print(f"RESULTS_DIR       = {res_dir}   exists={os.path.exists(res_dir)}")

ckpt = os.path.join(ckpt_dir, "stress_direct_best.keras")
print(f"\nCheckpoint: {ckpt}  exists={os.path.exists(ckpt)}")

# List what is actually in checkpoints folder
if os.path.exists(ckpt_dir):
    print(f"\nFiles in {ckpt_dir}:")
    for f in os.listdir(ckpt_dir):
        print(f"  {f}")

sys.path.insert(0, ml_path)
try:
    import keras
    keras.config.enable_unsafe_deserialization()
    from models import CUSTOM_OBJECTS
    m = keras.models.load_model(ckpt, custom_objects=CUSTOM_OBJECTS)
    print(f"\nModel loaded OK — output shape: {m.output_shape}")
except Exception as e:
    import traceback
    print(f"\nLoad failed: {e}")
    traceback.print_exc()