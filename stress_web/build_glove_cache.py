# build_glove_cache.py
# Run this ONCE to download GloVe and build the cache
# Takes about 5 minutes and needs ~900MB disk space

import os, pickle, urllib.request, zipfile, numpy as np

RESULTS_DIR = "E:/StressDetect/results"
GLOVE_DIR   = "E:/StressDetect/glove"
GLOVE_URL   = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_FILE  = os.path.join(GLOVE_DIR, "glove.6B.300d.txt")
CACHE_FILE  = os.path.join(RESULTS_DIR, "glove_cache.pkl")

os.makedirs(GLOVE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if os.path.exists(CACHE_FILE):
    print("Cache already exists:", CACHE_FILE)
else:
    if not os.path.exists(GLOVE_FILE):
        print("Downloading GloVe 6B (862MB) — please wait...")
        zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
        urllib.request.urlretrieve(GLOVE_URL, zip_path,
            reporthook=lambda b,bs,t: print(f"\r{b*bs/1e6:.0f}/{t/1e6:.0f} MB", end=""))
        print("\nExtracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("glove.6B.300d.txt", GLOVE_DIR)
        os.remove(zip_path)
        print("Extracted.")

    print("Building cache (loading 400k words)...")
    glove = {}
    with open(GLOVE_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
            if (i+1) % 100000 == 0:
                print(f"  Loaded {i+1:,} words...")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(glove, f)
    print(f"\nDone! Cache saved to: {CACHE_FILE}")
    print("Restart app.py now — text analysis will use real GloVe embeddings.")