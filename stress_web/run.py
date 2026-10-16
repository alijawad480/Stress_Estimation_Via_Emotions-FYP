"""run.py — One-click launcher"""
import subprocess, sys, os

print("Installing requirements...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

print("Starting StressDetect...")
os.system(f"{sys.executable} app.py")
