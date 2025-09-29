import os
import subprocess
import sys

# Step 1: Check Python version
if sys.version_info < (3, 11):
    sys.exit("Python 3.11 or higher is required.")

# Step 2: Create virtual environment
venv_dir = "venv"
if not os.path.exists(venv_dir):
    subprocess.run([sys.executable, "-m", "venv", venv_dir])

# Step 3: Activate virtual environment and install dependencies
pip = os.path.join(venv_dir, "Scripts", "pip.exe" if os.name == "nt" else "bin/pip")
subprocess.run([pip, "install", "--upgrade", "pip"])
subprocess.run([pip, "install", "-r", "requirements.txt"])

print("Setup complete! Activate the virtual environment and start coding!")
