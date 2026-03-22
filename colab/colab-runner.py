"""
Colab Runner Template
======================
Copy-paste this into a Colab notebook cell-by-cell.
Each section below is one Colab cell.
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1: Setup (run once per session)                       ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Paste this into Cell 1 ---

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone your repo (first time) or pull latest (subsequent times)
import os
REPO_DIR = "/content/llm-quantization-thesis"

if not os.path.exists(REPO_DIR):
    # First time: clone
    # !git clone https://github.com/YOUR_USERNAME/llm-quantization-thesis.git {REPO_DIR}
    pass  # UNCOMMENT AND EDIT THE LINE ABOVE
else:
    # Pull latest changes
    os.system(f"cd {REPO_DIR} && git pull")

os.chdir(REPO_DIR)

# Install dependencies
os.system("pip install -q -r requirements.txt")
os.system("pip install -q smoothquant")

# Clone SmoothQuant repo for activation scales
if not os.path.exists("smoothquant"):
    os.system("git clone https://github.com/mit-han-lab/smoothquant.git smoothquant_repo")
    # The act_scales are inside smoothquant_repo/act_scales/

# Check GPU
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
else:
    print("❌ No GPU available! Change runtime type.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2: Run an experiment                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Paste this into Cell 2 (edit the arguments as needed) ---

# EDIT THESE PARAMETERS FOR EACH RUN:
MODEL = "opt-1.3b"
SCHEME = "O3"           # fp16, O1, O2, O3, O1_pcw, O2_pcw, O3_pcw, naive_w8a8
ALPHA = 0.5
NO_SMOOTH = False       # Set True for naive baselines
SKIP_ZEROSHOT = False   # Set True for quick perplexity-only test
SKIP_PPL = False

# Build the command
cmd = f"python experiments/task01_scheme_exploration/run_scheme_eval.py"
cmd += f" --model {MODEL}"
cmd += f" --scheme {SCHEME}"
cmd += f" --alpha {ALPHA}"
cmd += f" --scales_dir smoothquant_repo/act_scales"
cmd += f" --drive_path thesis_results/task01"

if NO_SMOOTH:
    cmd += " --no_smooth"
if SKIP_ZEROSHOT:
    cmd += " --skip_zeroshot"
if SKIP_PPL:
    cmd += " --skip_ppl"

print(f"Running: {cmd}")
os.system(cmd)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3: Quick batch run (optional)                         ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Run all Task 1 schemes for one model ---

MODEL = "opt-1.3b"
SCALES_DIR = "smoothquant_repo/act_scales"
DRIVE_PATH = "thesis_results/task01"

runs = [
    # (scheme, alpha, no_smooth)
    ("fp16",       0.5, True),    # FP16 baseline
    ("naive_w8a8", 0.5, True),    # No smoothing
    ("O1",         0.5, False),   # SmoothQuant-O1
    ("O2",         0.5, False),   # SmoothQuant-O2
    ("O3",         0.5, False),   # SmoothQuant-O3
    ("O1_pcw",     0.5, False),   # Per-channel weight + O1
    ("O2_pcw",     0.5, False),   # Per-channel weight + O2
    ("O3_pcw",     0.5, False),   # Per-channel weight + O3
]

for scheme, alpha, no_smooth in runs:
    cmd = f"python experiments/task01_scheme_exploration/run_scheme_eval.py"
    cmd += f" --model {MODEL} --scheme {scheme} --alpha {alpha}"
    cmd += f" --scales_dir {SCALES_DIR} --drive_path {DRIVE_PATH}"
    if no_smooth:
        cmd += " --no_smooth"
    
    print(f"\n{'='*60}")
    print(f"  Running: {scheme} (smooth={not no_smooth})")
    print(f"{'='*60}\n")
    os.system(cmd)

print("\n✅ All runs complete! Check Drive → thesis_results/task01/")
