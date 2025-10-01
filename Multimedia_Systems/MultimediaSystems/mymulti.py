import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Input
VIDEO_PATH = "3076_THE_SOCIAL_NETWORK_01.33.57.154-01.34.01.880.avi"   
OUT_DIR = Path("diff_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Can't open {VIDEO_PATH}")

# Read
ret, prev = cap.read()
if not ret:
    raise RuntimeError("Not availiable video.")

# we turn to gray (for better difference)
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

mad_values = []      # Mean Absolute Difference 
nz_percent = []      # Percentage |diff| > thresh
sample_diff_paths = []

# threshold
CHANGE_THR = 5

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    diff_signed = gray.astype(np.int16) - prev_gray.astype(np.int16)

    # 0..255 (uint8)
    diff_abs = np.abs(diff_signed).astype(np.uint8)

    # MAD
    mad = diff_abs.mean()
    mad_values.append(mad)

    nz = (diff_abs > CHANGE_THR).sum()
    nz_percent.append(100.0 * nz / diff_abs.size)

    # saving some frames
    if frame_idx in (1, 10, 30):  # you can change to have more
        out_p = OUT_DIR / f"diff_frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_p), diff_abs)
        sample_diff_paths.append(out_p)

    prev_gray = gray

cap.release()

# MAD per frame
plt.figure()
plt.title("Mean Absolute Difference per frame")
plt.xlabel("Frame (n)")
plt.ylabel("MAD (0..255)")
plt.plot(np.arange(1, len(mad_values)+1), mad_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "mad.png", dpi=150)
plt.close()

# % pixels with big difference (over the threshold)
plt.figure()
plt.title(f"Percentage of pixels with |diff| > {CHANGE_THR}")
plt.xlabel("Frame (n)")
plt.ylabel("% pixels")
plt.plot(np.arange(1, len(nz_percent)+1), nz_percent)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "percent.png", dpi=150)
plt.close()

print("Photos and plots' directory: ", OUT_DIR.resolve())
