import cv2, numpy as np, heapq
from collections import Counter


in_path = "3076_THE_SOCIAL_NETWORK_01.33.57.154-01.34.01.880.avi"

BLOCK  = 16      # 16x16 is standard
RANGE  = 8       # search range ±R 
USE_ABS_RESIDUALS = True  # abs residuals 0..255 
MV_BITS_METHOD = "huffman" 

DOWNSCALE = 1.0  
STEP_FRAMES = 1  


def build_huffman_code(freq_dict):
    heap, cnt = [], 0
    for sym, f in freq_dict.items():
        heap.append((f, cnt, (sym, ""))); cnt += 1
    if len(heap) == 1:
        sym = next(iter(freq_dict))
        return {sym: "0"}
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        def addb(n, b):
            if isinstance(n[0], tuple):
                return (addb(n[0], b), addb(n[1], b))
            s, c = n; return (s, c + b)
        merged = (addb(n1,'0'), addb(n2,'1'))
        heapq.heappush(heap, (f1+f2, cnt, merged)); cnt += 1
    code = {}
    def collect(n):
        if isinstance(n[0], tuple):
            collect(n[0]); collect(n[1])
        else:
            s, c = n; code[s] = c[::-1]
    _, _, root = heap[0]; collect(root)
    return code

def bit_sum(freq, code): 
    return sum(freq[s]*len(code[s]) for s in freq)


cap = cv2.VideoCapture(in_path, cv2.CAP_MSMF)   # GStreamer for some reason is not opening
ok, prev_bgr = cap.read()
if not ok:
    raise RuntimeError("Cannot read first frame (MSMF).")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)             # rewind so we don't lose the first frame
ok, prev_bgr = cap.read()
if not ok:
    raise RuntimeError("Second read failed after rewind.")


if DOWNSCALE != 1.0:
    prev_bgr = cv2.resize(prev_bgr, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

H, W = prev.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# Collectors
residual_vals = []  # list of residual frames (np.uint8 for abs)
mv_dx_all, mv_dy_all = [], []

frames = 1
while True:
    # read with optional frame stepping
    ok, cur_bgr = cap.read()
    if not ok:
        break
    if STEP_FRAMES > 1:
        for _ in range(STEP_FRAMES-1):
            ok2, _ = cap.read()
            if not ok2:
                break

    if DOWNSCALE != 1.0:
        cur_bgr = cv2.resize(cur_bgr, (W, H), interpolation=cv2.INTER_AREA)
    cur = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    
    pred = np.zeros_like(cur)
    dx_map = np.zeros((H//BLOCK, W//BLOCK), dtype=np.int16)
    dy_map = np.zeros((H//BLOCK, W//BLOCK), dtype=np.int16)

    for by in range(0, H - BLOCK + 1, BLOCK):
        for bx in range(0, W - BLOCK + 1, BLOCK):
            T = cur[by:by+BLOCK, bx:bx+BLOCK]

            x0 = max(0, bx - RANGE)
            y0 = max(0, by - RANGE)
            x1 = min(W - BLOCK, bx + RANGE)
            y1 = min(H - BLOCK, by + RANGE)
            win = prev[y0:y1+BLOCK, x0:x1+BLOCK]
            
            res = cv2.matchTemplate(win, T, cv2.TM_SQDIFF)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            best_x, best_y = min_loc

            dx = (x0 + best_x) - bx
            dy = (y0 + best_y) - by

            pred[by:by+BLOCK, bx:bx+BLOCK] = prev[by+dy:by+dy+BLOCK, bx+dx:bx+dx+BLOCK]
            dx_map[by//BLOCK, bx//BLOCK] = dx
            dy_map[by//BLOCK, bx//BLOCK] = dy

    # residuals
    if USE_ABS_RESIDUALS:
        diff = cv2.absdiff(cur, pred)  # 0..255
        residual_vals.append(diff)
    else:
        raise NotImplementedError("For Γ keep USE_ABS_RESIDUALS=True like Part B baseline.")

    mv_dx_all.extend(dx_map.ravel().tolist())
    mv_dy_all.extend(dy_map.ravel().tolist())

    prev = cur
    frames += 1

cap.release()

# Huffman on residuals 
flat_res = np.concatenate([r.ravel() for r in residual_vals]).astype(np.uint16)  # 0..255
freq_res = Counter(flat_res.tolist())
code_res = build_huffman_code(freq_res)
bits_res = bit_sum(freq_res, code_res)

# MV bits (can be ran with fixed also)
n_diff_frames = frames - 1
blocks_per_frame = (H // BLOCK) * (W // BLOCK)

if MV_BITS_METHOD == "fixed":
    # fixed-length per coordinate based on RANGE
    from math import ceil, log2
    bits_per_comp = int(np.ceil(np.log2(2*RANGE + 1)))  # dx or dy
    n_mv = blocks_per_frame * n_diff_frames
    bits_mv = n_mv * 2 * bits_per_comp
elif MV_BITS_METHOD == "huffman":
    # Huffman per dx and dy (shift to 0..2R for dict keys)
    dx_shift = [d + RANGE for d in mv_dx_all]
    dy_shift = [d + RANGE for d in mv_dy_all]
    freq_dx = Counter(dx_shift); code_dx = build_huffman_code(freq_dx)
    freq_dy = Counter(dy_shift); code_dy = build_huffman_code(freq_dy)
    bits_mv = bit_sum(freq_dx, code_dx) + bit_sum(freq_dy, code_dy)
else:
    raise ValueError("MV_BITS_METHOD must be 'fixed' or 'huffman'")

# Totals & CR 
total_pixels = n_diff_frames * H * W
raw_bits = total_pixels * 8  # baseline: 8 bpp residuals

total_bits = bits_res + bits_mv
CR = raw_bits / total_bits

print(f"Frames total: {frames}, diff-frames: {n_diff_frames}, size: {W}x{H}")
print(f"Residual bits (Huffman): {bits_res/8/1024:.2f} KB")
print(f"MV bits ({MV_BITS_METHOD}): {bits_mv/8/1024:.2f} KB")
print(f"Total bitstream: {total_bits/8/1024:.2f} KB")
print(f"Baseline (raw residuals 8bpp): {raw_bits/8/1024:.2f} KB")
print(f"Compression Ratio (CR): {CR:.2f}x")

# Show Huffman codes
x = input("Do you want to see the Huffman codes? (y/n): ").strip().lower()

if x == "y":
    topk = 255  # change to 255 for all, or any number you like
    print(f"\nTop {topk} residual symbols (value, freq, code):")
    for val, f in freq_res.most_common(topk):
        print(f"value={val:3d}  freq={f}  code={code_res[val]}  len={len(code_res[val])}")
else:
    print("End.")
