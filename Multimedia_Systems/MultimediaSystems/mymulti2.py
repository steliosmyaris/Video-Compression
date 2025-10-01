import cv2, heapq, numpy as np
from collections import Counter, defaultdict

# Reading
cap = cv2.VideoCapture("3076_THE_SOCIAL_NETWORK_01.33.57.154-01.34.01.880.avi", cv2.CAP_MSMF)
ok, prev = cap.read()
if not ok:
    raise RuntimeError("Cannot read first frame (MSMF).")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start after test read
ok, prev = cap.read()
if not ok:
    raise RuntimeError("Second read failed after rewind.")

prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.int16)

H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

abs_values = []   # για |In+1 - In|
signed_values = [] # για (In+1 - In) + 255

frames = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = gray - prev
    # abs residual 0..255
    diff_abs = np.clip(np.abs(diff), 0, 255).astype(np.uint8)
    abs_values.append(diff_abs)
    # signed mapped 0..510
    signed_values.append((diff + 255).astype(np.int16))
    prev = gray
    frames += 1
cap.release()

N_diff = frames  # difference frames
total_pixels = N_diff * H * W
print(f"Frames: {frames+1} total, diff-frames: {N_diff}, size: {W}x{H}, pixels: {total_pixels}")

# Histogram 
abs_flat = np.concatenate([d.ravel() for d in abs_values])
freq_abs = Counter(abs_flat.tolist())

# SIGNED mapped (0..510)
signed_flat = np.concatenate([d.ravel() for d in signed_values])
freq_signed = Counter(signed_flat.tolist())

# Huffman 
def build_huffman_code(freq_dict):
    # min-heap of (freq, count, node) to make it stable
    heap, counter = [], 0
    for sym, f in freq_dict.items():
        heap.append((f, counter, (sym, "")))  # leaf node: (symbol, code)
        counter += 1
    if len(heap) == 1:
        # edge case: only one symbol -> give only 1 bit
        f, _, (sym, _) = heap[0]
        return {sym: "0"}
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        # we add '0' an '1' to the branch 
        def add_bit(node, b):
            if isinstance(node[0], tuple):
                return (add_bit(node[0], b), add_bit(node[1], b))
            sym, code = node
            return (sym, code + b)
        # merge the subtrees
        merged = (n1, n2)
        merged = add_bit(n1, '0'), add_bit(n2, '1')
        heapq.heappush(heap, (f1 + f2, counter, merged))
        counter += 1
    # code
    code = {}
    def collect(node):
        if isinstance(node[0], tuple):
            collect(node[0]); collect(node[1])
        else:
            sym, bits = node
            code[sym] = bits[::-1]  # reverse cause we added them to the right
    _, _, root = heap[0]
    collect(root)
    return code

code_abs = build_huffman_code(freq_abs)
code_signed = build_huffman_code(freq_signed)


def bit_length_sum(freq_dict, codebook):
    return sum(freq_dict[s]*len(codebook[s]) for s in freq_dict)

compressed_bits_abs = bit_length_sum(freq_abs, code_abs)
compressed_bits_signed = bit_length_sum(freq_signed, code_signed)

raw_bits_abs = total_pixels * 8   # baseline: 8 bits/pixel
raw_bits_signed = total_pixels * 9

print(f"[ABS] Raw: {raw_bits_abs/8/1024:.2f} KB | Compressed: {compressed_bits_abs/8/1024:.2f} KB | CR={raw_bits_abs/ compressed_bits_abs:.2f}x")
print(f"[SIGNED] Raw(9b): {raw_bits_signed/8/1024:.2f} KB | Compressed: {compressed_bits_signed/8/1024:.2f} KB | CR={raw_bits_signed/ compressed_bits_signed:.2f}x")

# See Huffman codes 
x = input("Do you want to see the Huffman codes? (y/n): ").strip().lower()

if x == "y":
    topk = 255  # default fallback
    print(f"\nTop {topk} ABS symbols (value, freq, code):")
    for sym, f in freq_abs.most_common(topk):
        print(f"value={sym:3d}  freq={f}  code={code_abs[sym]}  len={len(code_abs[sym])}")
else:
    print("End.")
