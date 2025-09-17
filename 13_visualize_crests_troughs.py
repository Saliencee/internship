from pathlib import Path
import argparse
import cv2
import numpy as np
from typing import Dict, List, Tuple, Set

ROOT = Path("skeletons_graphfix")
OUT  = Path("crests_troughs_vis")
MODE = "curvature"

SMOOTH_WIN = 5
CURVE_QUANTILE = 0.85
NMS = 5

COL_SMOOTH_WIN = 7
COL_NMS = 7
LABELS = False

Point = Tuple[int, int]
Edge  = Tuple[Point, Point]

def ordered_edge(a: Point, b: Point) -> Edge:
    """Return (a,b) sorted lexicographically as a fixed-length 2-tuple."""
    return (a, b) if a <= b else (b, a)
# ------------------------------------------------------------------------

def ensure_binary_white_on_black(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    return th

OFFSETS_8 = [(-1,-1), (-1,0), (-1,1),
             ( 0,-1),         ( 0,1),
             ( 1,-1), ( 1,0), ( 1,1)]

def neighbors8(pt: Tuple[int,int], mask: np.ndarray) -> List[Tuple[int,int]]:
    r, c = pt
    H, W = mask.shape
    out = []
    for dr, dc in OFFSETS_8:
        rr, cc = r+dr, c+dc
        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] != 0:
            out.append((rr, cc))
    return out

def endpoints_and_junctions(mask: np.ndarray) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    ys, xs = np.where(mask != 0)
    endpoints, junctions = [], []
    for r, c in zip(ys, xs):
        deg = len(neighbors8((r,c), mask))
        if deg == 1:
            endpoints.append((r,c))
        elif deg >= 3:
            junctions.append((r,c))
    return endpoints, junctions

def trace_paths(mask: np.ndarray) -> List[List[Tuple[int,int]]]:
    H, W = mask.shape
    skel_pts: Set[Tuple[int,int]] = set(zip(*np.where(mask != 0)))
    if not skel_pts:
        return []

    deg: Dict[Tuple[int,int], int] = {}
    for p in skel_pts:
        deg[p] = len(neighbors8(p, mask))

    visited_edges: Set[Edge] = set()
    paths: List[List[Tuple[int,int]]] = []

    def push_path(start: Tuple[int,int]):
        for nxt in neighbors8(start, mask):
            edge = ordered_edge(start, nxt)
            if edge in visited_edges:
                continue
            path = [start, nxt]
            visited_edges.add(edge)
            prev, cur = start, nxt
            while True:
                nbrs = neighbors8(cur, mask)
                nbrs = [q for q in nbrs if q != prev]
                if deg.get(cur, 0) != 2:
                    break
                if not nbrs:
                    break
                nxt2 = nbrs[0]
                edge2 = ordered_edge(cur, nxt2)
                if edge2 in visited_edges:
                    break
                path.append(nxt2)
                visited_edges.add(edge2)
                prev, cur = cur, nxt2
            paths.append(path)

    endpoints, junctions = endpoints_and_junctions(mask)
    seeds = endpoints + junctions
    for s in seeds:
        push_path(s)

    remaining = [p for p in skel_pts if all(ordered_edge(p, q) not in visited_edges for q in neighbors8(p, mask))]
    remaining = [p for p in remaining if deg.get(p, 0) == 2]
    seen: Set[Tuple[int,int]] = set()
    for s in remaining:
        if s in seen:
            continue
        nbrs = neighbors8(s, mask)
        if not nbrs:
            continue
        path = [s, nbrs[0]]
        visited_edges.add(ordered_edge(s, nbrs[0]))
        prev, cur = s, nbrs[0]
        seen.add(s)
        while True:
            nbrs2 = neighbors8(cur, mask)
            nbrs2 = [q for q in nbrs2 if q != prev]
            if not nbrs2:
                break
            nxt = nbrs2[0]
            edge = ordered_edge(cur, nxt)
            if edge in visited_edges:
                break
            path.append(nxt)
            visited_edges.add(edge)
            seen.add(cur)
            prev, cur = cur, nxt
            if cur == s:
                break
        if len(path) > 2:
            paths.append(path)

    return paths

def moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    if win % 2 == 0:
        win += 1
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(arr_pad, kernel, mode='valid')

def curvature_signed(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx*dx + dy*dy)**1.5 + 1e-6
    k = (dx*ddy - dy*ddx) / denom
    return k

def local_extrema_indices(arr: np.ndarray, half_window: int, mode: str) -> List[int]:
    L = len(arr)
    idxs: List[int] = []
    for i in range(L):
        lo = max(0, i - half_window)
        hi = min(L, i + half_window + 1)
        window = arr[lo:hi]
        if mode == 'max':
            if arr[i] == window.max() and np.sum(window == arr[i]) == 1:
                idxs.append(i)
        else:
            if arr[i] == window.min() and np.sum(window == arr[i]) == 1:
                idxs.append(i)
    return idxs

def detect_by_curvature(mask: np.ndarray,
                        smooth_win: int = SMOOTH_WIN,
                        curve_quantile: float = CURVE_QUANTILE,
                        nms: int = NMS) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:

    paths = trace_paths(mask)
    all_crests: List[Tuple[int,int]] = []
    all_troughs: List[Tuple[int,int]] = []

    for path in paths:
        if len(path) < 8:
            continue
        rs = np.array([p[0] for p in path], dtype=np.float32)
        cs = np.array([p[1] for p in path], dtype=np.float32)

        rs_s = moving_average(rs, smooth_win)
        cs_s = moving_average(cs, smooth_win)

        k = curvature_signed(cs_s, rs_s)
        abs_k = np.abs(k)
        if np.all(abs_k == 0):
            continue

        thr = np.quantile(abs_k, curve_quantile)
        cand_max = [i for i in local_extrema_indices(k, nms, mode='max') if k[i] > 0 and abs_k[i] >= thr]
        cand_min = [i for i in local_extrema_indices(k, nms, mode='min') if k[i] < 0 and abs_k[i] >= thr]

        all_crests.extend([(int(rs_s[i]), int(cs_s[i])) for i in cand_max])
        all_troughs.extend([(int(rs_s[i]), int(cs_s[i])) for i in cand_min])

    return all_crests, all_troughs

def detect_by_column(mask: np.ndarray,
                     smooth_win: int = COL_SMOOTH_WIN,
                     nms: int = COL_NMS) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:

    H, W = mask.shape
    cols = []
    ys = []
    for x in range(W):
        rows = np.where(mask[:, x] != 0)[0]
        if rows.size == 0:
            continue
        cols.append(x)
        ys.append(int(np.median(rows)))
    if not ys:
        return [], []
    cols = np.array(cols)
    ys = np.array(ys, dtype=np.float32)
    ys_s = moving_average(ys, smooth_win)

    mins = local_extrema_indices(ys_s, nms, mode='min')
    maxs = local_extrema_indices(ys_s, nms, mode='max')

    crests = [(int(ys_s[i]), int(cols[i])) for i in mins]
    troughs = [(int(ys_s[i]), int(cols[i])) for i in maxs]
    return crests, troughs

def draw_points_overlay(mask: np.ndarray,
                        crests: List[Tuple[int,int]],
                        troughs: List[Tuple[int,int]],
                        labels: bool = LABELS) -> np.ndarray:
    if mask.ndim == 2:
        bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        bgr = mask.copy()

    overlay = bgr.copy()
    alpha = 0.25
    cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0, dst=bgr)

    for i, (r, c) in enumerate(crests):
        cv2.circle(bgr, (int(c), int(r)), 3, (0,0,255), thickness=-1)  # red
        if labels:
            cv2.putText(bgr, f"C{i}", (int(c)+3, int(r)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1, cv2.LINE_AA)
    for i, (r, c) in enumerate(troughs):
        cv2.rectangle(bgr, (int(c)-3, int(r)-3), (int(c)+3, int(r)+3), (255,0,0), thickness=1)  # blue
        if labels:
            cv2.putText(bgr, f"T{i}", (int(c)+3, int(r)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1, cv2.LINE_AA)
    return bgr

def process_one(path: Path, out_dir: Path, mode: str, args) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"skip (read error): {path}")
        return False
    mask = ensure_binary_white_on_black(img)

    if mode == "curvature":
        crests, troughs = detect_by_curvature(
            mask,
            smooth_win=args.smooth_win,
            curve_quantile=args.curve_quantile,
            nms=args.nms,
        )
    elif mode == "column":
        crests, troughs = detect_by_column(
            mask,
            smooth_win=args.col_smooth_win,
            nms=args.col_nms,
        )
    else:
        raise ValueError(f"Unknown --mode '{mode}' (use 'curvature' or 'column')")

    vis = draw_points_overlay(mask, crests, troughs, labels=args.labels)

    dest = out_dir / path.relative_to(args.root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), vis)
    if ok:
        print(dest)
    else:
        print(f"failed to write: {dest}")
    return ok

def build_argparser():
    ap = argparse.ArgumentParser(description="Visualize crests/troughs on skeleton images.")
    ap.add_argument("--root", type=Path, default=ROOT, help="Input root directory (skeletons_clean)")
    ap.add_argument("--out", type=Path, default=OUT, help="Output root for overlays")
    ap.add_argument("--mode", choices=["curvature", "column"], default=MODE, help="Detection mode")
    ap.add_argument("--labels", action="store_true", help="Draw tiny labels near points")

    # Curvature params
    ap.add_argument("--smooth-win", type=int, default=SMOOTH_WIN, help="Moving-average window for path smoothing (odd)")
    ap.add_argument("--curve-quantile", type=float, default=CURVE_QUANTILE, help="Keep extrema with |k| >= this quantile (0-1)")
    ap.add_argument("--nms", type=int, default=NMS, help="Non-max suppression half-window along path")

    # Column params
    ap.add_argument("--col-smooth-win", type=int, default=COL_SMOOTH_WIN, help="Moving-average window for y(x) smoothing")
    ap.add_argument("--col-nms", type=int, default=COL_NMS, help="Non-max suppression half-window for column mode")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.smooth_win % 2 == 0:
        args.smooth_win += 1
    if args.col_smooth_win % 2 == 0:
        args.col_smooth_win += 1

    args.out.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_path in sorted(args.root.rglob("*.png")):
        if process_one(src_path, args.out, args.mode, args):
            count += 1
    print(f"Visualized crests/troughs for {count} images into '{args.out.resolve()}'")

if __name__ == "__main__":
    main()
