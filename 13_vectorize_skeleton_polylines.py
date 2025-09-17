from pathlib import Path
from collections import defaultdict
import json, math

import cv2
import numpy as np

# ------------------ CONFIG ------------------
ROOT = Path("skeletons_clean")
OUT  = Path("skeletons_simplified")

# redraw + simplification
EPSILON_PX   = 1.6      # approxPolyDP epsilon (px)
THICKNESS    = 1        # redraw thickness
ANTIALIAS    = True

# NEW: pre/post processing knobs
PRUNE_SPUR_PX     = 2   # remove leaf branches up to this length (px)
RESAMPLE_STEP_PX  = 1.0 # equal arc-length resampling step (px)
CHAIKIN_ITERS     = 1   # 0..2 is usually enough

SAVE_JSON    = False
SAVE_OVERLAY = False

# --------------------------------------------

N8 = [(-1,-1),(-1,0),(-1,1),
      (0,-1),        (0,1),
      (1,-1),(1,0),(1,1)]

def load_binary_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # foreground should be white
    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)
    return bw

def neighbors8(r, c, h, w):
    for dr, dc in N8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc

def build_graph_from_bw(bw: np.ndarray):
    """Foreground pixels become graph nodes; 8-neighbors define edges."""
    h, w = bw.shape
    ys, xs = np.where(bw > 0)
    nodes = [(int(r), int(c)) for r, c in zip(ys, xs)]
    node_set = set(nodes)
    adj = defaultdict(list)
    for r, c in nodes:
        for rr, cc in neighbors8(r, c, h, w):
            if (rr, cc) in node_set:
                adj[(r, c)].append((rr, cc))
    deg = {p: len(adj[p]) for p in adj}
    return nodes, adj, deg

def _find_endpoints_junctions(deg):
    endpoints = [p for p,d in deg.items() if d==1]
    junctions = [p for p,d in deg.items() if d>=3]
    return endpoints, junctions

# ---- Spur pruning: remove short leaf branches before vectorization ----
def prune_short_spurs(bw: np.ndarray, max_len: int = 3) -> np.ndarray:
    """Remove leaf branches up to max_len pixels from endpoints to first fork."""
    bw = (bw > 0).astype(np.uint8) * 255
    _, adj, deg = build_graph_from_bw(bw)

    def walk_from_endpoint(start):
        path = [start]
        prev = None
        curr = start
        steps = 0
        while True:
            nbrs = [n for n in adj.get(curr, []) if n != prev]
            if len(nbrs) == 0:  # isolated pixel
                break
            if len(nbrs) >= 2:  # reached junction
                break
            nxt = nbrs[0]
            steps += 1
            path.append(nxt)
            if deg.get(nxt,0) != 2:
                break
            prev, curr = curr, nxt
            if steps > max_len:
                return []  # too long, don't prune
        return path

    endpoints, _ = _find_endpoints_junctions(deg)
    to_zero = set()
    for ep in endpoints:
        path = walk_from_endpoint(ep)
        if path and len(path) <= max_len:
            to_zero.update(path)
    if to_zero:
        rr, cc = zip(*to_zero)
        bw[np.array(rr), np.array(cc)] = 0
    return bw

# ---- Extract polylines between degree!=2 nodes (endpoints & junctions) ----
def extract_polylines(bw: np.ndarray):
    nodes, adj, deg = build_graph_from_bw(bw)
    node_set = set(nodes)
    endpoints = [p for p in node_set if deg.get(p, 0) == 1]
    junctions = [p for p in node_set if deg.get(p, 0) >= 3]
    deg2 = [p for p in node_set if deg.get(p, 0) == 2]

    visited_edges = set()
    paths = []

    def add_edge(a, b):
        visited_edges.add((a, b))
        visited_edges.add((b, a))  # undirected

    def edge_visited(a, b):
        return (a, b) in visited_edges

    # paths that start/end at endpoints/junctions
    starts = endpoints + junctions
    for s in starts:
        for n in adj.get(s, []):
            if edge_visited(s, n):
                continue
            path = [s]
            prev = s
            curr = n
            add_edge(prev, curr)
            while True:
                path.append(curr)
                d = deg.get(curr, 0)
                if d != 2:  # stop at endpoint/junction
                    break
                a, b = adj[curr][0], adj[curr][1]
                nxt = a if a != prev else b
                if edge_visited(curr, nxt):
                    break
                add_edge(curr, nxt)
                prev, curr = curr, nxt
            paths.append(path)

    # pure loops (entire component degree==2)
    for p in deg2:
        for n in adj[p]:
            if edge_visited(p, n):
                continue
            path = [p]
            prev = p
            curr = n
            add_edge(prev, curr)
            while True:
                path.append(curr)
                a, b = adj[curr][0], adj[curr][1]
                nxt = a if a != prev else b
                if edge_visited(curr, nxt):
                    if nxt == path[0]:
                        path.append(nxt)
                    break
                add_edge(curr, nxt)
                prev, curr = curr, nxt
            if path[0] == path[-1]:
                paths.append(path)

    return paths

# ---- geometry helpers ----
def resample_polyline(points, step=1.0, closed=False):
    """Resample points at ~equal arc-length spacing."""
    if len(points) < 2:
        return points[:]
    pts = np.array([(float(c), float(r)) for r,c in points], dtype=np.float64)
    if closed and not (points[0]==points[-1]):
        pts = np.vstack([pts, pts[0]])
    seglens = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    if np.all(seglens == 0):
        return points[:]
    cum = np.hstack([[0.0], np.cumsum(seglens)])
    total = cum[-1]
    if total == 0:
        return points[:]
    n = max(2, int(np.ceil(total / step)) + 1)
    targets = np.linspace(0, total, n)
    out = []
    j = 0
    for t in targets:
        while j < len(cum)-1 and cum[j+1] < t:
            j += 1
        t0, t1 = cum[j], cum[j+1]
        if t1 == t0:
            x, y = pts[j]
        else:
            alpha = (t - t0) / (t1 - t0)
            x = (1 - alpha) * pts[j,0] + alpha * pts[j+1,0]
            y = (1 - alpha) * pts[j,1] + alpha * pts[j+1,1]
        out.append((float(x), float(y)))
    return [(int(round(y)), int(round(x))) for x,y in out]

def chaikin(points, iters=1, closed=False):
    """Chaikin corner-cutting (stable, no overshoot). Points are (r,c)."""
    if len(points) < 3 or iters<=0:
        return points[:]
    P = [(float(r), float(c)) for r,c in points]
    if closed and P[0] != P[-1]:
        P = P + [P[0]]
    for _ in range(iters):
        Q = []
        rng = range(len(P)-1)
        if not closed:
            Q.append(P[0])
        for i in rng:
            p0 = P[i]
            p1 = P[i+1]
            q = (0.75*p0[0] + 0.25*p1[0], 0.75*p0[1] + 0.25*p1[1])
            r = (0.25*p0[0] + 0.75*p1[0], 0.25*p0[1] + 0.75*p1[1])
            Q.extend([q, r])
        if not closed:
            Q.append(P[-1])
        else:
            Q[-1] = Q[0]
        P = Q
    return [(int(round(r)), int(round(c))) for r,c in P]

def approx_poly(points_rc, epsilon_px=1.5, closed=False):
    if len(points_rc) < 2:
        return points_rc[:]
    pts_xy = np.array([(c, r) for (r, c) in points_rc], dtype=np.float32).reshape((-1, 1, 2))
    peri = cv2.arcLength(pts_xy, closed)
    eps = max(float(epsilon_px), 0.01 * peri)
    approx = cv2.approxPolyDP(pts_xy, eps, closed)
    return [(int(p[0][1]), int(p[0][0])) for p in approx]

def redraw_polylines(shape, polylines, closed_flags=None, thickness=1, antialias=True):
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for i, pl_rc in enumerate(polylines):
        if len(pl_rc) < 2:
            continue
        closed = bool(closed_flags[i]) if closed_flags is not None else False
        pts = np.array([(c, r) for (r, c) in pl_rc], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=closed, color=255, thickness=thickness, lineType=line_type)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out

def vectorize_and_smooth_better(bw,
                                spur_len_px=2,
                                resample_step_px=1.0,
                                chaikin_iters=1,
                                epsilon_px=1.6,
                                thickness=1):
    # 1) prune short spurs
    bw_clean = prune_short_spurs(bw, max_len=spur_len_px)
    # 2) trace polylines
    raw_paths = extract_polylines(bw_clean)
    # 3) resample -> smooth -> simplify
    simplified, closed_flags = [], []
    for path in raw_paths:
        is_closed = (len(path) >= 2 and (path[0] == path[-1] or
                     np.hypot(path[0][0]-path[-1][0], path[0][1]-path[-1][1]) <= 1.5))
        resamp = resample_polyline(path, step=resample_step_px, closed=is_closed)
        smooth = chaikin(resamp, iters=chaikin_iters, closed=is_closed)
        approx = approx_poly(smooth, epsilon_px=epsilon_px, closed=is_closed)
        simplified.append(approx)
        closed_flags.append(is_closed)
    # 4) redraw
    redrawn = redraw_polylines(bw.shape, simplified, closed_flags=closed_flags,
                               thickness=thickness, antialias=True)
    return simplified, closed_flags, redrawn

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_path in sorted(ROOT.rglob("*.png")):
        try:
            bw = load_binary_image(src_path)
        except Exception as e:
            print(f"skip (read error): {src_path} -> {e}")
            continue
        polylines, closed_flags, redrawn = vectorize_and_smooth_better(
            bw,
            spur_len_px=PRUNE_SPUR_PX,
            resample_step_px=RESAMPLE_STEP_PX,
            chaikin_iters=CHAIKIN_ITERS,
            epsilon_px=EPSILON_PX,
            thickness=THICKNESS,
        )
        dest_path = OUT / src_path.relative_to(ROOT)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest_path), redrawn):
            print(f"failed to write: {dest_path}")
            continue
        if SAVE_JSON:
            json_path = dest_path.with_suffix(".json")
            export = [{"closed": bool(closed_flags[i]), "points": polylines[i]}
                      for i in range(len(polylines))]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
        if SAVE_OVERLAY:
            bg = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            fg = cv2.cvtColor(redrawn, cv2.COLOR_GRAY2BGR)
            fg[:, :, 0] = 0; fg[:, :, 2] = 0  # green
            overlay = cv2.addWeighted(bg, 0.7, fg, 0.9, 0)
            cv2.imwrite(str(dest_path.with_name(dest_path.stem + "_overlay.png")), overlay)
        print(dest_path)
        count += 1
    print(f"Vectorized & smoothed {count} PNGs into '{OUT.resolve()}'")

if __name__ == "__main__":
    main()
