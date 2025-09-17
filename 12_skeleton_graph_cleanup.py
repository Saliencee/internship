from pathlib import Path
import numpy as np
import cv2

from scipy.spatial import cKDTree
from skimage.graph import pixel_graph
from skimage.draw import line as sk_line

ROOT = Path("skeletons_clean")  
OUT  = Path("skeletons_graphfix") 

MAX_BRIDGE_DIST  = 3 
MAX_ALIGN_DEG    = 25 
PRUNE_LEN_THRESH = 4

OUT.mkdir(parents=True, exist_ok=True)

def ensure_01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.unique(img).size > 2:
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = img
    return th > 0

def build_graph(sk_bool):
    res = pixel_graph(sk_bool, connectivity=2)
    if isinstance(res, tuple):
        adj, raw_coords = res[0], np.asarray(res[1])
    else:
        adj = res
        raw_coords = None

    H, W = sk_bool.shape

    if raw_coords is None:
        coords = np.column_stack(np.nonzero(sk_bool))
    else:
        rc = np.asarray(raw_coords)
        if rc.ndim == 1:
            r, c = np.unravel_index(rc.astype(np.int64, copy=False), (H, W))
            coords = np.column_stack([r, c])
        elif rc.ndim == 2 and rc.shape[1] == 2:
            coords = rc
        elif rc.ndim == 2 and rc.shape[0] == 2:
            coords = rc.T
        else:
            coords = np.column_stack(np.nonzero(sk_bool))

    coords = coords.astype(np.int64, copy=False)
    return adj, coords


def degrees(adj):
    return adj.getnnz(axis=1)

def endpoint_indices(adj):
    return np.flatnonzero(degrees(adj) == 1)

def single_neighbor_of(adj, i):
    start, end = adj.indptr[i], adj.indptr[i + 1]
    row_idx = adj.indices[start:end]
    if row_idx.size == 1:
        return int(row_idx[0])
    return None

def unit_vec(v):
    n = np.hypot(v[0], v[1])
    if n == 0:
        return np.array([0.0, 0.0], dtype=float)
    return v / n

def angle_deg(u, v):
    u = unit_vec(u); v = unit_vec(v)
    dot = float(np.clip(u[0]*v[0] + u[1]*v[1], -1.0, 1.0))
    return np.degrees(np.arccos(dot))

def rasterize_line(sk_bool, p0, p1):
    r0, c0 = int(p0[0]), int(p0[1])
    r1, c1 = int(p1[0]), int(p1[1])
    rr, cc = sk_line(r0, c0, r1, c1)
    rr = np.clip(rr, 0, sk_bool.shape[0]-1)
    cc = np.clip(cc, 0, sk_bool.shape[1]-1)
    sk_bool[rr, cc] = True
    return sk_bool

def prune_spurs(sk_bool, max_len=PRUNE_LEN_THRESH):
    changed = True
    while changed:
        changed = False
        adj, coords = build_graph(sk_bool)
        deg = degrees(adj)
        endpoints = np.flatnonzero(deg == 1)
        if endpoints.size == 0:
            break

        to_zero = []

        for i in endpoints:
            path = [i]
            prev = -1
            cur = i
            steps = 0
            while True:
                start, end = adj.indptr[cur], adj.indptr[cur + 1]
                nbrs = adj.indices[start:end]
                if prev >= 0:
                    nbrs = nbrs[nbrs != prev]
                if nbrs.size == 0:
                    break
                nxt = int(nbrs[0]) 
                steps += 1
                path.append(nxt)
                prev, cur = cur, nxt
                cur_deg = adj.indptr[cur + 1] - adj.indptr[cur]
                if cur_deg != 2 or steps > max_len:
                    break

            if steps > 0 and steps <= max_len:
                to_zero.extend(path)

        if to_zero:
            to_zero = np.unique(to_zero)
            rr, cc = coords[to_zero, 0], coords[to_zero, 1]
            sk_bool[rr, cc] = False
            changed = True

    return sk_bool

def bridge_micro_gaps(sk_bool,
                      max_dist=MAX_BRIDGE_DIST,
                      max_align_deg=MAX_ALIGN_DEG):
    adj, coords = build_graph(sk_bool)
    deg = degrees(adj)
    eps = np.flatnonzero(deg == 1)
    if eps.size < 2:
        return sk_bool
    ep_coords = coords[eps]
    tree = cKDTree(ep_coords.astype(float))

    used = set()
    for idx, ep_idx in enumerate(eps):
        if int(ep_idx) in used:
            continue
        nb = single_neighbor_of(adj, int(ep_idx))
        if nb is None:
            continue
        v_tangent = coords[nb] - coords[ep_idx]
        cand_ids = tree.query_ball_point(ep_coords[idx], r=max_dist)

        best = None
        for j in cand_ids:
            if j == idx:
                continue
            other_ep = int(eps[j])
            if other_ep in used:
                continue
            v_gap = coords[other_ep] - coords[ep_idx]
            if np.hypot(v_gap[0], v_gap[1]) < 1e-6:
                continue
            if angle_deg(v_tangent, v_gap) > max_align_deg:
                continue
            nb2 = single_neighbor_of(adj, other_ep)
            if nb2 is not None:
                v_tangent2 = coords[nb2] - coords[other_ep]
                if angle_deg(v_tangent2, -v_gap) > max_align_deg:
                    continue
            d = float(np.hypot(v_gap[0], v_gap[1]))
            if (best is None) or (d < best[0]):
                best = (d, other_ep)

        if best is None:
            continue

        _, other_ep = best

        before = sk_bool.copy()
        sk_bool = rasterize_line(sk_bool, coords[ep_idx], coords[other_ep])

        overlap = np.count_nonzero(before & sk_bool)
        added   = np.count_nonzero(sk_bool) - np.count_nonzero(before)
        if added <= 0: 
            sk_bool = before
            continue

        used.add(int(ep_idx))
        used.add(int(other_ep))
        adj, coords = build_graph(sk_bool)
        deg = degrees(adj)

    return sk_bool

def process_skeleton(img):
    sk = ensure_01(img)           
    sk = bridge_micro_gaps(sk, MAX_BRIDGE_DIST, MAX_ALIGN_DEG)
    sk = prune_spurs(sk, PRUNE_LEN_THRESH)
    return (sk.astype(np.uint8) * 255)

count = 0
for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue

    out = process_skeleton(img)

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest_path), out)
    if not ok:
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Graph-fixed {count} PNGs into '{OUT.resolve()}'")