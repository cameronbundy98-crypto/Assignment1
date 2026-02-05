import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def transform_scan(scan, tx, ty, phi):
    """Same intent as original loader.py: rigid transform Nx2 scan."""
    R = rot2(phi)
    return (R @ scan.T).T + np.array([tx, ty])

def se2_compose(p, delta):
    """
    Compose pose p=(x,y,th) with delta=(dx,dy,dth) expressed in p's frame.
    Returns p ⊕ delta.
    """
    x, y, th = p
    dx, dy, dth = delta
    R = rot2(th)
    t = np.array([x, y]) + R @ np.array([dx, dy])
    return np.array([t[0], t[1], wrap_angle(th + dth)], dtype=float)

def se2_inverse(p):
    x, y, th = p
    R = rot2(th).T
    t = -R @ np.array([x, y])
    return np.array([t[0], t[1], wrap_angle(-th)], dtype=float)

def se2_between(p_i, p_j):
    """Return delta that maps i->j in i frame: delta = inv(p_i) ⊕ p_j."""
    inv_i = se2_inverse(p_i)
    return se2_compose(inv_i, p_j)

# -------------------------
# Nearest neighbor helper (KDTree if available)
# -------------------------
def nearest_neighbors(src, dst):
    """
    For each point in src, find nearest point in dst.
    Returns matched_dst (same shape as src) and mean squared error.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        dists, idx = tree.query(src, k=1)
        matched = dst[idx]
        mse = float(np.mean(dists**2))
        return matched, mse
    except Exception:
        # Fallback: brute force (slower but works anywhere)
        # src: (Ns,2), dst:(Nd,2)
        diffs = src[:, None, :] - dst[None, :, :]
        d2 = np.sum(diffs**2, axis=2)
        idx = np.argmin(d2, axis=1)
        matched = dst[idx]
        mse = float(np.mean(d2[np.arange(src.shape[0]), idx]))
        return matched, mse

def best_fit_transform(A, B):
    """
    Find rigid transform (R,t) that maps A -> B (least squares), both Nx2.
    """
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - (R @ centroid_A)
    return R, t

def icp_scan_match(ref_scan, mov_scan, init_delta=None, max_iters=20, tol=1e-6, max_corr_dist=0.5):
    """
    Estimate delta=(dx,dy,dtheta) so that transform_scan(mov_scan, dx,dy,dtheta) aligns to ref_scan.
    The transform is expressed in ref_scan frame (i frame).
    """
    if init_delta is None:
        dx, dy, dth = 0.0, 0.0, 0.0
    else:
        dx, dy, dth = map(float, init_delta)

    src = mov_scan.copy()
    prev_mse = None

    for _ in range(max_iters):
        # Apply current transform to moving scan
        src_w = transform_scan(src, dx, dy, dth)

        matched, mse = nearest_neighbors(src_w, ref_scan)

        # Optional correspondence gating
        # (we approximate by removing pairs farther than max_corr_dist)
        diffs = src_w - matched
        d2 = np.sum(diffs**2, axis=1)
        mask = d2 < (max_corr_dist**2)
        if np.sum(mask) < 10:
            break

        A = src_w[mask]
        B = matched[mask]

        R, t = best_fit_transform(A, B)

        # Update current transform: new_T = (R,t) ∘ old_T
        # For 2D: dth += atan2(R[1,0],R[0,0]); (dx,dy) = R*(dx,dy)+t
        inc_th = np.arctan2(R[1, 0], R[0, 0])
        inc_t = t

        # compose increments
        old_t = np.array([dx, dy])
        new_t = (R @ old_t) + inc_t
        dx, dy = float(new_t[0]), float(new_t[1])
        dth = wrap_angle(dth + float(inc_th))

        if prev_mse is not None and abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    return np.array([dx, dy, dth], dtype=float), (prev_mse if prev_mse is not None else 1e9)

# -------------------------
# Pose graph optimization (SE2, Gauss–Newton)
# -------------------------
def optimize_pose_graph_se2(initial_poses, edges, num_iters=10, anchor_weight=1e6):
    poses = initial_poses.copy()
    N = poses.shape[0]

    for it in range(num_iters):
        H = np.zeros((3 * N, 3 * N))
        b = np.zeros(3 * N)
        total_error = 0.0

        for (i, j, z) in edges:
            xi, yi, ti = poses[i]
            xj, yj, tj = poses[j]
            zx, zy, zt = z

            c, s = np.cos(ti), np.sin(ti)

            # predict pj = pi ⊕ z
            x_pred = xi + c * zx - s * zy
            y_pred = yi + s * zx + c * zy
            t_pred = wrap_angle(ti + zt)

            e = np.array([
                x_pred - xj,
                y_pred - yj,
                wrap_angle(t_pred - tj)
            ])
            total_error += float(e @ e)

            d_xpred_dti = -s * zx - c * zy
            d_ypred_dti =  c * zx - s * zy

            Ji = np.array([
                [1.0, 0.0, d_xpred_dti],
                [0.0, 1.0, d_ypred_dti],
                [0.0, 0.0, 1.0]
            ])

            Jj = np.array([
                [-1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0, -1.0]
            ])

            ii = slice(3 * i, 3 * i + 3)
            jj = slice(3 * j, 3 * j + 3)

            H[ii, ii] += Ji.T @ Ji
            H[ii, jj] += Ji.T @ Jj
            H[jj, ii] += Jj.T @ Ji
            H[jj, jj] += Jj.T @ Jj

            b[ii] += Ji.T @ e
            b[jj] += Jj.T @ e

        # anchor pose 0 at (0,0,0)
        H[0:3, 0:3] += np.eye(3) * anchor_weight

        dx = np.linalg.solve(H, -b)

        for k in range(N):
            poses[k, 0] += dx[3 * k + 0]
            poses[k, 1] += dx[3 * k + 1]
            poses[k, 2] = wrap_angle(poses[k, 2] + dx[3 * k + 2])

        print(f"[PGO] Iteration {it+1}, total error = {total_error:.6f}")

    return poses

# -------------------------
# SLAM pipeline: scans -> poses
# -------------------------
def estimate_poses_from_scans(scanlist,
                             odom_icp_iters=15,
                             odom_max_corr_dist=0.6,
                             loop_stride=25,
                             loop_min_sep=40,
                             loop_dist_thresh=1.5,
                             loop_max_edges=8,
                             loop_icp_iters=20,
                             loop_accept_mse=0.20):
    """
    1) Odometry edges: ICP between consecutive scans
    2) Initial trajectory via composing odometry
    3) Loop closure edges: try pairs i,j based on proximity in initial trajectory
    4) SE(2) pose-graph optimization using all accepted edges
    """
    N = len(scanlist)
    poses = np.zeros((N, 3), dtype=float)
    poses[0] = np.array([0.0, 0.0, 0.0], dtype=float)

    edges = []

    # --- odometry edges ---
    for i in range(N - 1):
        z_ij, mse = icp_scan_match(
            ref_scan=scanlist[i],
            mov_scan=scanlist[i + 1],
            init_delta=None,
            max_iters=odom_icp_iters,
            max_corr_dist=odom_max_corr_dist
        )
        poses[i + 1] = se2_compose(poses[i], z_ij)
        edges.append((i, i + 1, z_ij))

        if i % 25 == 0:
            print(f"[ICP-odom] {i}->{i+1} mse={mse:.4f}  z={z_ij}")

    # --- loop closure candidates ---
    loop_edges = 0
    for i in range(0, N, loop_stride):
        for j in range(i + loop_min_sep, N, loop_stride):
            # proximity test in current estimate (cheap)
            dij = np.linalg.norm(poses[j, :2] - poses[i, :2])
            if dij > loop_dist_thresh:
                continue

            # run ICP between non-consecutive scans to form a loop constraint
            z_ij, mse = icp_scan_match(
                ref_scan=scanlist[i],
                mov_scan=scanlist[j],
                init_delta=None,
                max_iters=loop_icp_iters,
                max_corr_dist=odom_max_corr_dist
            )

            if mse is not None and mse < loop_accept_mse:
                edges.append((i, j, z_ij))
                loop_edges += 1
                print(f"[ICP-loop] accepted {i}->{j} dist={dij:.2f} mse={mse:.4f}")
                if loop_edges >= loop_max_edges:
                    break
        if loop_edges >= loop_max_edges:
            break

    print(f"[Edges] odom={N-1}, loop={loop_edges}, total={len(edges)}")

    # --- pose graph optimization (full SE2) ---
    poses_opt = optimize_pose_graph_se2(poses, edges, num_iters=10)
    return poses_opt

# -------------------------
# IO: load scans + save slamlist
# -------------------------
def load_scans(prefix, fnum, suffix):
    fname = prefix + str(fnum) + "_scans" + suffix
    data = np.load(fname)
    scanlist = [data[k] for k in data]
    return scanlist

def save_slam(prefix, fnum, suffix, poses_opt):
    """
    evaluator.py loads sim_N_slam.npz and expects N arrays (each 3,)
    (it reads them into a list). :contentReference[oaicite:5]{index=5}
    README suggests saving as: np.savez(..., *slamlist). :contentReference[oaicite:6]{index=6}
    """
    slamlist = [poses_opt[i].astype(float) for i in range(poses_opt.shape[0])]
    out = prefix + str(fnum) + "_slam" + suffix
    np.savez(out, *slamlist)
    print(f"Saved SLAM poses to: {out}")
    return out

def load_slam_npz(prefix, fnum, suffix):
    fname = prefix + str(fnum) + "_slam" + suffix
    data = np.load(fname)
    slamlist = [data[k] for k in data]
    return np.asarray(slamlist)

# -------------------------
# Visualization (map + trajectory)
# -------------------------
def plot_map(scanlist, poses, title, out_png):
    all_world = []
    for scan, pose in zip(scanlist, poses):
        pts = transform_scan(scan, pose[0], pose[1], pose[2])
        all_world.append(pts)
    all_world = np.vstack(all_world)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_world[:, 0], all_world[:, 1], s=0.5)
    plt.plot(poses[:, 0], poses[:, 1], linewidth=1.0)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.savefig(out_png, dpi=250)
    plt.show()
    print(f"Saved map image to: {out_png}")

# -------------------------
# Main
# -------------------------
def main(args):
    scanlist = load_scans(args.prefix, args.fnum, args.suffix)

    slam_fname = args.prefix + str(args.fnum) + "_slam" + args.suffix

    if args.force or (not os.path.exists(slam_fname)):
        print("[SLAM] Computing poses from scans...")
        poses_opt = estimate_poses_from_scans(scanlist)
        save_slam(args.prefix, args.fnum, args.suffix, poses_opt)
    else:
        print("[SLAM] Found existing slam file, loading:", slam_fname)
        poses_opt = load_slam_npz(args.prefix, args.fnum, args.suffix)

    # Visualize estimated map (README asks to modify loader to visualize solution) 
    plot_map(
        scanlist,
        poses_opt,
        title="Estimated map (your SLAM solution)",
        out_png=args.prefix + str(args.fnum) + "_slam_map.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 1 loader + SLAM runner")
    parser.add_argument("fnum", type=int, help="N for sim_N_scans.npz")
    parser.add_argument("--prefix", type=str, default="sim_", help="File prefix")
    parser.add_argument("--suffix", type=str, default=".npz", help="File suffix")
    parser.add_argument("--force", action="store_true", help="Recompute even if sim_N_slam.npz exists")
    args = parser.parse_args()
    main(args)
