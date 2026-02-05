import numpy as np
import argparse


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative_pose(p1, p2):
    """
    Compute the relative pose from p1 to p2, expressed in p1's frame.

    Parameters:
        p1, p2: arrays of shape (3,) representing (x, y, theta).

    Returns:
        rel: array of shape (3,) with (dx, dy, dtheta) in p1's frame.
    """
    dx_world = p2[0] - p1[0]
    dy_world = p2[1] - p1[1]
    c, s = np.cos(p1[2]), np.sin(p1[2])
    dx_local = c * dx_world + s * dy_world
    dy_local = -s * dx_world + c * dy_world
    dtheta = wrap_angle(p2[2] - p1[2])
    return np.array([dx_local, dy_local, dtheta])


def load_ground_truth(prefix, fnum, suffix):
    """Load ground truth poses from sim_N_poses.npz."""
    fname = prefix + str(fnum) + "_poses" + suffix
    data = np.load(fname)
    poselist = [data[k] for k in data]
    return np.asarray(poselist)


def load_slam(prefix, fnum, suffix):
    """Load estimated poses from sim_N_slam.npz as an Nx3 matrix."""
    fname = prefix + str(fnum) + "_slam" + suffix
    data = np.load(fname)
    keys = list(data.keys())
    slamlist = [data[k] for k in data] # true poses from sim
    return np.asarray(slamlist)


def evaluate(gt_poses, est_poses):
    """
    Evaluate estimated poses against ground truth.

    Metrics:
        1. Sum of squared errors between true and estimated relative poses
           for temporally successive poses.
        2. Relative error in the final position and orientation.

    Parameters:
        gt_poses:  list of N arrays, each shape (3,)
        est_poses: Nx3 array of estimated poses

    Returns:
        Dictionary with evaluation results.
    """
    n = len(gt_poses)
    assert est_poses.shape == (n, 3), (
        f"Expected {n}x3 matrix, got {est_poses.shape}"
    )

    # --- Metric 1: Successive relative pose errors ---
    sse_trans = 0.0
    sse_rot = 0.0
    for i in range(n - 1):
        gt_rel = relative_pose(gt_poses[i], gt_poses[i + 1])
        est_rel = relative_pose(est_poses[i], est_poses[i + 1])
        err = gt_rel - est_rel
        err[2] = wrap_angle(err[2])
        sse_trans += err[0] ** 2 + err[1] ** 2
        sse_rot += err[2] ** 2

    # --- Metric 2: Final pose error ---
    gt_final = gt_poses[-1]
    est_final = est_poses[-1]

    gt_total = np.sqrt(gt_final[0] ** 2 + gt_final[1] ** 2)
    pos_err = np.sqrt((gt_final[0] - est_final[0]) ** 2 +
                      (gt_final[1] - est_final[1]) ** 2)
    orient_err = abs(wrap_angle(gt_final[2] - est_final[2]))

    # Relative errors (avoid division by zero)
    rel_pos_err = pos_err / gt_total if gt_total > 1e-9 else pos_err
    gt_total_rot = abs(wrap_angle(gt_final[2]))
    rel_orient_err = (orient_err / gt_total_rot
                      if gt_total_rot > 1e-6 else orient_err)

    return {
        "n_poses": n,
        "sse_relative_translation": sse_trans,
        "sse_relative_rotation": sse_rot,
        "sse_relative_total": sse_trans + sse_rot,
        "final_position_error_m": pos_err,
        "final_orientation_error_rad": orient_err,
        "final_relative_position_error": rel_pos_err,
        "final_relative_orientation_error": rel_orient_err,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scan-matching pose estimates"
    )
    parser.add_argument("fnum", type=int,
                        help="File number N (loads sim_N_poses.npz and sim_N_slam.npz)")
    parser.add_argument("--prefix", type=str, default="sim_",
                        help="File prefix (default: 'sim_')")
    parser.add_argument("--suffix", type=str, default=".npz",
                        help="File suffix (default: '.npz')")
    args = parser.parse_args()

    gt_poses = load_ground_truth(args.prefix, args.fnum, args.suffix)
    est_poses = load_slam(args.prefix, args.fnum, args.suffix)[:gt_poses.shape[0],:]
    for (g,e) in zip(gt_poses,est_poses):
        print(g,e)

    print(gt_poses.shape)
    print(est_poses.shape)

    results = evaluate(gt_poses, est_poses)

    print(f"Evaluation for {args.prefix}{args.fnum}")
    print(f"  Number of poses: {results['n_poses']}")
    print()
    print("Metric 1: Successive relative pose errors (sum of squared)")
    print(f"  Translation SSE: {results['sse_relative_translation']:.6f} m^2")
    print(f"  Rotation SSE:    {results['sse_relative_rotation']:.6f} rad^2")
    print(f"  Total SSE:       {results['sse_relative_total']:.6f}")
    print()
    print("Metric 2: Final pose error")
    print(f"  Position error:     {results['final_position_error_m']:.4f} m")
    print(f"  Orientation error:  {results['final_orientation_error_rad']:.4f} rad")
    print(f"  Relative pos error: {results['final_relative_position_error']:.4f}")
    print(f"  Relative ori error: {results['final_relative_orientation_error']:.4f}")


if __name__ == "__main__":
    main()
