import numpy as np
from math import cos, sin, pi, tanh, exp, atan2
import argparse
import shutil
import matplotlib.pyplot as plt

def transform_scan(scan, tx, ty, phi):
    """
    Applies a rigid-body transformation to a scan.

    Parameters:
        scan (ndarray): Nx2 array of (x, y) points.
        tx, ty (float): Translation parameters.
        phi (float): Rotation parameter (in radians).

    Returns:
        transformed_scan (ndarray): Transformed scan.
    """
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi),  np.cos(phi)]])
    return (rotation @ scan.T).T + np.array([tx, ty])

def load(args):
    """
    loads the stored scans and their poses (gen in MuJoCo) 
    """
    fname = args.prefix+str(args.fnum)+"_scans"+args.suffix
    data = np.load(fname)
    scanlist = [data[k] for k in data]

    fname = args.prefix+str(args.fnum)+"_poses"+args.suffix
    data = np.load(fname)
    poselist = [data[k] for k in data] # true poses from sim
    ## use the first pose as the initial pose, ignoring passed arguments
    return scanlist, poselist

def main(args) -> None:
    """
    Load file and plot 
    """
    scanlist, poselist = load(args) 
    fig = plt.figure();
    ax = plt.subplot(111)
    for p in poselist:
        ax.scatter(p[0],p[1],color='r')

    for scan,pose in zip(scanlist,poselist):
        transformed_scan=transform_scan(scan,pose[0],pose[1],pose[2]) ## this was better than dxi,dyi,dti
        x, y = zip(*transformed_scan) 
        ax.scatter(x,y,color='y')
    ax.scatter(poselist[0][0],poselist[0][1],color='r',label="ground truth")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Ground truth path and resulting map")
    plt.legend()
    plt.show()
    plt.savefig(args.prefix+str(args.fnum)+"_plot.png")

if __name__ == "__main__":
    """
    Code to handle command line arguments.
    Example:
    ```
    python3 loader.py 25
    ```
    where the last integer argument corresponds to `N` in `sim_N_scans.npz`

    """ 
    parser = argparse.ArgumentParser(description="Process multiple files with sequential numbering")
    parser.add_argument("fnum", type=int, help="Starting integer for file sequence")
    parser.add_argument("--prefix", type=str, default="sim_", help="Prefix for the files (default: '')")
    parser.add_argument("--suffix", type=str, default=".npz", help="Suffix for the files (default: '.npz')")
    args = parser.parse_args()
    main(args)

