## Assignment 1 

You can visualize the map as a point cloud using ground-truth poses by running: 

```
python3 loader.py 25
```
where the file being loaded is `sim_25_scans.npz`. 

Please modify this code to visualize your solution. 

I will use the `evaluator.py` file to evaluate your code. It expects file `sim_25_slam.npz`, for example, to exist. Run:
```
python3 evaluator.py 25
```
to see the output. Use this code to ensure that your pose estimation code outputs a file that can be suitably evaluated. 

You can store a list `slamlist` of estimated poses using :

```
np.savez(f"sim_"+str(args.fnum)+"_slam.npz", *slamlist)
```

