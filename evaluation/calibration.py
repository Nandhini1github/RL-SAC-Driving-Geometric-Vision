"""Estimate stereo extrinsics from noisy synthetic board corners and check epipolar alignment."""
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    rng=np.random.default_rng(814)
    size=(960,540)
    focal=size[0]/(2*np.tan(np.deg2rad(40)))
    k=np.array([[focal,0,480],[0,focal,270],[0,0,1.]])
    rot=cv2.Rodrigues(np.array([0.004,0.012,-0.003]))[0]
    trans=np.array([-0.32,0.003,0.001])
    board=np.zeros((9*6,3),np.float32)
    board[:,:2]=np.mgrid[:9,:6].T.reshape(-1,2)*0.075
    left,right=[],[]
    for _ in range(48):
        tilt=rng.uniform(-0.45,0.45,3)
        loc=rng.uniform([-.4,-.3,1.6],[.2,.1,3.4])
        r_board=cv2.Rodrigues(tilt)[0]
        for dst,r,t in [(left,r_board,loc),(right,rot@r_board,rot@loc+trans)]:
            points=cv2.projectPoints(board,cv2.Rodrigues(r)[0],t,k,None)[0]
            dst.append((points+rng.normal(0,.1,points.shape)).astype(np.float32))
    n=36
    fit=cv2.stereoCalibrate([board]*n,left[:n],right[:n],k.copy(),None,k.copy(),None,
                            size,flags=cv2.CALIB_FIX_INTRINSIC)
    rms,k1,d1,k2,d2,r,t,*_=fit
    r1,r2,p1,p2,*_=cv2.stereoRectify(k1,d1,k2,d2,size,r,t)
    l=cv2.undistortPoints(np.concatenate(left[n:]),k1,d1,R=r1,P=p1)
    rr=cv2.undistortPoints(np.concatenate(right[n:]),k2,d2,R=r2,P=p2)
    err=np.abs(l[:,:,1]-rr[:,:,1])
    report=dict(dataset="projected synthetic chessboard corners with known intrinsics",seed=814,
                train_boards=n,test_boards=48-n,corner_noise_sd_px=.1,
                fitted_reprojection_rms_px=rms,baseline_true_m=np.linalg.norm(trans),
                baseline_fit_m=np.linalg.norm(t),rotation_error_deg=float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(r@rot.T)[0]))),
                heldout_median_vertical_px=float(np.median(err)),heldout_p95_vertical_px=float(np.quantile(err,.95)))
    Path("results/calibration.json").write_text(json.dumps(report,indent=2))
    print(report)


if __name__=="__main__":
    main()
