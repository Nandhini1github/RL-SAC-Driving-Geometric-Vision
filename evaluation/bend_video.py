"""Render both independent CARLA bend trials at their 10 Hz simulation rate."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image

from evaluation.carla_media import AMBER, MUTED, TEAL, label


def summarize(df, meta):
    yaw = np.rad2deg(np.unwrap(np.deg2rad(df.yaw_deg)))
    return dict(frames=len(df), distance_m=float(np.linalg.norm(np.diff(df[["x_m","y_m"]], axis=0), axis=1).sum()),
                heading_change_deg=float(yaw[-1]-yaw[0]), collisions=meta["collisions"],
                lateral_rmse_m=float(np.sqrt(np.mean(df.tracking_m**2))),
                peak_lateral_error_m=float(df.tracking_m.abs().max()), overrides=int(df.override.sum()),
                missing_input_frames=int(df.sensor_missing.sum()), outside_reference_frames=int((~df.within_reference).sum()))


def path_pixels(df):
    actual = df[["x_m","y_m"]].to_numpy()
    reference = df[["reference_x_m","reference_y_m"]].to_numpy()
    paths = np.stack((actual,reference))
    origin = paths.min(axis=(0,1))
    scale = min(560/max(1, np.ptp(paths[:,:,0])), 225/max(1, np.ptp(paths[:,:,1])))
    return np.rint((paths-origin)*scale + [110,805]).astype(np.int32)


def render(frame, row, capture, turn, i, df, paths, yaw, rmse):
    label(frame, f"TD3 on a {turn} bend", 36, 76, 1.65)
    label(frame, f"{i/10:.1f} s", 1650, 76, 1.5, TEAL)
    images = [cv2.imread(str(capture/turn/f"{name}_{i:03}.jpg")) for name in ("chase","rgb")]
    if any(im is None for im in images):
        raise FileNotFoundError(f"Missing synchronized views for {turn} frame {i}")
    chase, rgb = images
    if row.lane_markings:
        forward = np.linspace(5.2,17.8,50)
        slope = np.tan(row.lane_heading_rad)
        right = row.lane_offset_m*np.hypot(1,slope) + slope*forward + .5*row.lane_curvature*forward**2
        f = 480/np.tan(np.deg2rad(40))
        pixels = np.column_stack((480+f*(right+.16)/(forward-1.6),270+f*1.5/(forward-1.6)))
        cv2.polylines(rgb, [pixels.astype(np.int32)], False, TEAL, 4, cv2.LINE_AA)
    frame[104:644,:960] = chase
    frame[104:644,960:] = rgb
    label(frame,"Chase camera",32,695,1.0)
    speed = f"Visual speed  {row.speed_mps:.1f} m/s" if np.isfinite(row.speed_mps) else "Visual speed unavailable"
    label(frame,speed,490,695,.85,MUTED)
    label(frame,"Front camera and fitted lane",988,695,1.0)
    status = "BRAKE OVERRIDE" if row.override else "TD3 CONTROL"
    label(frame,status,1510,695,.85,AMBER if row.override else TEAL)
    label(frame,"Path through the bend",50,767,.9)
    actual, reference = paths
    cv2.polylines(frame,[reference], False, MUTED, 8, cv2.LINE_AA)
    cv2.polylines(frame,[actual[:i+1]], False, TEAL, 3, cv2.LINE_AA)
    cv2.circle(frame,actual[i],6,TEAL,-1,cv2.LINE_AA)
    label(frame,"Lane center",690,850,.76,MUTED)
    label(frame,"Driven path",690,897,.76,TEAL)
    label(frame,"Evaluation only",660,957,.65,MUTED,1)
    label(frame,f"Applied steering   {row.steer:+.2f}",990,767,.9)
    # CARLA steering is negative left and positive right, normalized to [-1,1].
    cv2.line(frame,(1030,899),(1830,899),MUTED,1)
    axis_limit=max(.15, float(df.steer.abs().max()))
    points=np.column_stack((np.linspace(1030,1830,len(df)),899-75*df.steer/axis_limit)).astype(np.int32)
    cv2.polylines(frame,[points[:i+1]],False,TEAL,3,cv2.LINE_AA)
    cv2.circle(frame,points[i],5,TEAL,-1,cv2.LINE_AA)
    label(frame,"R",990,833,.65,MUTED)
    label(frame,"L",990,979,.65,MUTED)
    label(frame,f"Heading change  {abs(yaw[i]-yaw[0]):.0f} deg",987,1040,.85)
    label(frame,f"Tracking RMSE  {100*rmse[i]:.1f} cm",1470,1040,.85)


def main(root, capture):
    media=root/"media"
    media.mkdir(parents=True,exist_ok=True)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
             "-s","1920x1080","-r","10","-i","-","-an","-c:v","libx264","-preset","fast","-crf","19",
             "-pix_fmt","yuv420p","-movflags","+faststart",str(media/"td3-bends.mp4")]
    results, preview = {}, []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for turn in ("left","right"):
            directory=root/turn
            df=pd.read_csv(directory/"frames.csv")
            meta=json.loads((directory/"run.json").read_text())
            assert meta["scene"]==turn and len(df)==meta["frames"]
            results[turn]=summarize(df,meta)
            paths=path_pixels(df)
            yaw=np.rad2deg(np.unwrap(np.deg2rad(df.yaw_deg)))
            rmse=np.sqrt(df.tracking_m.pow(2).expanding().mean())
            for i,row in df.iterrows():
                frame=np.full((1080,1920,3),(33,28,22),np.uint8)
                render(frame,row,capture,turn,i,df,paths,yaw,rmse)
                encoder.stdin.write(frame.tobytes())
                if i%5==0:
                    small=cv2.resize(frame,(768,432),interpolation=cv2.INTER_AREA)
                    preview.append(Image.fromarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB)))
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Bend video encoding failed")
    preview[0].save(media/"preview.gif",save_all=True,append_images=preview[1:],duration=500,loop=0)
    (root/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=Path("results/carla_bends"))
    parser.add_argument("--capture-dir",type=Path,default=Path(".runtime/bends"))
    args=parser.parse_args()
    main(args.run_dir,args.capture_dir)
