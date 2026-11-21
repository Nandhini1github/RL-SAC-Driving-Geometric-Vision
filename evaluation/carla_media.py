"""Compose the CARLA evidence film in 1080p; camera panels retain native pixels."""
import json
import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

MEDIA=Path("results/media")
CAPTURE=Path(".runtime/capture")
CARLA=Path("results/carla")
INK=(244,242,235)
MUTED=(189,178,162)
TEAL=(194,208,92)
AMBER=(93,190,251)


def label(img,s,x,y,size=0.9,color=INK,thickness=2):
    cv2.putText(img,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def trace(img,x,y,w,h,series,maximum,colors,idx):
    for frac in (0,0.5,1):
        yy=round(y+h*(1-frac))
        cv2.line(img,(x,yy),(x+w,yy),(71,61,50),1,cv2.LINE_AA)
        label(img,f"{maximum*frac:g}",x-45,yy+5,.57,MUTED,1)
    for seq,color in zip(series,colors):
        points=[(round(x+w*j/(len(seq)-1)),round(y+h-h*min(max(v,0),maximum)/maximum))
                for j,v in enumerate(seq[:idx+1]) if np.isfinite(v)]
        if points:
            cv2.polylines(img,[np.int32(points)],False,color,3,cv2.LINE_AA)
            cv2.circle(img,points[-1],5,color,-1,cv2.LINE_AA)


def main():
    MEDIA.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(CARLA/"frames.csv")
    speed=df.speed_mps.to_numpy()
    ranges=df.range_est_m.to_numpy()
    truth=df.parked_box_front_m.to_numpy()
    bins=[]
    for key in ("3_8","8_16","16_30"):
        valid=df[key+"_valid"].sum()
        bins.append(dict(range_m=key.replace("_"," to "),coverage=valid/df[key+"_eligible"].sum(),
                         mae_m=df[key+"_abs"].sum()/valid,rmse_m=np.sqrt(df[key+"_sq"].sum()/valid)))
    abs_err=np.abs(truth-ranges)
    summary=dict(bins=bins,median_ms=df.vision_ms.median(),p95_ms=df.vision_ms.quantile(.95),
                 range_mae_m=float(np.nanmean(abs_err)),range_rmse_m=float(np.sqrt(np.nanmean(abs_err**2))),
                 detected_frames=int(np.isfinite(ranges).sum()),mean_coverage=df.coverage.mean(),
                 tracking_rmse_m=float(np.sqrt(np.mean(df.tracking_m**2))),shield_frames=int(df.override.sum()),
                 final_box_range_m=truth[-1],peak_speed_mps=float(max(speed)))
    if "speed_ref_mps" in df:
        summary.update(speed_rmse_mps=float(np.sqrt(np.mean((df.speed_mps-df.speed_ref_mps)**2))),
                       lane_offset_rmse_m=float(np.sqrt(np.mean((df.lane_offset_m-df.tracking_m)**2))),
                       missing_input_frames=int(df.sensor_missing.sum()))
    (CARLA/"metrics.json").write_text(json.dumps(summary,indent=2))
    t=np.arange(len(df))/10
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout="constrained")
    axes[0].plot(t,truth,label="Parked box front",color="#355875")
    axes[0].plot(t,ranges,".",label="Stereo surface",color="#c46e24")
    axes[0].set(xlabel="Simulation time (s)",ylabel="Forward range (m)")
    axes[1].plot(t,speed,color="#257b79",label="Vehicle speed")
    axes[1].fill_between(t,0,df.override*max(speed),color="#d8a74d",alpha=.35,step="post",label="Brake override")
    axes[1].set(xlabel="Simulation time (s)",ylabel="Speed (m/s)")
    xx=np.arange(3)
    for i,key in enumerate(("mae_m","rmse_m")):
        axes[2].bar(xx+.32*i,[b[key] for b in bins],.32,label=key.split("_")[0].upper())
    axes[2].set(xticks=xx+.16,xticklabels=[b["range_m"] for b in bins],xlabel="Reference depth band (m)",ylabel="Valid-pixel error (m)")
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(MEDIA/"carla.png",dpi=170)
    plt.close(fig)
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    output=MEDIA/"carla-stereo-stop.mp4"
    cmd=[ff,"-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","1920x1080","-r","10",
         "-i","-","-an","-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
    previews=[]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as process:
        for i,row in df.iterrows():
            img=np.full((1080,1920,3),(33,28,22),np.uint8)
            label(img,"Stereo guided braking in CARLA",32,87,1.65)
            label(img,f"{t[i]:.1f} s",1680,87,1.65,TEAL)
            img[139:679,:960]=cv2.imread(str(CAPTURE/f"rgb_{i:03}.jpg"))
            img[139:679,960:]=cv2.imread(str(CAPTURE/f"depth_{i:03}.jpg"))
            if "lane_heading_rad" in row and row.lane_markings:
                forward=np.linspace(5.2,18,32)
                slope=np.tan(row.lane_heading_rad)
                right=row.lane_offset_m*np.hypot(1,slope)+slope*forward
                f=480/np.tan(np.deg2rad(40))
                u=480+f*(right+.16)/(forward-1.6)
                v=139+270+f*1.5/(forward-1.6)
                cv2.polylines(img,[np.column_stack((u,v)).astype(np.int32)],False,TEAL,3,cv2.LINE_AA)
            label(img,"Left RGB and fitted lane" if "lane_heading_rad" in row else "Left RGB camera",32,721,1.0)
            label(img,"Stereo depth",990,721,1.0)
            label(img,"Warm = near     Cool = far (>30 m)     Black = invalid",1215,720,.71,MUTED)
            label(img,f"Speed   {speed[i]:.1f} m/s",50,805,1.12)
            label(img,f"Range   {ranges[i]:.1f} m",1000,805,1.12)
            label(img,"BRAKE OVERRIDE" if row.override else "SAC CONTROL",570,804,.83,AMBER if row.override else TEAL)
            label(img,"Stereo",1530,805,.82,TEAL)
            label(img,"Box reference",1660,805,.82,AMBER)
            trace(img,88,845,792,130,[speed],10,[TEAL],i)
            trace(img,1044,845,792,130,[ranges,truth],30,[TEAL,AMBER],i)
            label(img,"0 s",75,1012,.67,MUTED)
            label(img,f"{t[-1]:.1f} s",836,1012,.67,MUTED)
            label(img,"0 s",1028,1012,.67,MUTED)
            label(img,f"{t[-1]:.1f} s",1792,1012,.67,MUTED)
            for _ in range(2+(8 if i in (0,len(df)-1) else 0)):
                process.stdin.write(img.tobytes())
            if i%3==0:
                previews.append(Image.fromarray(cv2.cvtColor(cv2.resize(img,(768,432)),cv2.COLOR_BGR2RGB)))
            if i==min(29,len(df)-1):
                cv2.imwrite(str(MEDIA/"carla-poster.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,95])
        process.stdin.close()
        if process.wait(): raise RuntimeError("H.264 export failed")
    previews[0].save(MEDIA/"carla-preview.gif",save_all=True,append_images=previews[1:],duration=600,loop=0)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--sensor-run",action="store_true")
    if parser.parse_args().sensor_run:
        CARLA=Path("results/carla_sensor")
        MEDIA=CARLA/"media"
        CAPTURE=Path(".runtime/sensor_capture")
    main()
