"""Replay lossless sensor pairs and render the actual stereo estimates from both bend passes."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

from evaluation.carla_media import AMBER, INK, MUTED, TEAL, label
from geometric_vision.rig import obstacle
from geometric_vision.stereo import estimate_with_disparity, make_matcher

PANEL = (848, 477)
DARK = (33, 28, 22)


def sample_match(left, right, disparity):
    """Choose a textured, valid sample in front of the car to illustrate horizontal disparity."""
    h, w = left.shape
    valid = np.uint8((disparity >= 4) & (disparity <= 55))
    corners = cv2.goodFeaturesToTrack(left[h//2:7*h//8, w//3:2*w//3], 80, .02, 7,
                                      mask=valid[h//2:7*h//8, w//3:2*w//3])
    if corners is None:
        return None
    points = corners[:, 0, :] + [w//3, h//2]
    for u, v in points[np.argsort(np.linalg.norm(points - [w*.55, h*.72], axis=1))].astype(int):
        d = float(disparity[v,u])
        ur = round(u-d)
        a = left[v-3:v+4, u-3:u+4]
        b = right[v-3:v+4, ur-3:ur+4]
        if cv2.absdiff(a,b).mean() <= 15:
            return u, v, d
    return None


def panel(canvas, source, x, y, title, scale=None):
    image = cv2.resize(source, PANEL, interpolation=cv2.INTER_AREA)
    cv2.rectangle(image,(0,0),(PANEL[0]-1,40),DARK,-1)
    label(image,title,12,29,.78,INK)
    if scale is not None:
        gradient, near, far = scale
        label(image,near,492,27,.53,MUTED,1)
        image[12:24,543:703] = cv2.resize(gradient,(160,12))
        label(image,far,714,27,.53,MUTED,1)
    canvas[y:y+PANEL[1],x:x+PANEL[0]] = image


def telemetry_chart(canvas, values, idx, x, limits, dt):
    """Draw only observed samples, leaving gaps when a camera estimate is missing."""
    lower, upper = limits
    width, height, y = 790, 145, 850
    def pixel(j, value):
        return round(x + width*j/(len(values)-1)), round(y + height*(upper-value)/(upper-lower))
    for v in (lower, (lower+upper)/2, upper):
        _, yy = pixel(0,v)
        cv2.line(canvas,(x,yy),(x+width,yy),(71,61,50),1,cv2.LINE_AA)
        label(canvas,f"{v:g}",x-58,yy+5,.57,MUTED,1)
    for j in range(1,idx+1):
        if np.isfinite(values[j-1:j+1]).all():
            cv2.line(canvas,pixel(j-1,values[j-1]),pixel(j,values[j]),TEAL,3,cv2.LINE_AA)
    if np.isfinite(values[idx]):
        cv2.circle(canvas,pixel(idx,values[idx]),5,TEAL,-1,cv2.LINE_AA)
    label(canvas,"0 s",x-12,y+height+40,.67,MUTED)
    label(canvas,f"{(len(values)-1)*dt:.1f} s",x+width-58,y+height+40,.67,MUTED)


def dashboard_frame(rgb, depth, turn, i, row, speed, steering, dt):
    frame = np.full((1080,1920,3),DARK,np.uint8)
    label(frame,f"TD3 on a {turn} bend in CARLA",32,87,1.65)
    label(frame,f"{i*dt:.1f} s",1650,87,1.65,TEAL)
    frame[139:679,:960] = rgb
    frame[139:679,960:] = depth
    label(frame,"Left RGB and fitted lane",32,721,1.0)
    label(frame,"Stereo depth",990,721,1.0)
    label(frame,"Warm = near     Cool = far (>30 m)     Black = invalid",1215,720,.71,MUTED)
    speed_text = f"Visual speed   {speed[i]:.1f} m/s" if np.isfinite(speed[i]) else "Visual speed unavailable"
    label(frame,speed_text,50,805,1.08)
    override = int(row["override"])
    label(frame,"BRAKE OVERRIDE" if override else "TD3 CONTROL",575,804,.82,AMBER if override else TEAL)
    label(frame,f"Applied steering   {steering[i]:+.2f}",990,805,1.08)
    label(frame,"Negative = left    Positive = right",1510,803,.68,MUTED)
    speed_limit = max(10., float(np.ceil(np.nanmax(speed))))
    steer_limit = max(.4, float(np.ceil(np.max(np.abs(steering))*10)/10))
    telemetry_chart(frame,speed,i,90,(0,speed_limit),dt)
    telemetry_chart(frame,steering,i,1054,(-steer_limit,steer_limit),dt)
    return frame


def render(root, capture, dashboard=False):
    media = root/"media"
    media.mkdir(parents=True, exist_ok=True)
    ramp = np.arange(256, dtype=np.uint8).reshape(1,256)
    gray_ramp = cv2.cvtColor(ramp,cv2.COLOR_GRAY2BGR)
    depth_ramp = cv2.applyColorMap(255-ramp,cv2.COLORMAP_TURBO)
    movie_name = "td3-bend-dashboard.mp4" if dashboard else "stereo-geometry.mp4"
    ffmpeg = [imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
              "-s","1920x1080","-r","10","-i","-","-an","-c:v","libx264","-preset","fast","-crf","23",
              "-pix_fmt","yuv420p","-movflags","+faststart",str(media/movie_name)]
    previews = []
    cv2.setNumThreads(2)
    with subprocess.Popen(ffmpeg, stdin=subprocess.PIPE) as encoder:
        for turn in ("left","right"):
            meta=json.loads((root/turn/"run.json").read_text())
            with (root/turn/"frames.csv").open() as stream:
                rows=list(csv.DictReader(stream))
            if not meta.get("saved_rectified_pairs") or meta["scene"]!=turn or len(rows)!=meta["frames"]:
                raise ValueError(f"Incomplete stereo capture for {turn}")
            # Preserve the live rectifier precision for depth division at corridor boundaries.
            fx, baseline = np.float64(meta["rectified_fx_px"]), np.float64(meta["baseline_m"])
            h, w = meta["height"], meta["width"]
            k = np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]])
            speed = np.array([float(r["speed_mps"] or "nan") for r in rows])
            steering = np.array([float(r["steer"]) for r in rows])
            matcher = make_matcher()
            for i, row in enumerate(rows):
                pair=cv2.imread(str(capture/turn/f"pair_{row['frame']}.png"),cv2.IMREAD_GRAYSCALE)
                if pair is None or pair.shape!=(h,2*w):
                    raise ValueError(f"Missing or invalid pair for {turn} frame {row['frame']}")
                left,right = np.hsplit(pair,2)
                disp,z=estimate_with_disparity(left,right,matcher,fx,baseline)
                valid=np.isfinite(z)
                found=obstacle(z,k)
                logged=float(row["range_est_m"]) if row["range_est_m"] else None
                if (not np.isclose(valid.mean(),float(row["coverage"]),atol=1e-7)
                        or (found is None)!=(logged is None)
                        or (found is not None and not np.isclose(found[1],logged,atol=1e-4))):
                    raise ValueError(f"Replay differs from live perception at {turn} frame {row['frame']}")
                views=[cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) for gray in (left,right)]
                if dashboard:
                    rgb_path = capture/turn/f"rgb_{int(row['step']):03}.jpg"
                    rgb = cv2.imread(str(rgb_path))
                    if rgb is None or rgb.shape != (h,w,3):
                        raise ValueError(f"Missing synchronized RGB image at {rgb_path}")
                    views[0] = rgb
                depth_colors=cv2.applyColorMap(np.uint8(np.nan_to_num(1-np.clip(z/30,0,1))*255),cv2.COLORMAP_TURBO)
                depth_colors[~valid]=0
                if int(row["lane_markings"]):
                    forward=np.linspace(5.2,17.8,40)
                    slope=np.tan(float(row["lane_heading_rad"]))
                    lateral=(float(row["lane_offset_m"])*np.hypot(1,slope)+slope*forward
                             +.5*float(row["lane_curvature"])*forward**2)
                    pts=np.int32(np.column_stack((w/2+fx*(lateral+baseline/2)/(forward-1.6),
                                                  h/2+fx*1.5/(forward-1.6))))
                    cv2.polylines(views[0],[pts],False,TEAL,3,cv2.LINE_AA)
                if dashboard:
                    frame = dashboard_frame(views[0],depth_colors,turn,i,row,speed,steering,meta["dt_s"])
                else:
                    disp_gray=np.uint8(255*np.clip(disp/64,0,1))
                    match=sample_match(left,right,disp)
                    sample="No illustrative match"
                    if match is not None:
                        u,v,d=match
                        for view, col in zip(views,(u,round(u-d))):
                            cv2.drawMarker(view,(int(col),int(v)),AMBER,cv2.MARKER_CROSS,26,3,cv2.LINE_AA)
                        sample=f"Yellow match   d = {d:.2f} px     Z = {fx*baseline/d:.2f} m"
                    frame=np.full((1080,1920,3),DARK,np.uint8)
                    label(frame,f"Stereo geometry on a {turn} bend",60,52,1.35)
                    label(frame,f"Frame {row['frame']}     {i*meta['dt_s']:.1f} s",1350,49,.94,TEAL)
                    panel(frame,views[0],72,72,"Rectified left camera  /  fitted lane")
                    panel(frame,views[1],1000,72,"Rectified right camera")
                    panel(frame,cv2.cvtColor(disp_gray,cv2.COLOR_GRAY2BGR),72,552,"SGBM disparity",(gray_ramp,"0","64 px"))
                    panel(frame,depth_colors,1000,552,"Stereo depth",(depth_ramp,"0","30+ m"))
                    label(frame,sample,74,1063,.78,AMBER)
                    label(frame,f"Z = fB / d     f = {fx:.0f} px     B = {baseline:.2f} m",897,1063,.72)
                    label(frame,f"Valid depth   {valid.mean():.0%}",1590,1063,.71,MUTED)
                encoder.stdin.write(frame.tobytes())
                if i%5 == 0:
                    small=cv2.resize(frame,(704,396),interpolation=cv2.INTER_AREA)
                    previews.append(Image.fromarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB)))
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Stereo film encoding failed")
    preview_name = "dashboard-preview.gif" if dashboard else "preview.gif"
    previews[0].save(media/preview_name,save_all=True,append_images=previews[1:],duration=500,loop=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=Path("results/bend_stereo"))
    parser.add_argument("--capture-dir",type=Path,default=Path(".runtime/bend-stereo"))
    parser.add_argument("--dashboard",action="store_true",help="Show the RGB and stereo depth dashboard with control telemetry")
    args=parser.parse_args()
    render(args.run_dir,args.capture_dir,args.dashboard)
