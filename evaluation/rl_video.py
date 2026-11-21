"""Render a held-out bicycle-model rollout with metric axes and saved SAC weights."""
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image
from stable_baselines3 import SAC
from driving_environment.route import RouteEnv, pursuit
from evaluation.figures import MEDIA, OUT
from evaluation.carla_media import label, INK, MUTED, TEAL, AMBER


def main():
    torch.set_num_threads(1)
    models=[None]+[SAC.load(OUT/name,device="cpu") for name in ("fixed","adaptive")]
    envs=[RouteEnv() for _ in models]
    obs=[env.reset(seed=8000,options={"scene":"obstacle"})[0] for env in envs]
    tracks=[[] for _ in models]
    states=["tracking"]*3
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ff,"-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","1600x900","-r","10","-i","-",
         "-an","-c:v","libx264","-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",str(MEDIA/"controllers.mp4")]
    previews=[]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for tick in range(162):
            img=np.full((900,1600,3),(33,28,22),np.uint8)
            label(img,"Tracking the same obstacle bypass",40,68,1.65)
            label(img,"Camera-trained Gymnasium policies   |   Held-out route 8000   |   Training seed 37",42,114,.88,MUTED)
            label(img,f"{tick/10:.1f} s",1390,80,1.5,TEAL)
            for j,(model,env,name) in enumerate(zip(models,envs,("Pure pursuit","Fixed SAC","Adaptive SAC"))):
                top=150+230*j
                def point(x,y): return (round(425+26*x), round(top+120-26*y))
                cv2.line(img,(40,top),(1560,top),(81,69,56),1)
                label(img,name,45,top+42,1.02)
                label(img,f"{env.measured_speed:.1f} m/s    |    {states[j]}",1090,top+42,.87,AMBER)
                camera=cv2.resize(env.vision.motion.previous,(224,130),interpolation=cv2.INTER_NEAREST)
                img[top+62:top+192,60:284]=cv2.cvtColor(camera,cv2.COLOR_GRAY2BGR)
                pts=np.int32([point(x,env.path(x)) for x in np.linspace(0,40,160)])
                cv2.polylines(img,[pts],False,(185,180,170),4,cv2.LINE_AA)
                cv2.circle(img,point(env.ox,0),round(1.35*26),(64,98,182),2,cv2.LINE_AA)
                tracks[j].append(point(env.x,env.y))
                cv2.polylines(img,[np.int32(tracks[j])],False,TEAL,3,cv2.LINE_AA)
                cv2.circle(img,tracks[j][-1],8,TEAL,-1,cv2.LINE_AA)
                for x in (0,10,20,30,40):
                    label(img,f"{x} m",point(x,0)[0]-10,top+220,.63,MUTED,1)
            label(img,"Left: training camera.   Right: evaluation trajectory (teal), painted route (gray), obstacle radius (orange).",60,878,.83,MUTED)
            encoder.stdin.write(img.tobytes())
            if tick%3==0: previews.append(Image.fromarray(cv2.cvtColor(cv2.resize(img,(768,432)),cv2.COLOR_BGR2RGB)))
            if all(state!="tracking" for state in states): break
            for j,(model,env) in enumerate(zip(models,envs)):
                if states[j]!="tracking": continue
                a=pursuit(obs[j]) if model is None else model.predict(obs[j],deterministic=True)[0]
                obs[j],_,term,trunc,info=env.step(a)
                if term or trunc:
                    states[j]="visual finish" if info["success"] else "visual hazard" if info["collision"] else "lane loss" if info["departure"] else "timeout"
        for _ in range(12): encoder.stdin.write(img.tobytes())
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError("Video export failed")
    previews[0].save(MEDIA/"controllers.gif",save_all=True,append_images=previews[1:],duration=300,loop=0)


if __name__=="__main__":
    main()
