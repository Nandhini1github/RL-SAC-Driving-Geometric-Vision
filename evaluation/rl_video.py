"""Render a held-out bicycle-model rollout with metric axes and saved SAC weights."""
import argparse
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


def main(seed, compare_td3=False):
    torch.set_num_threads(1)
    if compare_td3:
        from sac_training.td3_camera import OUT as output, SETTINGS, base_command, load_actor
        models = [None, load_actor("clone",37), load_actor("final",37)]
        names = ("Tuned pure pursuit", "Imitation only", "TD3 after imitation")
        baseline = lambda obs: base_command(obs, SETTINGS)
        title = "From driving demonstrations to TD3"
    else:
        output = MEDIA
        models=[None]+[SAC.load(OUT/name,device="cpu") for name in ("fixed","adaptive")]
        names = ("Pure pursuit","SAC with fixed training","SAC with adaptive training")
        baseline = pursuit
        title = "How closely does each car follow the lane?"
    envs=[RouteEnv() for _ in models]
    obs=[env.reset(seed=seed,options={"scene":"obstacle"})[0] for env in envs]
    tracks=[[] for _ in models]
    states=["tracking"]*3
    squared_errors = [0.0] * 3
    reached_finish = [False] * 3
    failed_reference = [False] * 3
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ff,"-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","1600x900","-r","10","-i","-",
         "-an","-c:v","libx264","-crf","17","-pix_fmt","yuv420p","-movflags","+faststart",str(output/"controllers.mp4")]
    previews=[]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for tick in range(162):
            img=np.full((900,1600,3),(33,28,22),np.uint8)
            label(img,title,40,68,1.55)
            label(img,f"Gymnasium obstacle bypass   |   Held-out route {seed}   |   Training seed 37",42,114,.88,MUTED)
            label(img,f"{tick/10:.1f} s",1390,80,1.5,TEAL)
            for j,(model,env,name) in enumerate(zip(models,envs,names)):
                top=150+230*j
                def point(x,y): return (round(420+23*x), round(top+120-23*y))
                cv2.line(img,(40,top),(1560,top),(81,69,56),1)
                label(img,name,45,top+42,1.02)
                error = f"{np.sqrt(squared_errors[j] / env.steps):.2f} m" if env.steps else "--"
                label(img,f"Tracking RMSE   {error}",650,top+42,.87,MUTED)
                label(img,states[j],1100,top+42,.87,TEAL if states[j]=="Finished" else AMBER)
                camera=cv2.resize(env.vision.motion.previous,(224,130),interpolation=cv2.INTER_NEAREST)
                img[top+62:top+192,60:284]=cv2.cvtColor(camera,cv2.COLOR_GRAY2BGR)
                pts=np.int32([point(x,env.path(x)) for x in np.linspace(0,40,160)])
                cv2.polylines(img,[pts],False,(185,180,170),4,cv2.LINE_AA)
                cv2.line(img,point(40,-1),point(40,1),MUTED,2,cv2.LINE_AA)
                cv2.circle(img,point(env.ox,0),round(1.35*23),(64,98,182),2,cv2.LINE_AA)
                tracks[j].append(point(env.x,env.y))
                cv2.polylines(img,[np.int32(tracks[j])],False,TEAL,3,cv2.LINE_AA)
                cv2.circle(img,tracks[j][-1],8,TEAL,-1,cv2.LINE_AA)
                for x in (0,10,20,30,40):
                    tick_label = "40 m finish" if x == 40 else f"{x} m"
                    label(img,tick_label,point(x,0)[0]-10,top+220,.63,MUTED,1)
            label(img,"Camera view (left)    |    Actual path in teal, lane center in gray    |    Lower tracking RMSE is better",60,878,.83,MUTED)
            encoder.stdin.write(img.tobytes())
            if tick%3==0: previews.append(Image.fromarray(cv2.cvtColor(cv2.resize(img,(768,432)),cv2.COLOR_BGR2RGB)))
            if all(state!="tracking" for state in states): break
            for j,(model,env) in enumerate(zip(models,envs)):
                if states[j]!="tracking": continue
                a=baseline(obs[j]) if model is None else model.predict(obs[j],deterministic=True)[0]
                obs[j],_,term,trunc,info=env.step(a)
                # These measurements score the replay after the actor has taken its step.
                lateral_error, contact, reached = env.reference_metrics()
                squared_errors[j] += lateral_error ** 2
                reached_finish[j] |= reached
                failed_reference[j] |= contact or abs(lateral_error) > 2.6
                if term or trunc:
                    if reached_finish[j] and not failed_reference[j]:
                        states[j]="Finished"
                    elif env.invalid >= 3:
                        states[j]="Camera input lost"
                    elif info["collision"]:
                        states[j]="Visual hazard"
                    elif info["departure"]:
                        states[j]="Lane offset limit"
                    elif trunc:
                        states[j]="Time limit"
                    else:
                        states[j]="Finish unconfirmed"
        for _ in range(28): encoder.stdin.write(img.tobytes())
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError("Video export failed")
    previews.append(Image.fromarray(cv2.cvtColor(cv2.resize(img,(768,432)),cv2.COLOR_BGR2RGB)))
    previews[0].save(output/"controllers.gif",save_all=True,append_images=previews[1:],
                     duration=[300]*(len(previews)-1)+[2800],loop=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,help="Held-out obstacle route to replay")
    parser.add_argument("--compare-td3",action="store_true")
    args=parser.parse_args()
    seed=args.seed if args.seed is not None else (31000 if args.compare_td3 else 8002)
    main(seed,args.compare_td3)
