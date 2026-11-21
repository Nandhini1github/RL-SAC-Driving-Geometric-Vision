"""Dedicated CARLA server, one deterministic approach to a parked vehicle."""
import argparse
import csv
import json
from math import atan2, cos, hypot, radians, sin, tan
from pathlib import Path
from queue import Queue
from time import perf_counter

import carla
import cv2
import numpy as np
import torch
from stable_baselines3 import SAC

from geometric_vision.rig import Rectifier, obstacle
from geometric_vision.stereo import estimate, make_matcher
from safety.brake import needs_brake

W, H, BASELINE = 960, 540, 0.32
OUT = Path("results/carla")
RAW = Path(".runtime/capture")


def dot(a,b):
    return a.x*b.x + a.y*b.y + a.z*b.z


def choose_route(road):
    for pose in road.get_spawn_points():
        wp = road.get_waypoint(pose.location)
        pts=[wp]
        for _ in range(21):
            following=pts[-1].next(2)
            if len(following)!=1 or following[0].is_junction: break
            pts.append(following[0])
        if len(pts)==22 and all(dot(p.transform.get_forward_vector(), wp.transform.get_forward_vector())>0.998 for p in pts):
            return pts
    raise RuntimeError("No straight 42 m approach found")


def raw(image):
    return np.frombuffer(image.raw_data,np.uint8).reshape(H,W,4)


def run(port, frames):
    cv2.setNumThreads(2)
    torch.set_num_threads(1)
    c=carla.Client("127.0.0.1",port)
    c.set_timeout(30.0)
    world=c.get_world()
    road=world.get_map()
    route=choose_route(road)
    lib=world.get_blueprint_library()
    old=world.get_settings()
    settings=world.get_settings()
    settings.synchronous_mode=True
    settings.fixed_delta_seconds=0.1
    actors=[]
    queues=[]
    rows=[]
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    fx=W/(2*tan(radians(80/2)))
    k=np.array([[fx,0,W/2],[0,fx,H/2],[0,0,1.]])
    rect=Rectifier(k,k,None,None,np.eye(3),np.array([-BASELINE,0.,0.]),(W,H))
    matcher=make_matcher()
    model=SAC.load("results/control/fixed",device="cpu")
    world.apply_settings(settings)
    try:
        def vehicle(idx):
            pose=route[idx].transform
            pose.location += carla.Location(z=0.5)
            car=world.spawn_actor(lib.find("vehicle.tesla.model3"),pose)
            actors.append(car)
            return car
        ego=vehicle(0)
        parked=vehicle(16)
        parked.apply_control(carla.VehicleControl(hand_brake=True))
        for _ in range(12): world.tick()
        for kind, side in (("rgb",-BASELINE/2),("rgb",BASELINE/2),("depth",-BASELINE/2)):
            bp=lib.find("sensor.camera."+kind)
            for attr,val in (("image_size_x",str(W)),("image_size_y",str(H)),("fov","80"),
                             ("lens_k","0"),("lens_kcube","0"),("enable_postprocess_effects","true")):
                if bp.has_attribute(attr): bp.set_attribute(attr,val)
            sensor=world.spawn_actor(bp,carla.Transform(carla.Location(x=1.6,y=side,z=1.5)),attach_to=ego)
            actors.append(sensor)
            q=Queue()
            sensor.listen(q.put)
            queues.append(q)
        collisions=[]
        hit=world.spawn_actor(lib.find("sensor.other.collision"),carla.Transform(),attach_to=ego)
        actors.append(hit)
        hit.listen(lambda event:collisions.append(event.frame))
        prev=np.zeros(2)
        bins=[(3,8),(8,16),(16,30)]
        braking_frames=0
        hold=False
        for step in range(frames+8):
            tick=world.tick()
            imgs=[q.get(timeout=20) for q in queues]
            if len({im.frame for im in imgs})!=1:
                raise RuntimeError("Unmatched camera frame IDs")
            if step<8: continue
            bgr=raw(imgs[0])[:,:,:3].copy()
            start=perf_counter()
            left,right=rect.apply(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.cvtColor(raw(imgs[1]),cv2.COLOR_BGRA2GRAY))
            depth=estimate(left,right,matcher,rect.k[0,0],rect.baseline)
            found=obstacle(depth,rect.k)
            ms=(perf_counter()-start)*1000
            pose=ego.get_transform()
            loc=pose.location
            wp=road.get_waypoint(loc)
            right_axis=pose.get_right_vector()
            heading=radians(wp.transform.rotation.yaw-pose.rotation.yaw)
            err=-dot(loc-wp.transform.location,wp.transform.get_right_vector())
            speed=ego.get_velocity().length()
            lat=[-dot(wp.next(d)[0].transform.location-loc,right_axis) for d in (3,6,10)]
            z=found[1] if found else None
            bearing=-atan2(found[0]-BASELINE/2,found[1]+1.6) if found else 0.0
            obs=np.clip([speed/10,err/3,sin(heading),lat[0]/5,lat[1]/8,lat[2]/12,
                         min((hypot(found[0]-BASELINE/2,found[1]+1.6) if found else 40)/40,1),
                         sin(bearing),float(bool(found)),*prev],-1,1).astype(np.float32)
            action=model.predict(obs,deterministic=True)[0]
            cov=float(np.isfinite(depth).mean())
            brake=needs_brake(speed,z,cov,tick-imgs[0].frame) or hold
            # Terminal stop task with a stationary obstacle. Hold once nearly stopped.
            hold=hold or (brake and speed<1.0 and z is not None and z<5)
            steer=float(-action[0])
            throttle=0.0 if brake else float(max(action[1],0))
            pedal=1.0 if brake else float(max(-action[1],0))
            ego.apply_control(carla.VehicleControl(steer=steer,throttle=throttle,brake=pedal))
            prev=np.array([-steer, throttle-pedal])
            # Evaluation begins after control. Pixel-aligned left GT is remapped to the rectified view.
            gt_raw=raw(imgs[2]).astype(np.float32)
            gt=(gt_raw[:,:,2]+256*gt_raw[:,:,1]+65536*gt_raw[:,:,0])*(1000/16777215)
            gt=cv2.remap(gt,*rect.maps[0],cv2.INTER_NEAREST)
            target=parked.bounding_box.get_world_vertices(parked.get_transform())
            cam=imgs[0].transform
            truth=min(dot(vertex-cam.location,cam.get_forward_vector()) for vertex in target)
            row=dict(frame=imgs[0].frame,step=step-8,speed_mps=speed,steer=steer,requested_long=float(action[1]),
                     brake=pedal,throttle=throttle,override=int(brake),collision=int(bool(collisions)),tracking_m=err,
                     range_est_m=z,parked_box_front_m=truth,coverage=cov,vision_ms=ms)
            for lo,hi in bins:
                mask=(gt>=lo)&(gt<hi)
                valid=mask&np.isfinite(depth)
                dif=depth[valid]-gt[valid]
                key=f"{lo}_{hi}"
                row.update({key+"_eligible":int(mask.sum()),key+"_valid":len(dif),key+"_abs":float(abs(dif).sum()),
                            key+"_sq":float(np.square(dif).sum())})
            i=len(rows)
            rows.append(row)
            cv2.imwrite(str(RAW/f"rgb_{i:03}.jpg"),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
            color=cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip(1-depth/30,0,1))*255),cv2.COLORMAP_TURBO)
            color[~np.isfinite(depth)]=0
            cv2.imwrite(str(RAW/f"depth_{i:03}.jpg"),color)
            if i in (0,40):
                cv2.imwrite(str(OUT/f"left_{i}.png"),bgr)
                cv2.imwrite(str(OUT/f"right_{i}.png"),raw(imgs[1])[:,:,:3])
                np.savez_compressed(OUT/f"reference_{i}.npz",depth_m=gt.astype(np.float16))
            braking_frames=braking_frames+1 if brake and speed<0.15 else 0
            if braking_frames>=12 or collisions: break
        with (OUT/"frames.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT/"run.json").write_text(json.dumps(dict(server=c.get_server_version(),map=road.name,
            width=W,height=H,baseline_m=BASELINE,hfov_deg=80,dt_s=0.1,frames=len(rows),
            collisions=len(collisions),route_start=str(route[0].transform),end_reason="collision" if collisions else
            "stopped under override" if braking_frames>=12 else "frame budget"),indent=2))
        print((OUT/"run.json").read_text())
    finally:
        for actor in reversed(actors):
            if isinstance(actor,carla.Sensor) and actor.is_listening: actor.stop()
            actor.destroy()
        world.apply_settings(old)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=2540)
    parser.add_argument("--frames",type=int,default=130)
    args=parser.parse_args()
    run(args.port,args.frames)
