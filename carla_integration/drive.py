"""Dedicated CARLA server, one deterministic approach to a parked vehicle."""
import argparse
import csv
import json
from math import atan2, hypot, radians, sin, tan
from pathlib import Path
from queue import Queue
from time import perf_counter

import carla
import cv2
import numpy as np
import torch
from td3_training.td3_camera import OUT as WEIGHTS, SEEDS
from td3_training.tracking import TD3, OUT as TRACKING
from hashlib import sha256

from geometric_vision.rig import Rectifier, obstacle
from geometric_vision.lane import LaneCamera
from geometric_vision.odometry import GroundOdometry
from geometric_vision.stereo import estimate, make_matcher
from safety.brake import needs_brake

W, H, BASELINE = 960, 540, 0.32
OUT = Path("results/carla_td3")
RAW = Path(".runtime/history-capture")


def dot(a,b):
    return a.x*b.x + a.y*b.y + a.z*b.z


# Fixed Town10HD scene configuration (metres and degrees).
EGO_START = (-24.337173, -57.747787, 0.596735)
PARKED_START = (7.718659, -57.400124, 359.976562)


def raw(image):
    return np.frombuffer(image.raw_data,np.uint8).reshape(H,W,4)


def run(port, frames, output=OUT, capture=RAW):
    cv2.setNumThreads(2)
    torch.set_num_threads(1)
    c=carla.Client("127.0.0.1",port)
    c.set_timeout(30.0)
    world=c.get_world()
    road=world.get_map()
    lib=world.get_blueprint_library()
    old=world.get_settings()
    settings=world.get_settings()
    settings.synchronous_mode=True
    settings.fixed_delta_seconds=0.1
    actors=[]
    queues=[]
    rows=[]
    output.mkdir(parents=True,exist_ok=True)
    capture.mkdir(parents=True,exist_ok=True)
    fx=W/(2*tan(radians(80/2)))
    k=np.array([[fx,0,W/2],[0,fx,H/2],[0,0,1.]])
    rect=Rectifier(k,k,None,None,np.eye(3),np.array([-BASELINE,0.,0.]),(W,H))
    matcher=make_matcher()
    lanes=LaneCamera(rect.k, BASELINE)
    speedometer=GroundOdometry(rect.k)
    model = TD3()
    weights = {path.as_posix(): sha256(path.read_bytes()).hexdigest() for seed in SEEDS
               for path in (WEIGHTS/str(seed)/"final.pt", TRACKING/f"steering-{seed}.pt")}
    world.apply_settings(settings)
    try:
        def vehicle(start):
            x, y, yaw = start
            pose = carla.Transform(carla.Location(x=x, y=y, z=0.5), carla.Rotation(yaw=yaw))
            car=world.spawn_actor(lib.find("vehicle.tesla.model3"),pose)
            actors.append(car)
            return car
        ego=vehicle(EGO_START)
        parked=vehicle(PARKED_START)
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
            imgs=[q.get(timeout=20) for q in queues[:2]]
            if len({im.frame for im in imgs})!=1:
                raise RuntimeError("Unmatched camera frame IDs")
            bgr=raw(imgs[0])[:,:,:3].copy()
            start=perf_counter()
            left,right=rect.apply(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.cvtColor(raw(imgs[1]),cv2.COLOR_BGRA2GRAY))
            speed=speedometer.update(left,imgs[0].timestamp)
            if step<8:
                continue
            depth=estimate(left,right,matcher,rect.k[0,0],rect.baseline)
            found=obstacle(depth,rect.k)
            lane=lanes.estimate(left)
            ms=(perf_counter()-start)*1000
            missing = lane is None or speed is None
            z=found[1] if found else None
            bearing=-atan2(found[0]-BASELINE/2,found[1]+1.6) if found else 0.0
            cov=float(np.isfinite(depth).mean())
            err,heading,lat=(np.nan,np.nan,[np.nan]*3) if lane is None else lane.tracking_features()
            inference_ms = None
            if missing:
                model.reset()
                action=np.array([0.,-1.])
                brake=True
            else:
                obs=np.clip([speed/10,err/3,sin(heading),lat[0]/5,lat[1]/8,lat[2]/12,
                             min((hypot(found[0]-BASELINE/2,found[1]+1.6) if found else 40)/40,1),
                             sin(bearing),float(bool(found)),*prev],-1,1).astype(np.float32)
                inference_start = perf_counter()
                action=model.predict(obs)
                inference_ms = (perf_counter() - inference_start) * 1000
                brake=needs_brake(speed,z,cov,tick-imgs[0].frame) or hold
            # Terminal stop task with a stationary obstacle. Hold once nearly stopped.
            hold=hold or (not missing and brake and speed<1.0 and z is not None and z<5)
            steer=float(-action[0])
            throttle=0.0 if brake else float(max(action[1],0))
            pedal=1.0 if brake else float(max(-action[1],0))
            ego.apply_control(carla.VehicleControl(steer=steer,throttle=throttle,brake=pedal))
            prev=np.array([-steer, throttle-pedal])
            # Evaluation begins after control. Pixel-aligned left GT is remapped to the rectified view.
            reference = queues[2].get(timeout=20)
            while reference.frame < imgs[0].frame:
                reference = queues[2].get(timeout=20)
            if reference.frame != imgs[0].frame:
                raise RuntimeError("Missing aligned evaluation frame")
            pose=ego.get_transform()
            wp=road.get_waypoint(pose.location)
            ref_err=-dot(pose.location-wp.transform.location,wp.transform.get_right_vector())
            gt_raw=raw(reference).astype(np.float32)
            gt=(gt_raw[:,:,2]+256*gt_raw[:,:,1]+65536*gt_raw[:,:,0])*(1000/16777215)
            gt=cv2.remap(gt,*rect.maps[0],cv2.INTER_NEAREST)
            target=parked.bounding_box.get_world_vertices(parked.get_transform())
            cam=imgs[0].transform
            truth=min(dot(vertex-cam.location,cam.get_forward_vector()) for vertex in target)
            row=dict(frame=imgs[0].frame,step=step-8,speed_mps=speed,steer=steer,requested_long=None if missing else float(action[1]),
                     requested_steer=None if missing else float(action[0]),policy_ms=inference_ms,
                     brake=pedal,throttle=throttle,override=int(brake),collision=int(bool(collisions)),tracking_m=ref_err,
                     lane_offset_m=err,lane_heading_rad=heading,lane_markings=lane.markings if lane else 0,
                     speed_ref_mps=ego.get_velocity().length(),sensor_missing=int(missing),
                     lane_left_3m=lat[0],lane_left_6m=lat[1],lane_left_10m=lat[2],
                     range_est_m=z, policy_range_norm=None if missing else float(obs[6]),
                     policy_bearing_sine=None if missing else float(obs[7]),parked_box_front_m=truth,coverage=cov,vision_ms=ms)
            for lo,hi in bins:
                mask=(gt>=lo)&(gt<hi)
                valid=mask&np.isfinite(depth)
                dif=depth[valid]-gt[valid]
                key=f"{lo}_{hi}"
                row.update({key+"_eligible":int(mask.sum()),key+"_valid":len(dif),key+"_abs":float(abs(dif).sum()),
                            key+"_sq":float(np.square(dif).sum())})
            i=len(rows)
            rows.append(row)
            cv2.imwrite(str(capture/f"rgb_{i:03}.jpg"),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
            color=cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip(1-depth/30,0,1))*255),cv2.COLORMAP_TURBO)
            color[~np.isfinite(depth)]=0
            cv2.imwrite(str(capture/f"depth_{i:03}.jpg"),color)
            if i in (0,40):
                cv2.imwrite(str(output/f"left_{i}.png"),bgr)
                cv2.imwrite(str(output/f"right_{i}.png"),raw(imgs[1])[:,:,:3])
                np.savez_compressed(output/f"reference_{i}.npz",depth_m=gt.astype(np.float16))
            braking_frames=braking_frames+1 if hold and speed is not None and speed<0.15 else 0
            if braking_frames>=12: break
        with (output/"frames.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output/"run.json").write_text(json.dumps(dict(server=c.get_server_version(),map=road.name,
            controller="TD3",lane_history_lookahead_m=3.5,checkpoint_sha256=weights,observation_source="stereo RGB and ground-plane visual odometry",width=W,height=H,baseline_m=BASELINE,hfov_deg=80,dt_s=0.1,frames=len(rows),
            collisions=len(collisions),ego_spawn_xy_yaw=list(EGO_START),parked_spawn_xy_yaw=list(PARKED_START),
            end_reason="stopped under override" if braking_frames>=12 else "frame budget"),indent=2))
        print((output/"run.json").read_text())
    finally:
        for actor in reversed(actors):
            if isinstance(actor,carla.Sensor) and actor.is_listening: actor.stop()
            actor.destroy()
        world.apply_settings(old)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=2540)
    parser.add_argument("--frames",type=int,default=130)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--capture", type=Path, default=RAW)
    args=parser.parse_args()
    run(args.port,args.frames,args.output,args.capture)
