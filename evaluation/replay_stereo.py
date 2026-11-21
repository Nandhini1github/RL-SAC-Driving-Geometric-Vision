"""Re-run perception on the two published CARLA pairs, using only RGB for localization."""
import json
from pathlib import Path
import cv2
import numpy as np
from geometric_vision.rig import Rectifier, obstacle_xyz
from geometric_vision.stereo import estimate, make_matcher


def main():
    root = Path("results/carla_sensor")
    meta = json.loads((root/"run.json").read_text())
    w,h = meta["width"], meta["height"]
    f = w/(2*np.tan(np.deg2rad(meta["hfov_deg"]/2)))
    k = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])
    b = meta["baseline_m"]
    rig = Rectifier(k,k,None,None,np.eye(3),np.array([-b,0,0]),(w,h))
    out=[]
    for index in (0,40):
        imgs=[cv2.cvtColor(cv2.imread(str(root/f"{side}_{index}.png")),cv2.COLOR_BGR2GRAY) for side in ("left","right")]
        left,right=rig.apply(*imgs)
        depth=estimate(left,right,make_matcher(),rig.k[0,0],rig.baseline)
        point=obstacle_xyz(depth,rig.k)
        if point is None: raise RuntimeError(f"No obstacle at sample {index}")
        x,y,z=point
        ego=[z+1.6,x-b/2,1.5-y]
        gt=np.load(root/f"reference_{index}.npz")["depth_m"].astype(np.float32)
        mask=(gt>=3)&(gt<30)&np.isfinite(depth)
        out.append(dict(step=index,optical_xyz_m=point,ego_forward_right_up_m=ego,
                        valid_pixels_3_to_30_m=int(mask.sum()),mae_m=float(np.abs(depth[mask]-gt[mask]).mean())))
    (root/"sample_points.json").write_text(json.dumps(out,indent=2))
    print(out)


if __name__=="__main__":
    main()
