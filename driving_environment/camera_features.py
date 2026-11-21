"""Extract training observations from two grayscale images. No simulator object access."""
import cv2
import numpy as np

from driving_environment.cameras import K, F, W, H, HEIGHT, BASELINE
from geometric_vision.odometry import GroundOdometry


class CameraFeatures:
    def __init__(self):
        self.motion = GroundOdometry(K)
        self.match = cv2.StereoSGBM_create(minDisparity=0,numDisparities=16,blockSize=3,
            P1=72,P2=288,uniquenessRatio=8,disp12MaxDiff=1)
        self.v,self.u = np.indices((H,W),dtype=np.float32)
        self.z = np.arange(3.7,14.1,.4)
        self.cols = np.arange(-6,6.01,.08)
        self.umap = np.float32(W/2+F*(self.cols[None,:]+BASELINE/2)/self.z[:,None])
        self.vmap = np.broadcast_to(np.float32(H/2+F*HEIGHT/self.z[:,None]),self.umap.shape).copy()

    def lane(self, gray):
        road=cv2.remap(gray,self.umap,self.vmap,cv2.INTER_LINEAR)
        # Find a pair of bright markings with plausible lane width at each distance.
        centers,forward=[],[]
        prior=0.
        for row,z in zip(road,self.z):
            painted=np.flatnonzero(row>175)
            groups=np.split(painted,np.flatnonzero(np.diff(painted)>1)+1)
            marks=[self.cols[(g[0]+g[-1])//2] for g in groups if len(g)]
            candidates=[(l+r)/2 for l,r in zip(marks,marks[1:]) if 2.6<r-l<4.7]
            if not candidates: continue
            center=min(candidates,key=lambda x:abs(x-prior))
            centers.append(center)
            forward.append(z+1.6)
            prior=center
        if len(centers)<5: return None
        # Extrapolate the near tangent for the unseen first metres. Curved lookaheads
        # use the measured centers directly rather than assuming a straight lane.
        count=min(7,len(centers))
        slope,intercept=np.polyfit(forward[:count],centers[:count],1)
        left=[-(intercept+slope*d if d<forward[0] else np.interp(d,forward,centers)) for d in (3,6,10)]
        return float(intercept/np.hypot(1,slope)),float(np.arctan(slope)),left

    def measure(self,left,right,timestamp):
        speed=self.motion.update(left,timestamp)
        lane=self.lane(left)
        disparity=self.match.compute(left,right).astype(np.float32)/16
        depth=np.full_like(disparity,np.nan)
        np.divide(F*BASELINE,disparity,out=depth,where=disparity>0)
        x=(self.u-W/2)*depth/F-BASELINE/2
        y=(self.v-H/2)*depth/F
        mask=((y>-.2)&(y<HEIGHT-.45)&(abs(x)<2.2)&(depth<32)&(depth>1.)).astype('uint8')
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        objects=[]
        for idx in np.flatnonzero(stats[1:,cv2.CC_STAT_AREA]>=8)+1:
            chosen=labels==idx
            objects.append((float(np.median(depth[chosen])),float(np.median(x[chosen]))))
        point=min(objects) if objects else None
        return speed,lane,point
