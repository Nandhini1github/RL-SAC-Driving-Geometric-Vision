"""Metric visual odometry from tracked pavement features and calibrated camera height."""
import cv2
import numpy as np


class GroundOdometry:
    def __init__(self, k, height_m=1.5):
        self.k = k
        self.height_m = height_m
        self.previous = None
        self.time = None
        self.previous_ground = None
        self.scale = .065
        forward, right = np.mgrid[0:125,0:100].astype(np.float32)*self.scale
        z = 3.7+forward
        self.map_u = np.float32(k[0,2]+k[0,0]*(right-3.25)/z)
        self.map_v = np.float32(k[1,2]+k[1,1]*height_m/z)

    def ground_points(self, uv):
        z = self.k[1, 1] * self.height_m / (uv[:, 1] - self.k[1, 2])
        right = (uv[:, 0] - self.k[0, 2]) * z / self.k[0, 0]
        return np.column_stack((z, right)).astype(np.float32)

    def update(self, gray, timestamp):
        prev, start = self.previous, self.time
        self.previous, self.time = gray, timestamp
        ground=cv2.remap(gray,self.map_u,self.map_v,cv2.INTER_LINEAR)
        previous_ground=self.previous_ground
        self.previous_ground=ground
        if prev is None or not 0 < timestamp-start < 0.5:
            return None
        # Register the current ground patches against the previous metric road grid.
        # Row displacement measures forward travel; lateral search accommodates steering.
        shifts=[]
        for col in (12,38,64):
            patch=ground[22:65,col:col+24]
            search=previous_ground[17:90,col-8:col+32]
            if patch.std()<3: continue
            scores=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
            _,strength,_,(u,v)=cv2.minMaxLoc(scores)
            if strength<.55 or not 0<v<len(scores)-1: continue
            a,b,c=scores[v-1:v+2,u]
            refine=.5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>.0001 else 0.
            shifts.append((v+refine-5)*self.scale)
        if len(shifts)>=2 and np.ptp(shifts)<.22:
            return max(0.,float(np.median(shifts))/(timestamp-start))
        h, w = gray.shape
        mask = np.zeros_like(gray)
        mask[int(.72*h):int(.98*h),int(.10*w):int(.90*w)] = 255
        p = cv2.goodFeaturesToTrack(prev,90,.015,max(2,w//120),mask=mask)
        if p is None or len(p)<8:
            return None
        q,ok,_ = cv2.calcOpticalFlowPyrLK(prev,gray,p,None,winSize=(17,17),maxLevel=2)
        selected = ok.ravel().astype(bool) & (q[:,0,1] > .64*h)
        if selected.sum()<8:
            return None
        old = self.ground_points(p[selected,0])
        new = self.ground_points(q[selected,0])
        transform, inliers = cv2.estimateAffinePartial2D(new,old,method=cv2.RANSAC,
                                                       ransacReprojThreshold=.14,maxIters=150)
        if transform is None or inliers.sum()<6 or inliers.mean()<.42:
            return None
        if abs(np.linalg.norm(transform[0,:2])-1)>.12:
            return None
        return max(0.,float(transform[0,2])/(timestamp-start))
