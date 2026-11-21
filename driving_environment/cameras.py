"""Rectified image formation in the bicycle simulator. Truth is private to rendering."""
import cv2
import numpy as np

W, H = 192, 112
F = 115.
BASELINE = .32
HEIGHT = 1.5
K = np.array([[F,0,W/2],[0,F,H/2],[0,0,1.]])


class RoadCameras:
    def __init__(self, rng, path, obstacle_x):
        self.obstacle_x = obstacle_x
        self.positions = (-BASELINE/2, BASELINE/2)
        # Static textured road atlas at 10 cm/pixel, with two visible lane boundaries.
        self.road = rng.integers(35,125,(260,900),dtype=np.uint8)
        self.panel = rng.integers(60,205,(32,36),dtype=np.uint8)
        x = np.arange(-10,80,.1)
        for side in (-1,1):
            y = np.array([path(v) for v in x]) + side*1.75
            line = np.int32(np.column_stack(((x+10)*10,130-y*10)))
            cv2.polylines(self.road,[line],False,245,2)
        v,u = np.mgrid[H//2+1:H,:W].astype(np.float32)
        self.z = np.broadcast_to(np.minimum(75., F*HEIGHT/(v-H/2)),u.shape).copy()
        self.right = (u-W/2)*self.z/F
        self.panel_corners = np.float32([[0,0],[35,0],[35,31],[0,31]])

    def pair(self,x,y,yaw):
        cs,sn = float(np.cos(yaw)), float(np.sin(yaw))
        # Vehicle world coordinates use positive-left. Camera x uses positive-right.
        wx = x+(self.z+1.6)*cs+self.right*sn
        wy = y+(self.z+1.6)*sn-self.right*cs
        imgs = []
        for side in self.positions:
            frame = np.full((H,W),150,np.uint8)
            frame[H//2+1:] = cv2.remap(self.road,(wx+side*sn+10)*10,130-(wy-side*cs)*10,
                                          cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            if self.obstacle_x is not None:
                # Textured vertical rear panel, projected separately into the two cameras.
                dx,dy = self.obstacle_x-x,-y
                forward = dx*cs+dy*sn-1.6
                right = dx*sn-dy*cs-side
                if forward>.35:
                    u0,u1 = W/2+F*(right+np.array([-.75,.75]))/forward
                    v0,v1 = H/2+F*np.array([.05,HEIGHT])/forward
                    target=np.float32([[u0,v0],[u1,v0],[u1,v1],[u0,v1]])
                    lo,hi = max(0,int(np.ceil(u0))),min(W,int(np.ceil(u1)))
                    top,bottom = max(0,int(np.ceil(v0))),min(H,int(np.ceil(v1)))
                    if hi>lo and bottom>top:
                        mat=cv2.getPerspectiveTransform(self.panel_corners,target)
                        mat[0] -= lo*mat[2]
                        mat[1] -= top*mat[2]
                        frame[top:bottom,lo:hi]=cv2.warpPerspective(self.panel,mat,(hi-lo,bottom-top),
                                                                  borderMode=cv2.BORDER_REPLICATE)
            imgs.append(frame)
        return imgs
