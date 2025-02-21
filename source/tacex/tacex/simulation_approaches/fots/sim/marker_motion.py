# Modified version of the source code from:
# @article{zhao2024fots,
#   title={FOTS: A Fast Optical Tactile Simulator for Sim2Real Learning of Tactile-motor Robot Manipulation Skills},
#   author={Zhao, Yongqiang and Qian, Kun and Duan, Boyi and Luo, Shan},
#   journal={IEEE Robotics and Automation Letters},
#   year={2024}
# }
#
# https://github.com/Rancho-zhao/FOTS/tree/main


import cv2
import math
import numpy as np

class MarkerMotion():
    def __init__(self,
                frame0_blur,
                lamb,
                mm2pix=33.898, #FOTS default is 19.58
                N=11,
                M=9,
                W=240,
                H=320,
                is_flow=True):

        # self.model = model
        # self.data = data
        self.frame0_blur = frame0_blur
        print("resize image", frame0_blur.shape)
        if (W,H) != frame0_blur.shape:
            H = frame0_blur.shape[0]
            W = frame0_blur.shape[1]

        self.lamb = lamb

        self.mm2pix = mm2pix
        self.N = N # number of markers in rows
        self.M = M # numbers of markers in columns
        self.W = W
        self.H = H

        self.contact = []
        self.moving = False
        self.rotation = False

        self.mkr_rng = 0.5

        self.x = np.arange(0, self.W, 1)
        self.y = np.arange(0, self.H, 1)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.xx_init, self.yy_init = np.meshgrid(self.x, self.y)

        x = np.arange(int(self.H/(2*self.N)), self.H, int(self.H/self.N)) # get out indices of the markers in the image -> take the first N of them
        y = np.arange(int(self.W/(2*self.M)), self.W, int(self.W/self.M))

        xind, yind = np.meshgrid(x, y)
        self.xind = (xind.reshape([1, -1])[0]).astype(np.int16)
        self.yind = (yind.reshape([1, -1])[0]).astype(np.int16)
        
        #! marker init positions
        self.xx_marker_init = self.xx[xind, yind].reshape([self.M, self.N])
        self.yy_marker_init = self.yy[xind, yind].reshape([self.M, self.N])
        # print("test ", self.yy[xind, yind].size/self.M)
        # self.xx_marker_init = self.xx[xind, yind].reshape([self.M, 17])
        # self.yy_marker_init = self.yy[xind, yind].reshape([self.M, 17]) #todo -1 is workaround for not fitting shapes

        self.xx_marker_curr = self.xx_marker_init
        self.yy_marker_curr = self.yy_marker_init


    def _shear(self, center_x, center_y, lamb, shear_x, shear_y, xx, yy):
        # TODO: add force and torque effect
        g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) * lamb)

        dx, dy = shear_x * g, shear_y * g

        xx_ = xx + dx
        yy_ = yy + dy
        return xx_, yy_

    def _twist(self, center_x, center_y, lamb, theta, xx, yy):

        g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) * lamb)

        dx = xx - center_x
        dy = yy - center_y

        rotx = dx * np.cos(theta) - dy * np.sin(theta)
        roty = dx * np.sin(theta) + dy * np.cos(theta)  

        xx_ = xx + (rotx - dx) * g
        yy_ = yy + (roty - dy) * g
        return xx_, yy_

    def _dilate(self, lamb, xx, yy):
        dx, dy = 0.0, 0.0
        for i in range(len(self.contact)):
            g = np.exp(-(((xx - self.contact[i][1]) ** 2 + (yy - self.contact[i][0]) ** 2)) * lamb)

            dx += self.contact[i][2] * (xx - self.contact[i][1]) * g
            dy += self.contact[i][2] * (yy - self.contact[i][0]) * g

        xx_ = xx + dx
        yy_ = yy + dy
        return xx_, yy_

    def _generate(self,xx,yy):
        img = np.zeros_like(self.frame0_blur.copy())#
        for i in range(self.N):
            for j in range(self.M):
                ini_r = int(self.yy_marker_init[j,i]) #yy_marker_init
                ini_c = int(self.xx_marker_init[j,i]) #xx_marker_init
                r = int(yy[j, i])
                c = int(xx[j, i])
                if r >= self.H or r < 0 or c >= self.W or c < 0:
                    continue
                cv2.circle(img,(c,r), 3, (20,20,20), 4)

                #img[r, c, :] = self.frame0_blur[r, c, :] * 0
                k = 0.001
                pt1 = (ini_c, ini_r)
                pt2 = (c+int(k*(c-ini_c)), r+int(k*(r-ini_r)))
                color = (0, 255, 0)
                cv2.arrowedLine(img, pt1, pt2, color, 2,  tipLength=0.2)

        #img = img[:self.W, :self.H]
        return img

    def _motion_callback(self, xx, yy, depth_map, contact_mask, traj):
        for i in range(self.N):
            for j in range(self.M):
                r = int(yy[j, i])
                c = int(xx[j, i])
                if r >= self.H or r < 0 or c >= self.W or c < 0:
                    continue
                if contact_mask[r,c] == 1.0:
                    h = depth_map[r,c]
                    self.contact.append([r,c,h])
        # print("Max depth ", np.max(depth_map))
        if not self.contact:
            xx,yy = self.xx_marker_init, self.yy_marker_init
        
        xx_,yy_ = self._dilate(self.lamb[0], xx, yy) # normal load
        if len(traj) >= 2:
            # shear load
            # print("traj diff x,y ", (traj[-1][0]-traj[0][0]), (traj[-1][1]-traj[0][1]))
            xx_,yy_ = self._shear(int(traj[0][0]*self.mm2pix + self.H/2), 
                                int(traj[0][1]*self.mm2pix + self.W/2 ),
                                self.lamb[1],
                                int((traj[-1][0]-traj[0][0])*self.mm2pix),
                                int((traj[-1][1]-traj[0][1])*self.mm2pix),
                                xx_,
                                yy_)
            # twist load
            # print("theta is ", np.rad2deg(traj[-1][2]-traj[0][2]))
            # theta = max(min(traj[-1][2]-traj[0][2], 50 / 180.0 * math.pi), -50 / 180.0 * math.pi) # -50 = max angle, traj rotation is in rad!
            # xx_,yy_ = self._twist(int(traj[-1][0]*self.mm2pix + self.H/2), 
            #                     int(traj[-1][1]*self.mm2pix + self.W/2),
            #                     self.lamb[2],
            #                     theta,
            #                     xx_,
            #                     yy_)
        return xx_, yy_

    def marker_sim(self, depth_map, contact_mask, traj):
        # xind = (np.random.random(self.N * self.M) * self.W).astype(np.int16)
        # yind = (np.random.random(self.N * self.M) * self.H).astype(np.int16)

        xind = self.xind
        yind = self.yind

        xx = self.xx_marker_init#[xind, yind].reshape([self.M, self.N])
        yy = self.yy_marker_init#[xind, yind].reshape([self.M, self.N])
        #! updated marker positions
        xx_marker_, yy_marker_ = self._motion_callback(xx, yy, depth_map, contact_mask, traj)
        # self.xx_marker_curr += xx_marker_.astype(int)
        # self.yy_marker_curr += yy_marker_.astype(int)

        self.contact = []

        return xx_marker_, yy_marker_

    def _marker_motion(self, depth_map, contact_mask, traj):        
        x = np.arange(int(self.H/(2*self.N)), self.H, int(self.H/self.N)) # get indices of the markes in the image -> take the first N of them
        y = np.arange(int(self.W/(2*self.M)), self.W, int(self.W/self.M))

        xind, yind = np.meshgrid(x, y)
        xind = (xind.reshape([1, -1])[0]).astype(np.int16)
        yind = (yind.reshape([1, -1])[0]).astype(np.int16)
        
        xx_marker, yy_marker = self.xx[xind, yind].reshape([self.M, self.N]), self.yy[xind, yind].reshape([self.M, self.N])
        self.xx,self.yy = xx_marker, yy_marker

        img = self._generate(xx_marker, yy_marker)
        # if self.contact:
        xx_marker_, yy_marker_ = self._motion_callback(xx_marker, yy_marker, depth_map, contact_mask, traj)
        img = self._generate(xx_marker_, yy_marker_)
        self.contact = []

        return img