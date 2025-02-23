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
                num_markers_col=11,
                num_markers_row=9,
                tactile_img_width=240,
                tactile_img_height=320,
                x0=0,
                y0=0,
                is_flow=True):

        self.frame0_blur = frame0_blur

        self.lamb = lamb

        self.mm2pix = mm2pix
        self.num_markers_col = num_markers_col 
        self.num_markers_row = num_markers_row 
        self.tactile_img_width = tactile_img_width
        self.tactile_img_height = tactile_img_height

        self.contact = []
        self.moving = False
        self.rotation = False

        self.mkr_rng = 0.5

        self.x = np.arange(0, self.tactile_img_width, 1) # x is column-wise defined
        self.y = np.arange(0, self.tactile_img_height, 1) # y is row-wise defined
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # self.xx_init, self.yy_init = np.meshgrid(self.x, self.y)

        # compute marker indices based on number of markers per column/row and image shape
        marker_x_idx = np.linspace(x0, self.tactile_img_width-x0, self.num_markers_col, dtype=int)
        marker_y_idx = np.linspace(y0, self.tactile_img_height-y0, self.num_markers_row, dtype=int)
        # marker_x_idx = np.arange(int(self.tactile_img_width/self.num_markers_col), self.tactile_img_width - int(self.tactile_img_width/self.num_markers_col), int(self.tactile_img_width/self.num_markers_col)) 
        # marker_y_idx = np.arange(int(self.tactile_img_height/self.num_markers_row), self.tactile_img_height - int(self.tactile_img_height/self.num_markers_row), int(self.tactile_img_height/self.num_markers_row))

        marker_x_idx, marker_y_idx = np.meshgrid(marker_x_idx, marker_y_idx)
        self.marker_x_idx = (marker_x_idx.reshape([1, -1])[0]).astype(np.int16)
        self.marker_y_idx = (marker_y_idx.reshape([1, -1])[0]).astype(np.int16)
        
        self.init_marker_x_pos = self.xx[self.marker_y_idx, self.marker_x_idx].reshape([self.num_markers_row, self.num_markers_col])
        self.init_marker_y_pos = self.yy[self.marker_y_idx, self.marker_x_idx].reshape([self.num_markers_row, self.num_markers_col])


        self.marker_x_pos = self.init_marker_x_pos
        self.marker_y_pos = self.init_marker_y_pos


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

    def _generate(self, x_pos_of_all_markers, y_pos_of_all_markers):
        img = np.zeros_like(self.frame0_blur.copy())#
        for i in range(self.num_markers_col):
            for j in range(self.num_markers_row):
                init_y_pos = int(self.yy_marker_init[j,i]) # yy_marker_init (row)
                init_x_pos = int(self.xx_marker_init[j,i]) # xx_marker_init (column)
                y_pos = int(y_pos_of_all_markers[j, i]) # position row-wise
                x_pos = int(x_pos_of_all_markers[j, i]) # position column-wise
                if y_pos >= self.tactile_img_height or y_pos < 0 or x_pos >= self.tactile_img_width or x_pos < 0:
                    continue
                cv2.circle(img,(x_pos,y_pos), 3, (20,20,20), 4)

                #img[r, c, :] = self.frame0_blur[r, c, :] * 0
                k = 0.001
                pt1 = (init_x_pos, init_y_pos)
                pt2 = (init_y_pos+int(k*(y_pos-init_y_pos)), init_x_pos+int(k*(x_pos-init_x_pos)))
                color = (0, 255, 0)
                cv2.arrowedLine(img, pt1, pt2, color, 2,  tipLength=0.2)

        #img = img[:self.tactile_img_width, :self.tactile_img_height]
        return img

    def _motion_callback(self, init_marker_x_pos, init_marker_y_pos, depth_map, contact_mask, traj):
        for i in range(self.num_markers_col):
            for j in range(self.num_markers_row):
                y_pos = int(init_marker_y_pos[j, i]) # row-wise defined
                x_pos = int(init_marker_x_pos[j, i]) # column-wise defined
                if y_pos >= self.tactile_img_height or y_pos < 0 or x_pos >= self.tactile_img_width or x_pos < 0:
                    continue
                if contact_mask[y_pos, x_pos] == 1.0:
                    tactile_img_height = depth_map[y_pos, x_pos]
                    self.contact.append([y_pos, x_pos, tactile_img_height])

        if not self.contact:
            new_x_pos, new_y_pos = init_marker_x_pos, init_marker_y_pos
            return new_x_pos, new_y_pos
        
        # compute marker motion under normal load
        new_x_pos, new_y_pos = self._dilate(self.lamb[0], init_marker_x_pos, init_marker_y_pos) 
        if len(traj) >= 2:
            # under shear load
            # print("traj diff x,y ", (traj[-1][0]-traj[0][0]), (traj[-1][1]-traj[0][1]))
            new_x_pos, new_y_pos = self._shear(int(traj[0][0]*self.mm2pix + self.tactile_img_height/2), 
                                int(traj[0][1]*self.mm2pix + self.tactile_img_width/2 ),
                                self.lamb[1],
                                int((traj[-1][0]-traj[0][0])*self.mm2pix),
                                int((traj[-1][1]-traj[0][1])*self.mm2pix),
                                new_x_pos,
                                new_y_pos)
            # under twist load
            # print("theta is ", np.rad2deg(traj[-1][2]-traj[0][2]))
            # theta = max(min(traj[-1][2]-traj[0][2], 50 / 180.0 * math.pi), -50 / 180.0 * math.pi) # -50 = max angle, traj rotation is in rad!
            # new_x_pos,new_y_pos = self._twist(int(traj[-1][0]*self.mm2pix + self.tactile_img_height/2), 
            #                     int(traj[-1][1]*self.mm2pix + self.tactile_img_width/2),
            #                     self.lamb[2],
            #                     theta,
            #                     new_x_pos,
            #                     new_y_pos)
        return new_x_pos, new_y_pos

    def marker_sim(self, depth_map, contact_mask, traj):
        #! update marker positions
        new_marker_x_pos, new_marker_y_pos = self._motion_callback(self.init_marker_x_pos, self.init_marker_y_pos, depth_map, contact_mask, traj)
        # self.xx_marker_curr += xx_marker_.astype(int)
        # self.yy_marker_curr += yy_marker_.astype(int)

        self.contact = []

        return new_marker_x_pos, new_marker_y_pos

    def _marker_motion(self, depth_map, contact_mask, traj):        
        x = np.arange(int(self.tactile_img_height/(2*self.num_markers_col)), self.tactile_img_height, int(self.tactile_img_height/self.num_markers_col)) # get indices of the markes in the image -> take the first num_markers_col of them
        y = np.arange(int(self.tactile_img_width/(2*self.num_markers_row)), self.tactile_img_width, int(self.tactile_img_width/self.num_markers_row))

        xind, yind = np.meshgrid(x, y)
        xind = (xind.reshape([1, -1])[0]).astype(np.int16)
        yind = (yind.reshape([1, -1])[0]).astype(np.int16)
        
        xx_marker, yy_marker = self.xx[xind, yind].reshape([self.num_markers_row, self.num_markers_col]), self.yy[xind, yind].reshape([self.num_markers_row, self.num_markers_col])
        self.xx,self.yy = xx_marker, yy_marker

        img = self._generate(xx_marker, yy_marker)
        # if self.contact:
        xx_marker_, yy_marker_ = self._motion_callback(xx_marker, yy_marker, depth_map, contact_mask, traj)
        img = self._generate(xx_marker_, yy_marker_)
        self.contact = []

        return img