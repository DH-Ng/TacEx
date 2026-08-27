# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class TactileRGBFeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the PEG from tactile RGB image data.


    3 keypoints x 3 coordinates = 9 values
    -> For keypoints we use "top", "middle" and "bottom" part of the object.
    """

    def __init__(self, num_keypoints=3, coordinate_dim=3):
        super().__init__()
        num_channel = 6  # use two tactile rgb image concatenated

        output_dim = num_keypoints * coordinate_dim
        # Thats for image res. 120x120
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.LayerNorm([16, 58, 58]),
        #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.LayerNorm([32, 28, 28]),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.LayerNorm([64, 13, 13]),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.LayerNorm([128, 6, 6]),
        #     nn.AvgPool2d(6),
        # )

        # For 32x32 images
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(16, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Sequential(
            nn.Linear(128, output_dim),
        )

        self.data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, x):
        # convert NHWC to NCHW
        x = x.permute(0, 3, 1, 2)

        # Normalize the rgb image
        x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :]) # left sensor image
        x[:, 3:6, :, :] = self.data_transforms(x[:, 3:6, :, :]) # right sensor image

        # extract visual features
        cnn_x = self.cnn(x)
    
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class TactileRGBFeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    save_step_frequency: int = 1200
    """After how many steps a checkpoint should be saved"""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class TactileRGBFeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(
        self, cfg: TactileRGBFeatureExtractorCfg, device: str, log_dir: str | None = None
    ):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            log_dir: Directory to save checkpoints. If None, uses local "logs" folder resolved with respect to this file.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = TactileRGBFeatureExtractorNetwork(num_keypoints=3, coordinate_dim=3)
        self.feature_extractor.to(self.device)

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "logs"
            )
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.write_image_to_file:
            if not os.path.exists(f"{self.log_dir}/feature_extractor_images"):
                os.makedirs(f"{self.log_dir}/feature_extractor_images")

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            if len(list_of_files) == 0:
                print("[INFO] No checkpoint for feature extractor found!")
                print("[INFO] Training from scratch...")
            else:
                latest_file = max(list_of_files, key=os.path.getctime)
                checkpoint = latest_file
                print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
                self.feature_extractor.load_state_dict(
                    torch.load(checkpoint, weights_only=True)
                )

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(
                self.feature_extractor.parameters(), lr=1e-4
            )
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self, left_sensor_rgb_img: torch.Tensor, right_sensor_rgb_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocesses the input images.

        Args:
            left_sensor_rgb_img (torch.Tensor): Tactile RGB image tensor of left sensor. Shape: (N, H, W, 3).
            right_sensor_rgb_img (torch.Tensor): Tactile RGB image tensor of right sensor. Shape: (N, H, W, 3).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed RGB, depth, and segmentation
        """
        left_sensor_rgb_img = left_sensor_rgb_img / 255.0
        right_sensor_rgb_img = right_sensor_rgb_img / 255.0

        return left_sensor_rgb_img, right_sensor_rgb_img

    def _save_images(
        self, left_sensor_rgb_img: torch.Tensor, right_sensor_rgb_img: torch.Tensor
    ):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3).
        """
        save_images_to_file(left_sensor_rgb_img, f"{self.log_dir}/feature_extractor_images/left_sensor_rgb_img_{self.step_count}.png")
        save_images_to_file(right_sensor_rgb_img, f"{self.log_dir}/feature_extractor_images/right_sensor_rgb_img_{self.step_count}.png")

    def step(
        self,
        left_sensor_rgb_img: torch.Tensor, 
        right_sensor_rgb_img: torch.Tensor,
        gt_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3).
            gt_pose (torch.Tensor): Ground truth pose tensor (positions of the keypoints). Shape: (N, 9), because 3 points with 3 coordinates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """

        if self.cfg.write_image_to_file:
            self._save_images(left_sensor_rgb_img, right_sensor_rgb_img)

        left_sensor_rgb_img, right_sensor_rgb_img = self._preprocess_images(
            left_sensor_rgb_img, right_sensor_rgb_img
        )

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    img_input = torch.cat(
                        (left_sensor_rgb_img, right_sensor_rgb_img), dim=-1
                    )
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % self.cfg.save_step_frequency == 0:
                        print(f"Saving feature extractor at {self.step_count}. Loss = {pose_loss.detach().cpu().numpy()}")
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(
                                self.log_dir,
                                f"cnn_step_{self.step_count}_loss_{pose_loss.detach().cpu().numpy()}.pth",
                            ),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            img_input = torch.cat((left_sensor_rgb_img, right_sensor_rgb_img), dim=-1)
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
