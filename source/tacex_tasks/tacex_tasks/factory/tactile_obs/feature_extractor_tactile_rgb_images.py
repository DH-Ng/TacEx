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

    def __init__(self, num_keypoints=3, coordinate_dim=3, num_images=2):
        super().__init__()
        self.num_images = num_images
        num_channel = (
            3 * num_images
        )  # 6, when we use two tactile rgb images concatenated

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
        # x: (N, H, W, 3 * num_images)
        x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW

        # Normalize each RGB image independently.
        x = torch.cat(
            [
                self.data_transforms(x[:, i * 3 : (i + 1) * 3])
                for i in range(self.num_images)
            ],
            dim=1,
        )

        # extract visual features
        cnn_x = self.cnn(x)

        # [N, 128, 1, 1] -> [N, 128]
        cnn_x = cnn_x.flatten(1)

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

    checkpoint_path: str = ""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    single_image: bool = False


class TactileRGBFeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(
        self,
        cfg: TactileRGBFeatureExtractorCfg,
        device: str,
        log_dir: str | None = None,
    ):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            log_dir: Directory to save checkpoints. If None, uses local "logs" folder resolved with respect to this file.
        """

        self.cfg = cfg
        self.device = device
        self.single_image = cfg.single_image
        self.num_images = 1 if self.single_image else 2

        # Feature extractor model
        self.feature_extractor = TactileRGBFeatureExtractorNetwork(
            num_keypoints=3, coordinate_dim=3, num_images=self.num_images
        )
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
            print("[INFO - Feature Extractor]: Trying to load checkpoint...")
            if self.cfg.checkpoint_path == "":
                list_of_files = glob.glob(self.log_dir + "/*.pth")
                if len(list_of_files) == 0:
                    print(
                        "[INFO - Feature Extractor]: No checkpoint for feature extractor found!"
                    )
                    print("[INFO - Feature Extractor]: Training from scratch...")
                else:
                    latest_file = max(list_of_files, key=os.path.getctime)
                    checkpoint = latest_file
                    print(
                        f"[INFO - Feature Extractor]: Loading feature extractor checkpoint from {checkpoint}"
                    )
                    self.feature_extractor.load_state_dict(
                        torch.load(checkpoint, weights_only=True)
                    )
            else:
                checkpoint = self.cfg.checkpoint_path
                print(
                    f"[INFO - Feature Extractor]: Loading feature extractor checkpoint from {checkpoint}"
                )
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

    def _save_images(self, image: torch.Tensor, filename: str):
        """Save one RGB image batch to the feature-extractor image folder."""
        save_images_to_file(
            image,
            os.path.join(self.log_dir, "feature_extractor_images", filename),
        )

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
            if left_sensor_rgb_img is not None:
                self._save_images(
                    left_sensor_rgb_img, f"sensor_left_rgb_img_{self.step_count}.png"
                )
            if right_sensor_rgb_img is not None:
                self._save_images(
                    right_sensor_rgb_img, f"sensor_right_rgb_img_{self.step_count}.png"
                )

        if self.single_image:
            if right_sensor_rgb_img is not None and left_sensor_rgb_img is not None:
                raise ValueError("Pass only one image when single_image=True.")
            if left_sensor_rgb_img is not None:
                images = left_sensor_rgb_img / 255.0
            else:
                images = right_sensor_rgb_img / 255.0
        else:
            if right_sensor_rgb_img is None or left_sensor_rgb_img is None:
                raise ValueError(
                    "right_sensor_rgb_img is required when single_image=False."
                )
            left_sensor_rgb_img = left_sensor_rgb_img / 255.0
            right_sensor_rgb_img = right_sensor_rgb_img / 255.0
            images = torch.cat((left_sensor_rgb_img, right_sensor_rgb_img), dim=-1)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(images)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % self.cfg.save_step_frequency == 0:
                        print(
                            f"Saving feature extractor at {self.step_count}. Loss = {pose_loss.detach().cpu().numpy()}"
                        )
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
            predicted_pose = self.feature_extractor(images)
            return None, predicted_pose
