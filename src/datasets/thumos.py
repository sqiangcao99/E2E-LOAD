#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import pdb

import einops
import numpy as np
import os
import random
import pandas
import torch
import torch.utils.data
from torchvision import transforms
from itertools import chain as chain

import src.utils.logging as logging
from src.utils.env import pathmgr

from . import transform as transform
from . import utils as utils
from . import helper as helper
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import (
    MaskingGenerator,
    MaskingGenerator3D,
    create_random_augment,
)

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Thumos(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB 
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0

        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1  
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )  

        logger.info("Constructing THUMOS {}...".format(mode))
        self._construct_loader()
        self.aug = False  
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0  
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:  
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # Get the data root here;
        self.data_root = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, self.cfg.DATA.PATH_PREFIX
        )

        ## Video Path
        self.video_root = os.path.join(
            self.data_root, self.cfg.DATA.VIDEO_FORDER
        )

        ## Target Path
        self.target_root = os.path.join(
            self.data_root, self.cfg.DATA.TARGET_FORDER
        )
        # self.sessions contians all the {train/val} sessions;
        self.sessions = getattr(
            self.cfg.DATA,
            ("train" if self.mode == "train" else "test").upper()
            + "_SESSION_SET",
        ) 

        
        self._path_to_videos_work = []
        self._path_to_videos_long = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.epoch = 0.0
        
        ## This is the customs defiend function;
        (
            self._path_to_videos_work, 
            self._path_to_videos_long, 
            self._labels,
        ) = helper.load_image_lists(
            self.cfg,
            self.sessions,
            self.video_root,
            self.target_root,
            self.mode,
            return_list=True,
        )  

        self._path_to_videos_work = list(
            chain.from_iterable(  
                [[x] * self._num_clips for x in self._path_to_videos_work]
            ) 
        ) # repeat the videos;

        self._path_to_videos_long = list(
            chain.from_iterable(  
                [[x] * self._num_clips for x in self._path_to_videos_long]
            ) 
        )  # repeat the videos;

        self._labels = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._labels]
            ) 
        )
        
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(
                        len(self._labels) // self._num_clips
                    ) 
                ]
            )
        ) 

        logger.info(
            "Constructing THUMOS dataloader (size: {}).".format(
                len(self._path_to_videos_work),
            )  
        ) 
        assert (
            len(self._path_to_videos_work) > 0
        ), "Failed to load THUMOS dataset"

    def _set_epoch_num(self, epoch):
        self.epoch = epoch 

    def __getitem__(self, index):

        work_frames_path = self._path_to_videos_work[index]

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            (
                long_frames_path,
                long_repeat_times,
                long_key_padding_mask,
            ) = self._path_to_videos_long[index]
        else:
            long_frames_path = None
            long_repeat_times = None
            long_key_padding_mask = None

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1  

            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]  

            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE  
        elif self.mode in ["val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -2

            min_scale = self.cfg.DATA.TEST_CROP_SIZE
            max_scale = self.cfg.DATA.TEST_CROP_SIZE
            crop_size = self.cfg.DATA.TEST_CROP_SIZE

        elif self.mode in ["test"]:  
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )  
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1  
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [
                    self.cfg.DATA.TEST_CROP_SIZE
                ]
            )
            # The testing is deterministic and no jitter should be performed. 
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        target_fps = self.cfg.DATA.TARGET_FPS 

        work_frames, work_time_diff_aug = helper.load_frames(
            work_frames_path,
            max_spatial_scale=min_scale,
            time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
        )  # frames = torch.Size([16, 720, 1280, 3]) T H W C; # time_idx array([[13.36949863, 76.36949863,  0.]])
        
        num_aug = (
            self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL
            * self.cfg.AUG.NUM_SAMPLE  
            if self.mode in ["train"]
            else 1
        )

        f_out_work = [None] * num_aug  
        labels = self._labels[index]  # which is related to all the current labels;
        labels = np.stack(labels, axis=0) # (32, 22)
        
        # Here applys data augmentation;
        if self.cfg.MODEL.LONG_MEMORY_ENABLE:

            long_frames, long_time_diff_aug = helper.load_frames(
                long_frames_path,
                max_spatial_scale=min_scale,
                time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
            )  # frames = torch.Size([16, 720, 1280, 3]) T H W C; # time_idx array([[13.36949863, 76.36949863,  0.]])
            
            # Repeat the long history frames; 
            assert long_time_diff_aug is None
            patch_size = self.cfg.MODEL.CHUNK_SIZE // self.cfg.MODEL.CHUNK_SAMPLE_RATE
            long_frames = einops.rearrange(long_frames, '(nc cz) h w c -> nc cz h w c', cz=patch_size)
            long_frames = torch.repeat_interleave(long_frames, torch.from_numpy(long_repeat_times), dim=0)
            long_frames = einops.rearrange(long_frames, 'nc cz h w c -> (nc cz) h w c')

            f_out_long = [None] * num_aug  
            # Initialize for iterations; 
            idx = -1
            for _ in range(num_aug):  
                idx += 1
                f_out_long[idx] = long_frames.clone().float().float()
                f_out_long[idx] = f_out_long[idx] / 255.0

                if (
                    self.mode in ["train"]
                    and self.cfg.DATA.SSL_COLOR_JITTER  # false: Self-supervised data augmentation;
                ):
                    f_out_long[idx] = transform.color_jitter_video_ssl(
                        f_out_long[idx],
                        bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,  # [0.4, 0.4, 0.4]
                        hue=self.cfg.DATA.SSL_COLOR_HUE,  # 0.1
                        p_convert_gray=self.p_convert_gray,  # 0.0
                        moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,  # F
                        gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                        gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                    ) 

                if (
                    self.aug and self.cfg.AUG.AA_TYPE
                ):  
                    aug_transform = create_random_augment(
                        input_size=(
                            f_out_long[idx].size(1),
                            f_out_long[idx].size(2),
                        ),
                        auto_augment=self.cfg.AUG.AA_TYPE,
                        interpolation=self.cfg.AUG.INTERPOLATION,
                    )
                    # T H W C -> T C H W.
                    f_out_long[idx] = f_out_long[idx].permute(0, 3, 1, 2)
                    list_img_long = self._frame_to_list_img(f_out_long[idx])
                    list_img_long = aug_transform(list_img_long)
                    f_out_long[idx] = self._list_img_to_frames(list_img_long)
                    f_out_long[idx] = f_out_long[idx].permute(0, 2, 3, 1)
                # Perform color normalization.
                f_out_long[idx] = utils.tensor_normalize(
                    f_out_long[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                f_out_long[idx] = f_out_long[idx].permute(3, 0, 1, 2)

                scl, asp = (
                    self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                    self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                )
                relative_scales = (
                    None
                    if (self.mode not in ["train"] or len(scl) == 0)
                    else scl
                )
                relative_aspect = (
                    None
                    if (self.mode not in ["train"] or len(asp) == 0)
                    else asp
                )

                f_out_long[idx] = utils.spatial_sampling(
                    f_out_long[idx],
                    spatial_idx=spatial_sample_index,  # -1
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP
                    if self.mode in ["train"]
                    else False,  # True;
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE
                    if self.mode in ["train"]
                    else False,  # True;,  # False;
                    aspect_ratio=relative_aspect,  # None;
                    scale=relative_scales,
                    motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT  # False;
                    if self.mode in ["train"]
                    else False,
                )

                if self.rand_erase:  # Default false;
                    erase_transform = RandomErasing(
                        self.cfg.AUG.RE_PROB,
                        mode=self.cfg.AUG.RE_MODE,
                        max_count=self.cfg.AUG.RE_COUNT,
                        num_splits=self.cfg.AUG.RE_COUNT,
                        device="cpu",
                    )
                    f_out_long[idx] = erase_transform(
                        f_out_long[idx].permute(1, 0, 2, 3)
                    ).permute(1, 0, 2, 3)

                if self.cfg.AUG.GEN_MASK_LOADER:  # Default False;
                    mask = self._gen_mask()
                    f_out_long[idx] = f_out_long[idx] + [torch.Tensor(), mask]

            long_frames = f_out_long[0] if num_aug == 1 else f_out_long

        idx = -1
        for _ in range(num_aug):  
            idx += 1
            f_out_work[idx] = work_frames.clone().float()
            f_out_work[idx] = f_out_work[idx] / 255.0

            if (
                self.mode in ["train"]
                and self.cfg.DATA.SSL_COLOR_JITTER  # false: Self-supervised data augmentation;
            ):
                f_out_work[idx] = transform.color_jitter_video_ssl(
                    f_out_work[idx],
                    bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,  # [0.4, 0.4, 0.4]
                    hue=self.cfg.DATA.SSL_COLOR_HUE,  # 0.1
                    p_convert_gray=self.p_convert_gray,  # 0.0
                    moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,  # F
                    gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                    gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                )

            if (
                self.mode in ["train"] and self.aug and self.cfg.AUG.AA_TYPE
            ):  # 'rand-m7-n4-mstd0.5-inc1' 
                aug_transform = create_random_augment(
                    input_size=(
                        f_out_work[idx].size(1),
                        f_out_work[idx].size(2),
                    ), 
                    auto_augment=self.cfg.AUG.AA_TYPE,
                    interpolation=self.cfg.AUG.INTERPOLATION,
                )
                # T H W C -> T C H W.
                f_out_work[idx] = f_out_work[idx].permute(0, 3, 1, 2)
                list_img_work = self._frame_to_list_img(f_out_work[idx])
                list_img_work = aug_transform(list_img_work)
                f_out_work[idx] = self._list_img_to_frames(list_img_work)
                f_out_work[idx] = f_out_work[idx].permute(0, 2, 3, 1)

            # Perform color normalization.
            f_out_work[idx] = utils.tensor_normalize(
                f_out_work[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            f_out_work[idx] = f_out_work[idx].permute(3, 0, 1, 2)

            scl, asp = (
                self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
            )
            relative_scales = (
                None if (self.mode not in ["train"] or len(scl) == 0) else scl
            )
            relative_aspect = (
                None if (self.mode not in ["train"] or len(asp) == 0) else asp
            )

            f_out_work[idx] = utils.spatial_sampling(
                f_out_work[idx],
                spatial_idx=spatial_sample_index,  # -1
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP
                if self.mode in ["train"]
                else False,  # True;,  # True;
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE
                if self.mode in ["train"]
                else False,  # True;,  # False;
                aspect_ratio=relative_aspect,  # None;
                scale=relative_scales,
                motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT  # False;
                if self.mode in ["train"]
                else False,
            )

            if self.rand_erase:  # Default false;
                erase_transform = RandomErasing(
                    self.cfg.AUG.RE_PROB,
                    mode=self.cfg.AUG.RE_MODE,
                    max_count=self.cfg.AUG.RE_COUNT,
                    num_splits=self.cfg.AUG.RE_COUNT,
                    device="cpu",
                )
                f_out_work[idx] = erase_transform(
                    f_out_work[idx].permute(1, 0, 2, 3)
                ).permute(1, 0, 2, 3)

            if self.cfg.AUG.GEN_MASK_LOADER:  # Default False;
                mask = self._gen_mask()
                f_out_work[idx] = f_out_work[idx] + [torch.Tensor(), mask]

        work_frames = f_out_work[0] if num_aug == 1 else f_out_work
        
        if num_aug > 1 and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            labels = [labels] * num_aug

        # if self.cfg.AUG.CURRENT.SHUFFLE.ENABLE:
        #     pdb.set_trace()
        
        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            if self.cfg.DATA.ZERO_MASK:
                long_key_padding_mask_ = np.repeat(long_key_padding_mask, 3)

                long_key_padding_mask_[long_key_padding_mask_==0]=1
                long_key_padding_mask_[long_key_padding_mask_<0]=0
                long_frames = long_frames * long_key_padding_mask_[None,:,None,None]
                long_frames = long_frames.to(torch.float32)
            
            return work_frames, long_frames, long_key_padding_mask, labels 

        else:
            return work_frames, labels

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=np.int)
            n_mask = round(
                self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos_work) 
