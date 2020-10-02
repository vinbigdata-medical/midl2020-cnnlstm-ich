import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from albumentations import pytorch
from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import random
from torch.utils.data import Dataset
import torch

from .sampler import adj_slices_sampler


class RSNAHemorrhageDS(Dataset):
    def __init__(self, cfg, mode="train"):
        super(RSNAHemorrhageDS, self).__init__()
        self.cfg = cfg
        self.CLASSES = self.cfg.CONST.LABELS
        self.mode = mode

        self.train_path = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN)
        self.valid_path = os.path.join(cfg.DIRS.DATA, cfg.DIRS.VALID)
        self.test_path = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TEST)

        self.train_df = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN_CSV))
        self.valid_df = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.VALID_CSV))
        self.test_df = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TEST_CSV))

        if self.mode == "train":
            self.train_aug = A.Compose([
                A.RandomResizedCrop(cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE,
                                    interpolation=cv2.INTER_LINEAR, scale=(0.8, 1)),
                A.OneOf([
                    A.HorizontalFlip(p=1.),
                    A.VerticalFlip(p=1.),
                ]),
                A.OneOf([
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=30,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    A.GridDistortion(
                        distort_limit=0.2,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    A.OpticalDistortion(
                        distort_limit=0.2,
                        shift_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    A.NoOp()
                ]),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(p=1.),
                    A.GaussNoise(p=1.),
                    A.NoOp()
                ]),
                A.OneOf([
                    A.MedianBlur(blur_limit=3, p=1.),
                    A.Blur(blur_limit=3, p=1.),
                    A.NoOp()
                ])
            ])
        self.totensor = pytorch.transforms.ToTensor(
            normalize={"mean": cfg.DATA.MEAN,
                       "std": cfg.DATA.STD})

        self.train_studies = self.train_df["StudyInstanceUID"].unique()
        self.valid_studies = self.valid_df["StudyInstanceUID"].unique()
        self.test_studies = self.test_df["StudyInstanceUID"].unique()

    def _load_img(self, file_path):
        img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            img = self.train_aug(image=img)["image"]

        # Normalize by ImageNet statistics
        img_tensor = self.totensor(image=img)["image"]
        return img_tensor


class RSNAHemorrhageDS2d(RSNAHemorrhageDS):
    def __init__(self, cfg, mode="train"):
        super(RSNAHemorrhageDS2d, self).__init__(cfg, mode)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_df)
        elif self.mode == "valid":
            return len(self.valid_df)
        elif self.mode == "test":
            return len(self.test_df)

    def __getitem__(self, idx):
        if self.mode == "train":
            info = self.train_df.loc[idx]
            data_path = self.train_path
        elif self.mode == "valid":
            info = self.valid_df.loc[idx]
            data_path = self.valid_path
        elif self.mode == "test":
            info = self.test_df.loc[idx]
            data_path = self.test_path

        img_path = os.path.join(data_path, info["image"] + ".jpg")
        img = self._load_img(img_path)
        if self.mode == "train" or self.mode == "valid":
            labels = info[self.CLASSES].values
            labels = labels.astype(np.float32)
            return img, torch.from_numpy(labels).type("torch.FloatTensor")
        else:
            img_id = info["image"]
            return img, img_id


class RSNAHemorrhageDS3d(RSNAHemorrhageDS):
    def __init__(self, cfg, mode="train"):
        super(RSNAHemorrhageDS3d, self).__init__(cfg, mode)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_studies)
        elif self.mode == "valid":
            return len(self.valid_studies)
        elif self.mode == "test":
            return len(self.test_studies)

    def _load_study(self, study_df):
        img_names = study_df["image"].values
        if not self.mode == "test":
            labels = study_df[self.CLASSES].values

        if self.mode == "train":
            nslices = self.cfg.DATA.NUM_SLICES
            l = len(img_names)
            idx = adj_slices_sampler(nslices, l)
            img_names = img_names[idx]
            labels = labels[idx]

        if self.mode == "train":
            data_path = self.train_path
        elif self.mode == "valid":
            data_path = self.valid_path
        elif self.mode == "test":
            data_path = self.test_path

        img_paths = [os.path.join(data_path, img_name + ".jpg")
                     for img_name in img_names]
        imgs = [self._load_img(img_path) for img_path in img_paths]
        imgs = torch.stack(imgs)

        if self.mode == "train" or self.mode == "valid":
            return imgs, torch.from_numpy(labels).type('torch.FloatTensor')

        elif self.mode == "test":
            return imgs, img_names.tolist()

    def __getitem__(self, idx):
        if self.mode == "train":
            study_name = self.train_studies[idx]
            study_df = self.train_df[self.train_df["StudyInstanceUID"] == study_name]
        elif self.mode == "valid":
            study_name = self.valid_studies[idx]
            study_df = self.valid_df[self.valid_df["StudyInstanceUID"] == study_name]
        elif self.mode == "test":
            study_name = self.test_studies[idx]
            study_df = self.test_df[self.test_df["StudyInstanceUID"] == study_name]
        data_bunch = self._load_study(study_df)
        return data_bunch
