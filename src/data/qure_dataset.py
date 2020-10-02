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

from .sampler import adj_slices_sampler, nonadj_slices_sampler
from .utils import apply_window, pad_slices


class Qure500DS(Dataset):
    def __init__(self, cfg, data_dir):
        super(Qure500DS, self).__init__()
        self.data_dir = data_dir
        self.study_ids = sorted(os.listdir(self.data_dir))
        self.totensor = pytorch.transforms.ToTensor(
            normalize={"mean": cfg.DATA.MEAN,
                       "std": cfg.DATA.STD}
        )

    def __len__(self):
        return len(self.study_ids)

    def _load_img(self, file_path):
        img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        img_tensor = self.totensor(image=img)["image"]
        return img_tensor

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        study_dir = os.path.join(self.data_dir, study_id)

        subfolders = os.listdir(study_dir)
        if len(subfolders) > 2:
            # print(study_id)
            if study_id == "CQ500-CT-162":
                subfolder = "CT Plain-3"
            elif study_id == "CQ500-CT-172":
                subfolder = "CT 55mm Plain"
            elif study_id == "CQ500-CT-337":
                subfolder = "CT 2.55mm"
            elif study_id == "CQ500-CT-470":
                subfolder = "CT PRE CONTRAST 5MM STD"
            elif study_id == "CQ500-CT-73":
                subfolder = "CT Plain"
            else:
                subfolder = subfolders[0]
        else:
            subfolder = subfolders[0]
        study_dir = os.path.join(study_dir, subfolder)

        img_nums = sorted([int(img_name.split(".")[0]) for img_name
                           in os.listdir(study_dir)])
        if study_id == "CQ500-CT-345":
            # remove non-head CT slices
            img_nums = [i for i in img_nums if i < 24]

        img_names = [str(i) + ".jpg" for i in img_nums]
        img_paths = [os.path.join(study_dir, img_name) for img_name
                     in img_names]
        imgs = [self._load_img(img_path) for img_path in img_paths]
        imgs = torch.stack(imgs)

        return imgs, study_id