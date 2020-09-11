import cv2
import numpy as np
import pydicom
import glob
import os
from pathlib import Path
from joblib import Parallel, delayed
import argparse


parser = argparse.ArgumentParser(
            description='Preprocess data')

parser.add_argument('--window-level', default=40, type=int, help='Window Level')
parser.add_argument('--window-width', default=80, type=int, help='Window Width')
parser.add_argument('--data-dir', default='/mnt/data/kaggle/rsna-intracranial-hemorrhage-detection/origin/', type=str, help='Base data dir')
parser.add_argument('--output-dir', default='/mnt/data/kaggle/rsna-intracranial-hemorrhage-detection/preprocess/', type=str, help='Preprocess output dir')


args = parser.parse_args()
WINDOW_LEVEL = args.window_level
WINDOW_WIDTH = args.window_width


input_dir = Path(args.data_dir)
# output_dir = input_dir.parent.parent/f'preprocess/{input_dir.stem}_L{WINDOW_LEVEL}_W{WINDOW_WIDTH}'
output_dir = Path(args.output_dir)/f'{input_dir.stem}_L{WINDOW_LEVEL}_W{WINDOW_WIDTH}'
output_dir.mkdir(exist_ok=True, parents=True)
img_paths = input_dir.glob('*.dcm')


def preprocess(img_path, output_dir):
    image_name = img_path.stem
    # params
    window_min = WINDOW_LEVEL-(WINDOW_WIDTH // 2)
    window_max = WINDOW_LEVEL+(WINDOW_WIDTH // 2)
    # read dicom file
    r = pydicom.read_file(img_path.as_posix())
    # convert to hounsfield unit
    img = (r.pixel_array * r.RescaleSlope) + r.RescaleIntercept
    # apply brain window
    img = np.clip(img, window_min, window_max)
    img = 255 * ((img - window_min)/WINDOW_WIDTH)
    img = img.astype(np.uint8)
    # write to output_dir
    cv2.imwrite((output_dir/f'{image_name}.jpg').as_posix(), img)
    return


def catch_wrapper(img_path, output_dir):
    try:
        preprocess(img_path, output_dir)
    except Exception as e:
        print(e, img_path.stem)


if __name__ == "__main__":
    print(f'======================== Start process {input_dir.stem} for level: {WINDOW_LEVEL}, width: {WINDOW_WIDTH} ========================')
    Parallel(n_jobs=8, verbose=1)(delayed(catch_wrapper)(f, output_dir) for f in img_paths)