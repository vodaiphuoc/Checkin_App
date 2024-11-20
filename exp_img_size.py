"""This file is only for local development """
import uuid
import cv2 as cv
import glob
from tqdm import tqdm
import pandas as pd

user_folders = glob.glob("face_dataset\images\*")

img_size = []
for user_folder in user_folders:
    user_name = user_folder.split('\\')[-1]

    user_imgs = []
    img_path = []
    for path in glob.glob(f"face_dataset\images\{user_name}\*"):
        image = cv.imread(path)
        imgH, imgW, _ = image.shape
        if imgW/imgH > 0.9:
            newW = imgH*960//1280
            img = image[:,imgW//2-newW//2: imgW//2+newW//2]

            cv.imwrite(path.replace('.jpg','') + str(uuid.uuid1())+ '.jpg',img)