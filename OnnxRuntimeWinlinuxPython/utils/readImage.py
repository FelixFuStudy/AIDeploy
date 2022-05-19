# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import os
import cv2
from PIL import Image
import numpy as np

__all__ = ['allImagePath', 'loadBatchImages']


def allImagePath(imagePath, batchSize=4, imageSuffix="jpg"):
    """
    以batchsize形式，获取文件夹下所有图片
    :param imagePath:
    :param batchSize:
    :param imageSuffix:
    :return:
    """
    all_imagePath = [os.path.join(imagePath, img) for img in os.listdir(os.path.join(imagePath))
                     if img.endswith(imageSuffix)]  # 获得imagePath路径下所有图片的路径
    all_imagePath = np.asarray(all_imagePath)  # 将all_imagePath转为ndarray格式，这样可以reshape

    # 不能被batchsize整除的余数
    tempImageIndex = len(all_imagePath) % batchSize
    if tempImageIndex == 0:  # 当图片数量整除batchsize时
        batchImages = all_imagePath.reshape(-1, batchSize)
        return list(batchImages), None, list(all_imagePath)
    else:   # 当图片数量不能整除batchsize时
        tempImagePath = all_imagePath[-tempImageIndex:]
        batchImages = all_imagePath[:-tempImageIndex].reshape(-1, batchSize)
        return list(batchImages), list(tempImagePath), list(all_imagePath)


def loadBatchImages(batchImagePath, height, width, MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]):
    batchImages = []
    hws = []
    for imagePath in batchImagePath:
        image = Image.open(imagePath)
        image = np.asarray(image)
        hws.append((image.shape[1], image.shape[0]))
        # resize
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_BITS2)
        # to tensor
        image = image / 255
        is_gray = len(image.shape)
        if is_gray == 2:
            image = np.expand_dims(image, axis=2)
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        # normal
        if is_gray == 2:
            image[0] = (image[0] - MEAN[0]) / STD[0]
        else:
            image[0] = (image[0] - MEAN[0]) / STD[0]
            image[1] = (image[1] - MEAN[1]) / STD[1]
            image[2] = (image[2] - MEAN[2]) / STD[2]
        batchImages.append(image)
    return batchImages, hws


def loadBatchImagesAnomaly(batchImagePath, height, width, MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225]):
    batchImages = []
    hws = []
    for imagePath in batchImagePath:
        # 方式1和方式2是基本等价的，方式1是AICore train、demo中使用的resize方法，而方式2是使用onnx部署时方法，因为C++用的是opencv，所以需要将pillow库修改为opencv库
        # 方式1
        # image = Image.open(imagePath)
        # image = image.resize((height, width), Image.ANTIALIAS)  # resize
        # image = image.convert('RGB')
        # image = np.asarray(image)
        # hws.append((image.shape[1], image.shape[0]))
        # 方式2
        image = Image.open(imagePath)
        image = image.convert('RGB')
        image = np.asarray(image)
        # image = cv2.resize(image, (height, width), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)
        hws.append((image.shape[1], image.shape[0]))

        # normalization
        image = image / 255
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        image[0] = (image[0] - MEAN[0]) / STD[0]
        image[1] = (image[1] - MEAN[1]) / STD[1]
        image[2] = (image[2] - MEAN[2]) / STD[2]
        batchImages.append(image)
    return batchImages, hws


if __name__ == "__main__":
    batchImagesPath, batchImagesPath2 = allImagePath(r"E:\AIDeploy\Env\DemoData\segmentation\images")
    for batchImagePath in batchImagesPath:
        batchImages = loadBatchImages(batchImagePath, 480, 480)
    batchImages2 = loadBatchImages(batchImagesPath2, 480, 480)

