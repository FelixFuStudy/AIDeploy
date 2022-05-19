# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import numpy as np
from loguru import logger

from .readImage import loadBatchImages, allImagePath, loadBatchImagesAnomaly
from .palette import colorize_mask, get_palette


def preCls(onnxR, image_path, mean, std):
    images = []  # 存放所有图片，ndarray格式
    hws = []  # 存放原始图片的hw，方便显示时resize到原始大小
    batchImagesPath, batchImagesPath2, all_imagesPath = allImagePath(image_path, batchSize=onnxR.input_dims[0][0], imageSuffix="bmp")
    onnx_batchsize = onnxR.input_dims[0][0]
    onnx_channels = onnxR.input_dims[0][1]
    onnx_height = onnxR.input_dims[0][2]
    onnx_width = onnxR.input_dims[0][3]
    for batchImagePath in batchImagesPath:
        batchImages, hw = loadBatchImages(batchImagePath, onnx_height, onnx_width, MEAN=mean, STD=std)
        images.append(batchImages)
        hws.append(hw)
    if batchImagesPath2 is not None:
        batchImages2, hw = loadBatchImages(batchImagesPath2, onnx_height, onnx_width, MEAN=mean, STD=std)
        hws.append(hw)
        while len(batchImages2) < onnx_batchsize:
            batchImages2.append(np.random.random((onnx_channels, onnx_height, onnx_width)))
        images.append(batchImages2)
    return images, hws, all_imagesPath


def postCls(results, hws, all_imagesPath):
    preds = np.argmax(results, axis=1)

    for i in range(len(all_imagesPath)):
        logger.info("pred:{}, image path:{}".format(preds[i], all_imagesPath[i]))


def preSeg(onnxR, image_path, mean, std):
    images = []  # 存放所有图片，ndarray格式
    hws = []  # 存放原始图片的hw，方便显示时resize到原始大小
    batchImagesPath, batchImagesPath2, all_imagesPath = allImagePath(image_path, batchSize=onnxR.input_dims[0][0])
    onnx_batchsize = onnxR.input_dims[0][0]
    onnx_channels = onnxR.input_dims[0][1]
    onnx_height = onnxR.input_dims[0][2]
    onnx_width = onnxR.input_dims[0][3]
    for batchImagePath in batchImagesPath:
        batchImages, hw = loadBatchImages(batchImagePath, onnx_height, onnx_width, MEAN=mean, STD=std)
        images.append(batchImages)
        hws.append(hw)
    if batchImagesPath2 is not None:
        batchImages2, hw = loadBatchImages(batchImagesPath2, onnx_height, onnx_width, MEAN=mean, STD=std)
        hws.append(hw)
        while len(batchImages2) < onnx_batchsize:
            batchImages2.append(np.random.random((onnx_channels, onnx_height, onnx_width)))
        images.append(batchImages2)

    return images, hws, all_imagesPath


def postSeg(results, hws, all_imagesPath, num_class):
    for img, img_path, hw in zip(results[:len(all_imagesPath)], all_imagesPath, hws):
        # 存储灰度图
        # im = Image.fromarray(img)
        # im.convert('L').save(img_path[:-3]+"png")
        # 存储彩色图
        im = np.uint8(img)
        im = colorize_mask(im, get_palette(num_class))  # 21 是number class
        im = im.resize(hw)
        im.save(img_path[:-4] + "_mask.png")


def preAnomaly(onnxR, image_path, mean, std, suffix):
    images = []  # 存放所有图片，ndarray格式
    hws = []  # 存放原始图片的hw，方便显示时resize到原始大小
    batchImagesPath, batchImagesPath2, all_imagesPath = allImagePath(image_path, batchSize=onnxR.input_dims[0][0], imageSuffix=suffix)
    onnx_batchsize = onnxR.input_dims[0][0]
    onnx_channels = onnxR.input_dims[0][1]
    onnx_height = onnxR.input_dims[0][2]
    onnx_width = onnxR.input_dims[0][3]
    for batchImagePath in batchImagesPath:
        batchImages, hw = loadBatchImagesAnomaly(batchImagePath, onnx_height, onnx_width, MEAN=mean, STD=std)
        images.append(batchImages)
        hws.append(hw)
    if batchImagesPath2 is not None:
        batchImages2, hw = loadBatchImagesAnomaly(batchImagesPath2, onnx_height, onnx_width, MEAN=mean, STD=std)
        hws.append(hw)
        while len(batchImages2) < onnx_batchsize:
            batchImages2.append(np.random.random((onnx_channels, onnx_height, onnx_width)))
        images.append(batchImages2)

    return images, hws, all_imagesPath


def Nearest(img, bigger_height, bigger_width, channels):
    near_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]
            near_row = round(row)
            near_col = round(col)
            if near_row == img.shape[0] or near_col == img.shape[1]:
                near_row -= 1
                near_col -= 1

            near_img[i][j] = img[near_row][near_col]

    return near_img


def denormalization(x, mean=[0.335782, 0.335782, 0.335782], std=[0.256730, 0.256730, 0.256730]):
    mean = np.array(mean)
    std = np.array(std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x

def plot_fig(image, score, threshold, mean=[0.335782, 0.335782, 0.335782], std=[0.256730, 0.256730, 0.256730], img_p=None, output_dir="D:/Downloads"):
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    from skimage import morphology
    from skimage.segmentation import mark_boundaries
    import time

    vmax = score.max() * 255.
    vmin = score.min() * 255.

    image = denormalization(image, mean, std)
    heat_map = score * 255
    mask = score
    threshold = np.median(score) if threshold is None else threshold

    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(image, mask, color=(1, 0, 0), mode='thick')
    fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(image)
        ax_img[0].title.set_text('Image')

        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(image, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')

        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')

        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        dstPath = os.path.join(output_dir, "pictures")
        os.makedirs(dstPath, exist_ok=True)
        ngtype = img_p.replace('\\','/').split('/')[-2]
        image_name = ngtype + "_" + img_p.replace('\\','/').split('/')[-1][:-4] + ".png"
        fig_img.savefig(os.path.join(dstPath, image_name), dpi=100)
        plt.close()

