import os
import numpy as np
import pickle
import math
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from skimage import morphology
from skimage.segmentation import mark_boundaries

import onnxruntime as ort


class OnnxRuntime:
    def __init__(self):
        self.onnxPath = None
        self.device = None

        # ONNX信息
        self.session = None
        self.input_names = []
        self.input_dims = []
        self.output_names = []
        self.output_dims = []

    def initOnnx(self, onnxPath, device="cpu"):
        self.onnxPath = onnxPath
        self.device = device

        self.session = ort.InferenceSession(self.onnxPath)

        # 模型输入
        for i, onnx_input in enumerate(self.session.get_inputs()):
            # 获取名称
            self.input_names.append(onnx_input.name)
            # 获取B, C, H, W
            input_batchSize = onnx_input.shape[0]
            input_channels = onnx_input.shape[1]
            input_height = onnx_input.shape[2]
            input_width = onnx_input.shape[3]
            self.input_dims.append((input_batchSize, input_channels, input_height, input_width))

        # 模型输出
        for i, onnx_output in enumerate(self.session.get_outputs()):
            # 获取名称
            self.output_names.append(onnx_output.name)
            # 获取dims
            dims = []
            for tmp_dim in onnx_output.shape:
                dims.append(tmp_dim)
            self.output_dims.append(dims)

    def inferAnomaly(self, batchImages):
        vectors = self.session.run([self.output_names[0]], {self.input_names[0]: batchImages})[0]
        return vectors


def denormalization(x, mean=None, std=None):
    mean = np.array(mean)
    std = np.array(std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def plot_fig(image, score, threshold, mean=None, std=None, img_p=None, output_dir="D:/Downloads"):
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
        image = Image.open(imagePath)
        image = image.resize((height, width), Image.ANTIALIAS)  # resize
        image = image.convert('RGB')
        image = np.asarray(image)
        hws.append((image.shape[1], image.shape[0]))
        # 方式2
        # image = Image.open(imagePath)
        # image = image.convert('RGB')
        # image = np.asarray(image)
        # image = cv2.resize(image, (height, width), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # hws.append((image.shape[1], image.shape[0]))

        # normalization
        image = image / 255
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        image[0] = (image[0] - MEAN[0]) / STD[0]
        image[1] = (image[1] - MEAN[1]) / STD[1]
        image[2] = (image[2] - MEAN[2]) / STD[2]
        batchImages.append(image)
    return batchImages, hws


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

def PaDimONNX(
        model_path=r'E:\AIDeploy\Env\DemoData\anomaly\onnxs3\padim_b36.onnx',
        image_path=r"E:\AIDeploy\Env\DemoData\anomaly\all",
        pkl_path=r"E:\AIDeploy\Env\DemoData\anomaly\onnxs3\features.pkl",
        threshold_path=r"E:\AIDeploy\Env\DemoData\anomaly\onnxs3\threshold.txt",
        mean_=[0.335782, 0.335782, 0.335782],
        std_=[0.256730, 0.256730, 0.256730],
        image_size=224,
        suffix="bmp",
        output_dir="E:/"
):
    # 1. 初始化ONNX模型
    onnxR = OnnxRuntime()
    onnxR.initOnnx(model_path)

    # 2. 前处理
    images, hws, all_imagesPath = preAnomaly(onnxR, image_path, mean=mean_, std=std_, suffix=suffix)

    # 3. 推理
    # 读取训练好的模型
    with open(pkl_path, 'rb') as f:
        train_output = pickle.load(f)
    # 读取阈值信息
    with open(threshold_path, 'r') as threshold_file:
        threshold = eval(threshold_file.readline())
        max_score = eval(threshold_file.readline())
        min_score = eval(threshold_file.readline())
    # 推理
    for ii, batch_image in enumerate(images):
        embedding_vectors = onnxR.inferAnomaly(batch_image)

        # 对特征向量进行对比
        B, H_W, C = embedding_vectors.shape
        dist_list = []
        for i in range(H_W):
            mean = train_output[0][i, :]
            conv_inv = np.linalg.inv(train_output[1][i, :, :])
            dist = [mahalanobis(sample[i, :], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, int(math.sqrt(H_W)), int(math.sqrt(H_W)))

        # upsample
        for t in range(B):
            if ii*len(batch_image)+t >= len(all_imagesPath):
                break
            dist_one = np.expand_dims(dist_list[t], axis=0)
            dist_one = dist_one.transpose(1, 2, 0)
            score_map = cv2.resize(dist_one, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map, sigma=4)

            # Normalization
            scores = (score_map - min_score) / (max_score - min_score)  # (49, 224, 224)

            plot_fig(batch_image[t], scores, threshold, mean=mean_, std=std_, img_p=all_imagesPath[ii*len(batch_image)+t], output_dir=output_dir)



if __name__ == "__main__":
    PaDimONNX()
