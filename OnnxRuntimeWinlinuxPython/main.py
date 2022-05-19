import argparse
import numpy as np
from loguru import logger
import pickle
import math
import random
from random import sample
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
from skimage import morphology
from skimage.segmentation import mark_boundaries
import cv2

from OnnxRuntimeLib import OnnxRuntime
from utils import preCls, postCls, preSeg, postSeg, preAnomaly
from utils import Nearest, plot_fig


if __name__ == "__main__":
    # exporter settings
    parser = argparse.ArgumentParser()
    # for cls
    # parser.add_argument('--model', type=str,
    #                     default=r'E:\AIDeploy\Env\DemoData\classification\onnxs\PZb0b8.onnx',
    #                     help="set model checkpoint path")
    # parser.add_argument('--image', type=str,
    #                     default=r"E:\AIDeploy\Env\DemoData\classification\images",
    #                     help='input image to use')
    # parser.add_argument('--type', type=str,
    #                     default=r"cls",
    #                     help='cls, seg, det, anomaly')
    pass
    # for seg
    # parser.add_argument('--model', type=str,
    #                     default=r'E:\AIDeploy\Env\DemoData\segmentation\onnxs\PSPNet2_resnet50.onnx',
    #                     help="set model checkpoint path")
    # parser.add_argument('--image', type=str,
    #                     default=r"E:\AIDeploy\Env\DemoData\segmentation\images",
    #                     help='input image to use')
    # parser.add_argument('--type', type=str,
    #                     default=r"seg",
    #                     help='cls, seg, det, anomaly')
    pass
    # for anomaly
    parser.add_argument('--model', type=str,
                        default=r'E:\AIDeploy\Env\DemoData\anomaly\onnxs\PaDiM2_b36.onnx',
                        help="set model checkpoint path")
    parser.add_argument('--image', type=str,
                        default=r"E:\AIDeploy\Env\DemoData\anomaly\images",
                        help='input image to use')
    parser.add_argument('--type', type=str,
                        default=r"anomaly",
                        help='cls, seg, det, anomaly')
    # parser.add_argument('--pkl', type=str,
    #                     default=r"E:\AIDeploy\Env\DemoData\anomaly\onnxs\features.pkl",
    #                     help='pkl path')
    parser.add_argument('--threshold', type=str,
                        default=r"E:\AIDeploy\Env\DemoData\anomaly\onnxs\threshold.txt",
                        help='threshold.txt path')

    args = parser.parse_args()
    if args.type == "cls":
        # 1. 初始化ONNX模型
        onnxR = OnnxRuntime()
        onnxR.initOnnx(args.model)

        # 2. 前处理
        images, hws, all_imagesPath = preCls(onnxR, args.image, mean=[0], std=[1])

        # 3. 推理
        results = []
        for i, batch_image in enumerate(images):
            result = onnxR.inferCls(batch_image)
            results.append(result)

        results = np.asarray(results)
        results = np.concatenate(results, axis=0)
        hws = np.concatenate(hws, axis=0)

        # 4. 后处理
        postCls(results, hws, all_imagesPath)
    elif args.type == "seg":
        # 1. 初始化ONNX模型
        onnxR = OnnxRuntime()
        onnxR.initOnnx(args.model)

        # 2. 前处理
        images, hws, all_imagesPath = preSeg(onnxR, args.image, mean=[0.45734706, 0.43338275, 0.40058118], std=[0.23965294, 0.23532275, 0.2398498])

        # 3. 推理
        results = []
        for batch_image in images:
            result = onnxR.inferSeg(batch_image)
            results.append(result)

        results = np.asarray(results)
        results = np.concatenate(results, axis=0)
        hws = np.concatenate(hws, axis=0)

        # 4. 后处理
        postSeg(results, hws, all_imagesPath, 21)
    elif args.type == "anomaly":
        # 1. 初始化ONNX模型
        onnxR = OnnxRuntime()
        onnxR.initOnnx(args.model)

        # 2. 前处理
        images, hws, all_imagesPath = preAnomaly(onnxR, args.image, mean=[0.335782, 0.335782, 0.335782],
                                                 std=[0.256730, 0.256730, 0.256730], suffix="bmp")

        # 3. 推理
        # # 读取训练好的模型
        # with open(args.pkl, 'rb') as f:
        #     train_output = pickle.load(f)
        # 读取阈值信息
        with open(args.threshold, 'r') as threshold_file:
            threshold = eval(threshold_file.readline())
            logger.info("threshold is {}".format(str(threshold)))
            max_score = eval(threshold_file.readline())
            logger.info("max_score is {}".format(str(max_score)))
            min_score = eval(threshold_file.readline())
            logger.info("min_score is {}".format(str(min_score)))
        # 推理
        for ii, batch_image in enumerate(images):
            scores = onnxR.inferAnomaly(batch_image)
            for i in range(len(batch_image)):
                plot_fig(
                    batch_image[i], scores[i], threshold,
                    mean=[0.335782, 0.335782, 0.335782], std=[0.256730, 0.256730, 0.256730],
                    img_p=all_imagesPath[ii*len(batch_image)+i],
                    output_dir=r"E:/"

                )

