# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

from loguru import logger
import onnxruntime as ort

__all__ = ['OnnxRuntime']


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
        logger.info("Init ONNX model")
        self.onnxPath = onnxPath
        self.device = device

        logger.info("Loading onnx file ......")
        self.session = ort.InferenceSession(self.onnxPath)

        # 模型输入
        logger.info("ONNX input Info ...... ")
        for i, onnx_input in enumerate(self.session.get_inputs()):
            logger.info("--ONNX input_{} info--".format(i))
            logger.info("Input Name:{}".format(onnx_input.name))
            logger.info("Input Shape:{}".format(onnx_input.shape))
            logger.info("Input Type:{}".format(onnx_input.type))
            # 获取名称
            self.input_names.append(onnx_input.name)
            # 获取B, C, H, W
            input_batchSize = onnx_input.shape[0]
            input_channels = onnx_input.shape[1]
            input_height = onnx_input.shape[2]
            input_width = onnx_input.shape[3]
            self.input_dims.append((input_batchSize, input_channels, input_height, input_width))

        # 模型输出
        logger.info("ONNX output Info ...... ")
        for i, onnx_output in enumerate(self.session.get_outputs()):
            logger.info("--ONNX output_{} info--".format(i))
            logger.info("Output Name:{}".format(onnx_output.name))
            logger.info("Output Shape:{}".format(onnx_output.shape))
            logger.info("Output Type:{}".format(onnx_output.type))
            # 获取名称
            self.output_names.append(onnx_output.name)
            # 获取dims
            dims = []
            for tmp_dim in onnx_output.shape:
                dims.append(tmp_dim)
            self.output_dims.append(dims)

    def inferSeg(self, batchImages):
        prediction = self.session.run([self.output_names[0]], {self.input_names[0]: batchImages})[0]
        return prediction

    def inferCls(self, batchImages):
        prediction = self.session.run([self.output_names[0]], {self.input_names[0]: batchImages})[0]
        return prediction

    def inferAnomaly(self, batchImages):
        vectors = self.session.run([self.output_names[0]], {self.input_names[0]: batchImages})[0]
        return vectors




