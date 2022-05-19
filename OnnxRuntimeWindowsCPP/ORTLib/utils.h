#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "params.h"
#include "engine.h"
#include "F_log.h"

// 分类、异常检测、分割前处理
int normalization(
	std::vector<cv::Mat>& cv_images,//输入图像
	std::vector<float> &input_tensor_values,//目标地址
	std::shared_ptr<ORTCore_ctx> ctx, //执行上下文
	Ort::MemoryInfo &memory_info	// ort类型
);

// \! 分类后处理
int clsPost(
	float* floatarr,
	std::vector<std::vector<ClassifyResult>>& outputs,
	const int batch,
	const int num_class
);

// \! 异常检测后处理
int anomalyPost(
	float* floatarr, // onnxruntime 推理的结果
	std::vector<cv::Mat>& outputs, // 存储输出的结果
	const int output_batch, // output的batchsize
	const int output_height,//output的高
	const int output_width//output的宽
);

// \! 分割后处理
int segPost(
	float* floatarr, // onnxruntime 推理的结果
	std::vector<cv::Mat>& outputs, // 存储输出的结果
	const int output_batch, // output的batchsize
	const int output_height,//output的高
	const int output_width//output的宽
);