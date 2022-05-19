/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : ORTCore 核心库
*****************************************************************************/
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "params.h"
#include "engine.h"


// \! 核心接口
class ORTCORE {
public:
	typedef struct ORTCore_ctx ORTCore_ctx; // 执行上下文

	// \! 初始化
	std::shared_ptr<ORTCore_ctx> init(
		const Params& params,	// @param:params     初始化参数
		int& nErrnoFlag			// @param:nErrnoFlag 初始化错误码，详情见params.h
	);

	// \! 分类
	int classify(
		std::shared_ptr<ORTCore_ctx> ctx,
		const std::vector<cv::Mat> &cvImages, 
		std::vector<std::vector<ClassifyResult>>& outputs
	);

	// \! 异常检测
	int anomaly(
		std::shared_ptr<ORTCore_ctx> ctx, // 执行上下文
		const std::vector<cv::Mat> &cvImages,  // 输入图片
		std::vector<cv::Mat>& outputs		// 输出图片
	);

	// \! 分割
	int segment(
		std::shared_ptr<ORTCore_ctx> ctx,
		const std::vector<cv::Mat> &cvImages, 
		std::vector<cv::Mat>& outputs
	);
	// \! 目标检测网络
	int detect(
		std::shared_ptr<ORTCore_ctx> ctx,
		const std::vector<cv::Mat> &cvImages, 
		std::vector<std::vector<BBox>>& outputs
	);

	// \! 获得BatchSize, Channels, H, W
	int getInputDimsK(
		std::shared_ptr<ORTCore_ctx> ctx, 
		int& nBatch,
		int& nChannels,
		int& nHeight, 
		int &nWidth
	);

	// \! 获得BatchSize, H, W
	int getOutputDimsK(
		std::shared_ptr<ORTCore_ctx> ctx,
		int& nBatch,
		int& nHeight,
		int &nWidth
	);

	// \! 获得BatchSize, NumClass
	int getOutputDimsK(
		std::shared_ptr<ORTCore_ctx> ctx,
		int& nBatch,
		int &numClass
	);
	
};

