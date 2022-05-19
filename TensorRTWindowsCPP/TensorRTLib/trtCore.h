/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : TensorRT 核心库
*****************************************************************************/
#pragma once

//#ifdef type_trt_core_api_exports
//#define type_trt_core_api __declspec(dllexport)
//#else
//#define type_trt_core_api __declspec(dllimport)
//#endif

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "params.h"


// \! API接口
//class TYPE_TRT_CORE_API TRTCORE {
class TRTCORE{
public:
	typedef struct TRTCore_ctx TRTCore_ctx; // 声明需要用到的一个结构体，包含（params、engine、pool）, 定义在engine.h中

	// @param:params     初始化参数
	// @param:nErrnoFlag 初始化错误码，详情见params.h
	std::shared_ptr<TRTCore_ctx> init(const Params& params, int& nErrnoFlag);

	// \! 分类
	int classify(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<cv::Mat> &cvImages, std::vector<std::vector<ClassifyResult>>& outputs);

	// \! 分割
	// if verbose=true,return the probability graph, else return the class id image
	int segment(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<cv::Mat> &cvImages, std::vector<cv::Mat>& outputs, bool verbose = false);

	// \! 目标检测网络
	int detect(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<cv::Mat> &cvImages, std::vector<std::vector<BBox>>& outputs);

	// \! 获得GPU数量
	int getDevices();

	// \! 获得BatchSize, Channels, H, W
	int getDims(std::shared_ptr<TRTCore_ctx> ctx, int& nBatch, int& nChannels, int& nHeight, int &nWidth);
};

