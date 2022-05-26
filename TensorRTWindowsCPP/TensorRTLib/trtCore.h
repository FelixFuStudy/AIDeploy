/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : TensorRT 核心库
*****************************************************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "loguru.hpp" // https://github.com/emilk/loguru
#include "params.h"


// \! TRTCORE接口
class TRTCORE{
public:
	typedef struct TRTCore_ctx TRTCore_ctx; // 声明需要用到的一个结构体，包含（params、engine、pool）, 定义在engine.h中

	// \! 初始化
	// \@param params     初始化参数
	// \@param nErrnoFlag 初始化错误码，详情见params.h
	std::shared_ptr<TRTCore_ctx> init(const Params& params, int& nErrnoFlag);

	// \! 分类
	// \@param ctx:执行上下文
	// \@param vInCoreImages:输入图像列表，Mat格式
	// \@param vvOutClsRes:输出结果，ClassifyResult格式
	int classify(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<cv::Mat> &cvImages,std::vector<std::vector<ClassifyResult>>& outputs);

	// \! 分割
	// \@param ctx: 执行上下文
	// \@param vInCoreImages: 输入图片vector，cvImage
	// \@param vOutCoreImages:输出图片vector，cvImage
	// \@param verbose: 如果为true,return the probability graph, else return the class id image
	int segment(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<cv::Mat> &cvImages, std::vector<cv::Mat>& outputs, bool verbose = false);

	// \! 目标检测
	int detect(
		std::shared_ptr<TRTCore_ctx> ctx, 
		const std::vector<cv::Mat> &cvImages, 
		std::vector<std::vector<BBox>>& outputs
	);

	// \! 异常检测
	int anomaly(
		std::shared_ptr<TRTCore_ctx> ctx,
		const std::vector<cv::Mat> &cvImages,
		std::vector<cv::Mat>& outputs
	);

	// \! 获取显卡数量
	// \@param ctx:执行上下文
	// \@param number:gpu数量
	int getNumberGPU(std::shared_ptr<TRTCore_ctx> ctx,int& number);

	// \! 获取输入维度
	// \@param ctx:执行上下文
	// \@param nBatch:batchsize
	// \@param nChannels:channels
	// \@param nHeight:height
	// \@param nWidth:width
	// \@param index:第index个输入，加入onnx有多个输入，则通过index来指定
	int getInputDims(std::shared_ptr<TRTCore_ctx> ctx,int & nBatch,	int & nChannels,int & nHeight,int & nWidth,int index=0);

	// \! 获取输出维度
	// \@param ctx：执行上下文
	// \@param nBatch:batchsize
	// \@param nHeight:Height
	// \@param nWidth:Width
	// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
	int getOutputDims(std::shared_ptr<TRTCore_ctx> ctx,	int& nBatch,int& nHeight,int &nWidth,int index=0);
	
	// \! 获取输出维度
	// \@param ctx：执行上下文
	// \@param nBatch:batchsize
	// \@param nNumClass:NumClass 类别数，针对分类
	// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
	int getOutputDims2(std::shared_ptr<TRTCore_ctx> ctx,int & nBatch,int & nNumClass,int index = 0);

	// \! 获取输入名称
	// \@param ctx：执行上下文
	// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
	std::string getInputNames(std::shared_ptr<TRTCore_ctx> ctx,int index = 0);

	// \! 获取输出名称
	// \@param ctx：执行上下文
	// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
	std::string getOutputNames(std::shared_ptr<TRTCore_ctx> ctx,int index = 0);
};

