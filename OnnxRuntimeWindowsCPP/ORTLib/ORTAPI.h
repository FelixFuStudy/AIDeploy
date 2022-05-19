/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : ONNXRuntime 库的API，在ORTCore上进一步分装，去除对opencv的依赖
*****************************************************************************/
#ifndef ORTAPI_H
#define ORTAPI_H

#if defined (TYPE_ORTAPI_API_EXPORTS)
#define TYPE_ORTAPI_API __declspec(dllexport)
#else
#define TYPE_ORTAPI_API __declspec(dllimport)
#endif

#include <iostream>
#include <string>
#include <vector>

#include "params.h"


// \! 图像的基础定义
class CoreImage
{
public:
	void SetValue(int channal, int width, int height, int step, unsigned char* data)
	{
		channal_ = channal;
		width_ = width;
		height_ = height;
		imagestep_ = step;
		imagedata_ = data;
	};

public:
	//图像通道
	int channal_;
	//宽	
	int width_;
	//高
	int height_;
	//每行字节数
	int imagestep_;
	//图像数据
	unsigned char *imagedata_;
};

// \! API接口
class ORTCORE;  // 核心层接口
class TYPE_ORTAPI_API ORTAPI { // API接口
public:
	typedef struct ORTCore_ctx ORTCore_ctx;   //声明需要用到的一个结构体，包含（params、session）
	ORTAPI();
	~ORTAPI();

	// \! 初始化
	std::shared_ptr<ORTCore_ctx> init(
		const Params& params,	// @param:params     初始化参数
		int& nErrnoFlag			// @param:nErrnoFlag 初始化错误码，详情见params.h
	); 	

	// \! 分类
	int classify(
		std::shared_ptr<ORTCore_ctx> ctx, 
		const std::vector<CoreImage*> &vInCoreImages, 
		std::vector<std::vector<ClassifyResult>>& vvOutClsRes
	);

	// \! 异常检测
	int anomaly(
		std::shared_ptr<ORTCore_ctx> ctx, 
		const std::vector<CoreImage*> &vInCoreImages, 
		std::vector<CoreImage*>& vOutCoreImages,
		float threshold,
		int pixel_value
	);
	
	// \! 分割
	int segment(
		std::shared_ptr<ORTCore_ctx> ctx, 
		const std::vector<CoreImage*> &vInCoreImages, 
		std::vector<CoreImage*>& vOutCoreImages
	);

	// \! 目标检测网络
	int detect(
		std::shared_ptr<ORTCore_ctx> ctx, 
		const std::vector<CoreImage*> &vInCoreImages, 
		std::vector<std::vector<BBox>>& vvOutBBoxs
	);

	// \! 获得InputDims
	int getInputDimsK(
		std::shared_ptr<ORTCore_ctx> ctx, 
		int& nBatch,
		int& nChannels,
		int& nHeight, 
		int &nWidth
	);

	// \! 获得OutputDims
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

private:
	ORTCORE *m_pORTCore; // 为了方便其他软件开发，定义了两层接口：开发层接口、应用层接口。应用层接口是在开发层之上封装的对外接口。
};
#endif