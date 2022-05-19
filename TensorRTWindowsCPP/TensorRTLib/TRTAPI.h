/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : TRTCORE 库的API，在TRTCORE上进一步分装，去除对opencv的依赖
*****************************************************************************/
#ifndef TRTAPI_H
#define TRTAPI_H

#if defined (TYPE_TRTAPI_API_EXPORTS)
#define TYPE_TRTAPI_API __declspec(dllexport)
#else
#define TYPE_TRTAPI_API __declspec(dllimport)
#endif

//#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "params.h"


// \! ---------定义接口 输入、输出格式 Start------------
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
class TRTCORE;  // 开发层接口
class TYPE_TRTAPI_API TRTAPI { // 应用层接口
public:
	typedef struct TRTCore_ctx TRTCore_ctx;   //声明需要用到的一个结构体，包含（params、engine、pool）
	TRTAPI();

	~TRTAPI();
	// @param:params     初始化参数
	// @param:nErrnoFlag 初始化错误码，详情见params.h
	std::shared_ptr<TRTCore_ctx> init(const Params& params, int& nErrnoFlag); 	// \! 初始化

	// \! 分类
	int classify(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*> &vInCoreImages, std::vector<std::vector<ClassifyResult>>& vvOutClsRes);

	// \! 分割
	int segment(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*> &vInCoreImages, std::vector<CoreImage*>& vOutCoreImages);

	// \! 目标检测网络
	int detect(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*> &vInCoreImages, std::vector<std::vector<BBox>>& vvOutBBoxs);

	// 获取显卡数量
	int getDevices();

	int getDims(std::shared_ptr<TRTCore_ctx> ctx, int& nBatch, int& nChannels, int& nHeight, int &nWidth);

private:
	TRTCORE *m_pTRTCore; // 为了方便其他软件开发，定义了两层接口：开发层接口、应用层接口。应用层接口是在开发层之上封装的对外接口。
};
#endif