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

#include <iostream>
#include <string>
#include <vector>

#include "params.h"
#include "loguru.hpp" // https://github.com/emilk/loguru


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
	TRTAPI();	// 构造函数
	~TRTAPI();	// 析构函数
				
	// \! 初始化
	// \@param:params     初始化参数
	// \@param:nErrnoFlag 初始化错误码，详情见params.h
	std::shared_ptr<TRTCore_ctx> init(const Params& params, int& nErrnoFlag); 	

	// \! 分类
	// \@param ctx:执行上下文
	// \@param vInCoreImages:输入图像列表，CoreImage格式
	// \@param vvOutClsRes:输出结果，ClassifyResult格式
	int classify(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<CoreImage*> &vInCoreImages,	std::vector<std::vector<ClassifyResult>>& vvOutClsRes);

	// \! 分割
	// \@param ctx: 执行上下文
	// \@param vInCoreImages: 输入图片vector，CoreImage
	// \@param vOutCoreImages:输出图片vector，CoreImage
	// \@param verbose: 如果为true,return the probability graph, else return the class id image
	int segment(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*> &vInCoreImages, std::vector<CoreImage*>& vOutCoreImages,bool verbose=false);

	// \! 目标检测
	// \@param ctx:执行上下文
	// \@param vInCoreImages:输入图片数组，CoreImage
	// \@param vvOutBBoxs:输出结果数组，BBox
	int detect(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<CoreImage*> &vInCoreImages, std::vector<std::vector<BBox>>& vvOutBBoxs);

	// \! 异常检测
	// \@param ctx:执行上下文
	// \@param vInCoreImages:输入图片列表，CoreImage
	// \@param vOutCoreImages:输出图片数组，CoreImage
	// \@param threshold:阈值
	// \@param maxValue:最大值，归一化时使用
	// \@param minValue:最小值，归一化时使用
	// \@param pixel_value:二值化图像的值，归一化时使用
	int anomaly(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<CoreImage*> &vInCoreImages,std::vector<CoreImage*>& vOutCoreImages,const float threshold,const float maxValue= 27.811206817626953,const float minValue= 1.6174373626708984,const int pixel_value=255);
	
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
	int getInputDims(std::shared_ptr<TRTCore_ctx> ctx,int& nBatch, int& nChannels, int& nHeight, int& nWidth,int index=0);

	// \! 获取输出维度
	// \@param ctx：执行上下文
	// \@param nBatch:batchsize
	// \@param nHeight:Height
	// \@param nWidth:Width
	// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
	int getOutputDims(std::shared_ptr<TRTCore_ctx> ctx,int& nBatch,int& nHeight,int& nWidth,int index=0);

private:
	TRTCORE *m_pTRTCore; // 为了方便其他软件开发，定义了两层接口：开发层接口、应用层接口。应用层接口是在开发层之上封装的对外接口。
};
#endif