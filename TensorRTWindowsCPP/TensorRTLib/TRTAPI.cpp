#define TYPE_TRTAPI_API_EXPORTS
#include <windows.h>

#include "TRTAPI.h"
#include "trtCore.h"

// \! 构造函数
TRTAPI::TRTAPI()
{
	m_pTRTCore = nullptr;
}

// \! 析构函数
TRTAPI::~TRTAPI()
{
	if (m_pTRTCore != nullptr)
	{
		delete m_pTRTCore;
	}
}

// \! 初始化
// \@param:params     初始化参数
// \@param:nErrnoFlag 初始化错误码，详情见params.h
std::shared_ptr<TRTCore_ctx> TRTAPI::init(const Params& params, int& nErrnoFlag)
{
	// 1. 创建日志
	loguru::add_file(params.log_path.c_str(), loguru::Append, loguru::Verbosity_MAX);

	// 2. 初始化一个TRTCORE实例
	m_pTRTCore = new TRTCORE();	
	if (m_pTRTCore == nullptr) {
		LOG_F(INFO, "new TRTCORE() Error!!!!");
		return nullptr;
	}

	// 3. 初始化m_pTRTCore
	return m_pTRTCore->init(params, nErrnoFlag);
}

// \! 分类
// \@param ctx:执行上下文
// \@param vInCoreImages:输入图像列表，CoreImage格式
// \@param vvOutClsRes:输出结果，ClassifyResult格式
int TRTAPI::classify(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<CoreImage*>& vInCoreImages,std::vector<std::vector<ClassifyResult>>& vvOutClsRes)
{
	LOG_F(INFO, "classify start ......");

	// 1. 空指针检查
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		LOG_F(INFO, "Init failed, can't call classify");
		return LY_UNKNOW_ERROR;
	}
	
	// 2. 将CoreImage转成Opencv
	std::vector<cv::Mat> input_images;
	for (int i = 0; i < vInCoreImages.size(); i++) {
		cv::Mat cv_img = cv::Mat(
			vInCoreImages[i]->height_, 
			vInCoreImages[i]->width_, 
			CV_8UC(vInCoreImages[i]->channal_), 
			vInCoreImages[i]->imagedata_, 
			vInCoreImages[i]->imagestep_
		).clone();// 此处为什么要用clone？--》因为如不用clone，离开这个作用域，tmp量被释放，input_images中无数据了
		input_images.push_back(cv_img);
	}

	// 3.分类
	m_pTRTCore->classify(ctx, input_images, vvOutClsRes);

	return LY_OK;
}

// \! 分割
// \@param ctx: 执行上下文
// \@param vInCoreImages: 输入图片vector，CoreImage
// \@param vOutCoreImages:输出图片vector，CoreImage
int TRTAPI::segment(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*>& vInCoreImages, std::vector<CoreImage*>& vOutCoreImages, bool verbose)
{
	LOG_F(INFO, "segment start ......");

	// 1. 指针判断
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		LOG_F(INFO, "Init failed, can't call segment");
		return LY_UNKNOW_ERROR;
	}

	// 2. coreImage -> cvImage
	std::vector<cv::Mat> input_images, output_images;
	for (int i = 0; i < vInCoreImages.size(); i++) {
		cv::Mat cv_img = cv::Mat(vInCoreImages[i]->height_, vInCoreImages[i]->width_, CV_8UC(vInCoreImages[i]->channal_), vInCoreImages[i]->imagedata_, vInCoreImages[i]->imagestep_).clone();
		input_images.push_back(cv_img);
	}

	// 3. TRTCORE执行分割操作
	m_pTRTCore->segment(ctx, input_images, output_images, verbose);

	// 4. coreImage->opencv image
	int engine_output_batch, engine_output_height, engine_output_width;
	this->getOutputDims(ctx, engine_output_batch, engine_output_height, engine_output_width);//获得输出维度信息
	for (int n = 0; n < output_images.size(); n++)
	{
		output_images[n].convertTo(output_images[n], CV_8U);//onnx的输出是float32，而main函数中用8U定义
		memcpy_s(vOutCoreImages[n]->imagedata_, engine_output_width * engine_output_height, output_images[n].data, engine_output_width * engine_output_height);
		vOutCoreImages[n]->channal_ = 1;
		vOutCoreImages[n]->imagestep_ = engine_output_width;
		vOutCoreImages[n]->height_ = engine_output_height;
		vOutCoreImages[n]->width_ = engine_output_width;
	}
	return LY_OK;
}

// \! 目标检测
// \@param ctx:执行上下文
// \@param vInCoreImages:输入图片数组，CoreImage
// \@param vvOutBBoxs:输出结果数组，BBox
int TRTAPI::detect(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*>& vInCoreImages,std::vector<std::vector<BBox>>& vvOutBBoxs)
{
	LOG_F(INFO, "detect start ......");

	return LY_UNKNOW_ERROR;
}

// \! 异常检测
// \@param ctx:执行上下文
// \@param vInCoreImages:输入图片列表，CoreImage
// \@param vOutCoreImages:输出图片数组，CoreImage
// \@param threshold:阈值
// \@param maxValue:最大值，归一化时使用
// \@param minValue:最小值，归一化时使用
// \@param pixel_value:二值化图像的值，归一化时使用
int TRTAPI::anomaly(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<CoreImage*> &vInCoreImages,std::vector<CoreImage*>& vOutCoreImages, const float threshold, const float maxValue,	const float minValue,const int pixel_value) 
{
	LOG_F(INFO, "anomaly start ......");

	// 1.检查指针是否为空
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		LOG_F(INFO, "指针为空");
		return LY_UNKNOW_ERROR;
	}

	// 3.将CoreImage转成Opencv
	std::vector<cv::Mat> input_images, output_images;
	for (int i = 0; i < vInCoreImages.size(); i++) {
		cv::Mat cv_img = cv::Mat(
			vInCoreImages[i]->height_,
			vInCoreImages[i]->width_,
			CV_8UC(vInCoreImages[i]->channal_),
			vInCoreImages[i]->imagedata_,
			vInCoreImages[i]->imagestep_).clone();
		input_images.push_back(cv_img);
	}

	// 4.核心库推理
	m_pTRTCore->anomaly(ctx, input_images, output_images);

	// 5. opencv -> coreimage
	int input_batch, input_channels,input_height, input_width;
	m_pTRTCore->getInputDims(ctx, input_batch, input_channels, input_height, input_width);
	int output_batch, output_height, output_width;
	m_pTRTCore->getOutputDims(ctx, output_batch, output_height, output_width);
	for (int i = 0; i < output_images.size(); i++) {
		cv::resize(output_images[i], output_images[i], cv::Size(input_height, input_width), cv::INTER_LINEAR);
		output_images[i] = (output_images[i] - minValue) / (maxValue - minValue);
		cv::threshold(output_images[i], output_images[i], threshold, pixel_value, cv::THRESH_BINARY);
		output_images[i].convertTo(output_images[i], CV_8U);
		//cv::imwrite("E:/test/" + std::string(std::to_string(i)) + ".png", output_images[i]);
	}
	for (int n = 0; n < output_images.size(); n++)
	{
		memcpy_s(vOutCoreImages[n]->imagedata_, input_width * input_height, output_images[n].data, input_width * input_height);
		vOutCoreImages[n]->channal_ = 1;
		vOutCoreImages[n]->imagestep_ = input_width;
		vOutCoreImages[n]->height_ = input_height;
		vOutCoreImages[n]->width_ = input_width;
	}
	return LY_OK;
}

// \! 获取显卡数量
// \@param ctx:执行上下文
// \@param number:gpu数量
int TRTAPI::getNumberGPU(std::shared_ptr<TRTCore_ctx> ctx,int& number)
{
	return this->m_pTRTCore->getNumberGPU(ctx, number);
}

// \! 获取输入维度
// \@param ctx:执行上下文
// \@param nBatch:batchsize
// \@param nChannels:channels
// \@param nHeight:height
// \@param nWidth:width
// \@param index:第index个输入，加入onnx有多个输入，则通过index来指定
int TRTAPI::getInputDims(std::shared_ptr<TRTCore_ctx> ctx, int & nBatch, int & nChannels, int & nHeight, int & nWidth,	int index)
{
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		LOG_F(INFO, "init failed, can't call getDims");
		return LY_UNKNOW_ERROR;
	}
	m_pTRTCore->getInputDims(ctx, nBatch, nChannels, nHeight, nWidth, index);
	return LY_OK;
}

// \! 获取输出维度
// \@param ctx：执行上下文
// \@param nBatch:batchsize
// \@param nHeight:Height
// \@param nWidth:Width
// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
int TRTAPI::getOutputDims(std::shared_ptr<TRTCore_ctx> ctx,int& nBatch,int& nHeight,int &nWidth,int index)
{
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		LOG_F(INFO, "init failed, can't call getDims");
		return LY_UNKNOW_ERROR;
	}
	m_pTRTCore->getOutputDims(ctx, nBatch, nHeight, nWidth, index);
	return LY_OK;
}