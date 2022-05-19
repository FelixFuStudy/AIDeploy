#define TYPE_TRTAPI_API_EXPORTS
#include <windows.h>
#include "TRTAPI.h"
#include "trtCore.h"


std::shared_ptr<TRTCore_ctx> TRTAPI::init(const Params& params, int& nErrnoFlag) {
	m_pTRTCore = new TRTCORE();
	if (m_pTRTCore == nullptr) {
		OutputDebugString("Init failed");
		return nullptr;
	}
	return m_pTRTCore->init(params, nErrnoFlag);
}

int TRTAPI::classify(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*>& vInCoreImages, std::vector<std::vector<ClassifyResult>>& vvOutClsRes)
{
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		OutputDebugString("Init failed, can't call classify");
		return LY_UNKNOW_ERROR;
	}
	// ½«CoreImage×ª³ÉOpencv
	std::vector<cv::Mat> input_images;
	for (int i = 0; i < vInCoreImages.size(); i++) {
		cv::Mat cv_img = cv::Mat(vInCoreImages[i]->height_, vInCoreImages[i]->width_, CV_8UC(vInCoreImages[i]->channal_), vInCoreImages[i]->imagedata_, vInCoreImages[i]->imagestep_).clone();
		input_images.push_back(cv_img);
	}
	m_pTRTCore->classify(ctx, input_images, vvOutClsRes);
	return LY_OK;
}

int TRTAPI::segment(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*>& vInCoreImages, std::vector<CoreImage*>& vOutCoreImages)
{
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		OutputDebugString("Init failed, can't call segment");
		return LY_UNKNOW_ERROR;
	}

	std::vector<cv::Mat> input_images, output_images;
	for (int i = 0; i < vInCoreImages.size(); i++) {
		cv::Mat cv_img = cv::Mat(vInCoreImages[i]->height_, vInCoreImages[i]->width_, CV_8UC(vInCoreImages[i]->channal_), vInCoreImages[i]->imagedata_, vInCoreImages[i]->imagestep_).clone();
		input_images.push_back(cv_img);
	}
	m_pTRTCore->segment(ctx, input_images, output_images);

	int height = input_images[0].rows;
	int width = input_images[0].cols;
	for (int n = 0; n < output_images.size(); n++)
	{
		cv::resize(output_images[n], output_images[n], cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
		memcpy_s(vOutCoreImages[n]->imagedata_, width * height, output_images[n].data, width * height);
		vOutCoreImages[n]->channal_ = 1;
		vOutCoreImages[n]->imagestep_ = width;
		vOutCoreImages[n]->height_ = height;
		vOutCoreImages[n]->width_ = width;
	}
	return LY_OK;
}

int TRTAPI::detect(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<CoreImage*>& vInCoreImages, std::vector<std::vector<BBox>>& vvOutBBoxs)
{
	return LY_UNKNOW_ERROR;
}

TRTAPI::TRTAPI()
{
	m_pTRTCore = nullptr;
}


TRTAPI::~TRTAPI()
{
	if (m_pTRTCore != nullptr)
	{
		delete m_pTRTCore;
	}
}

int TRTAPI::getDevices()
{
	return TRTCORE().getDevices();
}

int TRTAPI::getDims(std::shared_ptr<TRTCore_ctx> ctx, int & nBatch, int & nChannels, int & nHeight, int & nWidth)
{
	if (ctx == nullptr || m_pTRTCore == nullptr) {
		OutputDebugString("init failed, can't call getDims");
		return LY_UNKNOW_ERROR;
	}
	m_pTRTCore->getDims(ctx, nBatch, nChannels, nHeight, nWidth);
	return LY_OK;
}
