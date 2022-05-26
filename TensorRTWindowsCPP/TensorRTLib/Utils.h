#pragma once
#include <cuda_runtime.h>  // cuda库
#include <cuda_runtime_api.h>
#include "loguru.hpp" // https://github.com/emilk/loguru

// \! 指定device id的GPU是否兼容
bool IsCompatible(int device)
{
	cudaError_t st = cudaSetDevice(device);
	if (st != cudaSuccess)
		return false;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	if (deviceProp.major < 3)
		return false;

	return true;
}

// \! 前处理 ：分类、分割、异常检测、目标检测的标准化、归一化处理
// \! （1）8U->32F;(2)减均值，除方差;(3)如果是3通道的，BGR->RGB;(4)转成一维数组
// \@param buffers: 缓存，tensorRT中的samples定义
// \@param cv_images: 输入图片，openCV格式
// \@param params: 参数结构体
// \@param inputName: onnx的输入名称
int normalization(const samplesCommon::BufferManager & buffers,std::vector<cv::Mat>& cv_images, const Params& params, std::string inputName)
{
	// 1.分配host内存
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(inputName));
	int nums = cv_images.size();	// 获得图片数量-batchsize
	int depth = cv_images[0].channels();	// 获得channels
	int height = cv_images[0].rows;	// 获得height
	int width = cv_images[0].cols;	// 获得width

	if (depth == 1) {// 通道数为1
		for (int n = 0, index = 0; n < nums; n++)
		{
			cv_images[n].convertTo(cv_images[n], CV_32F, 1.0 / 255);
			cv_images[n] = (cv_images[n] - params.meanValue[0]) / params.stdValue[0];
			memcpy_s(hostDataBuffer + n * height * width, height * width * sizeof(float), cv_images[n].data, height * width * sizeof(float));
		}
	}
	else if (depth == 3) {// 通道数为3
		for (int n = 0, index = 0; n < nums; n++)
		{
			cv_images[n].convertTo(cv_images[n], CV_32F, 1.0 / 255);
			std::vector<cv::Mat> bgrChannels(3);
			cv::split(cv_images[n], bgrChannels);
			for (int d = 0; d < 3; d++) {
				bgrChannels[2 - d] = (bgrChannels[2 - d] - params.meanValue[d]) / params.stdValue[d];	// 当前图像通道是RGB，转成BGR
				memcpy_s(hostDataBuffer + height * width * (3 * n + (2 - d)), height * width * sizeof(float), bgrChannels[2 - d].data, height * width * sizeof(float));
			}
		}
	}
	else {
		LOG_F(INFO, "不支持的图像类型");
		return LY_WRONG_IMG;
	}
	return LY_OK;
}

// \! 异常检测后处理
// \@param buffers: TensorRT samples中定义的缓存
// \@param out_masks: 存储输出mask的
// \@param engine_output_batch: ONNX模型的batch
// \@param engine_output_height: ONNX模型的height
// \@param engine_output_width: ONNX模型的width
// \@param engine_output_name: ONNX模型的输出名称
int anomalyPost(const samplesCommon::BufferManager & buffers, std::vector<cv::Mat>& out_masks, const int engine_output_batchsize,const int engine_output_height,const int engine_output_width,const std::string outputName)
{
	// 拷贝数据
	float* output = static_cast<float*>(buffers.getHostBuffer(outputName));
	// 生成out_mask
	for (int i = 0; i < engine_output_batchsize; i++){
			cv::Mat tmp = cv::Mat(
				engine_output_height,
				engine_output_width, 
				CV_32F, 
				output + engine_output_height * engine_output_width * i
			).clone();
			out_masks.push_back(tmp);
	}
	return LY_OK;
}

// \! 比较大小
bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
	return lhs.first > rhs.first;
}
// \！分类任务中返回最大的N类得分类别,输入大小是类别数目c个，输出大小N
std::vector<int> Argmax(const std::vector<float>& v, int N)
{
	std::vector<std::pair<float, int>> pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));

	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}
// \!分类后处理
// \@ param buffers:tensorRT sample定义的缓存
// \@ param outputs:输出结果
// \@ param batch:batchsize
// \@ param num_class:类别数量
// \@ param output_name:onnx输出名称
int clsPost(const samplesCommon::BufferManager & buffers,std::vector<std::vector<ClassifyResult>>& outputs, const int batch, const int num_class,std::string output_name)
{
	float* output_buffer = static_cast<float*>(buffers.getHostBuffer(output_name));

	// Top K
	int N = 3;
	auto K = N > num_class ? num_class : N;

	for (int b = 0; b < batch; b++) {
		float sum{ 0.0f };
		for (int i = 0; i < num_class; i++) {
			output_buffer[b * num_class + i] = exp(output_buffer[b * num_class + i]);
			sum += output_buffer[b * num_class + i];
		}

		// output存放一张图片的所有类别的置信度
		std::vector<float> output;
		for (int j = 0; j < num_class; j++) {
			output.push_back(output_buffer[b * num_class + j] / sum);
		}

		// output topk的index 放入maxN中
		std::vector<int> maxN = Argmax(output, K);
		std::vector<ClassifyResult> classifyResults;

		for (int i = 0; i < K; ++i)
		{
			int idx = maxN[i];
			classifyResults.push_back(std::make_pair(idx, output[idx]));
		}

		outputs.push_back(classifyResults);
	}
	return LY_OK;
}

// \! 分割后处理
// \@param buffers: TensorRT samples中定义的缓存
// \@param out_masks: 存储输出mask的
// \@param engine_output_batch: ONNX模型的batch
// \@param engine_output_height: ONNX模型的height
// \@param engine_output_width: ONNX模型的width
// \@param engine_output_name: ONNX模型的输出名称
// \@param verbose: 如果为true,return the probability graph, else return the class id image
int segPost(const samplesCommon::BufferManager & buffers, std::vector<cv::Mat>& out_masks, const int engine_output_batchsize,const int engine_output_height,const int engine_output_width,const std::string outputName,bool verbose=false)
{
	float* output = static_cast<float*>(buffers.getHostBuffer(outputName));
	if (verbose) {
		// TODO
		return LY_OK;
	}
	else {
		// 生成out_mask
		for (int i = 0; i < engine_output_batchsize; i++) {
			cv::Mat tmp = cv::Mat(
				engine_output_height,
				engine_output_width,
				CV_32F,
				output + engine_output_height * engine_output_width * i
			).clone();
			out_masks.push_back(tmp);
		}
	}
	return LY_OK;
}
