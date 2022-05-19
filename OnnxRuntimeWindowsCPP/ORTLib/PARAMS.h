#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <windows.h>

// \! 定义网络类型
enum NetWorkType
{
	FF_CLS = 0,  //!< 分类网络
	FF_SEG = 1,  //!< 分割网络
	FF_DET = 2,  //!< 检测网络
	FF_SIM = 3,  //!< 双流相似度网络
	FF_ANOMALY = 4,  // 异常检测网络.
};

// \! 定义返回码
enum ErrorCode {
	FF_OK=0,				// OK
	FF_ERROR_PNULL=1,		// 指针为空
	FF_ERROR_NETWORK=2,    // 网络类型错误
	FF_ERROR_INPUT=3,      // 输入格式错误
	FF_UNKNOW_ERROR=99,		// 未知错误，需要开发人员调试
};

// \! 定义初始化参数
struct Params
{
	NetWorkType netType = FF_ANOMALY;	            // 网络类型
	std::string onnxFilePath;	                    // onnx文件路径
	std::string log_path = "./";					// log日志保存路径，设置为空则不保存日志文件
	std::vector<float> stdValue{ 1.f, 1.f, 1.f };	// 归一化时用到的方差。superAI训练的模型不需要设置stdValue和meanValue，使用此默认值即可
	std::vector<float> meanValue{ 0.f, 0.0, 0.0 };	// 归一化时用到的均值。其他训练方式可能会需要此参数。
	int sessionThread = 1;							// 执行onnx 算子使用的线程数
};

// \! 分类返回结果
typedef std::pair<int, float> ClassifyResult;

// \! 检测返回结果
typedef struct BBox
{
	float x1, y1, x2, y2;     // 左上角、右下角坐标值
	float det_confidence;     // 检测框中包含目标的置信度
	float class_confidence;   // 检测框目标类别为第class_id类的置信度
	unsigned int class_id;    // 类别
}DetectResult;

// \! 分割返回结果
//typedef CoreImage SegementResult;

// \! 异常检测返回结果
//typedef CoreImage AnomalyResult;