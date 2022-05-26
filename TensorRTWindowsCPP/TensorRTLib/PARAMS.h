#pragma once
#include <iostream>
#include <string>
#include <vector>


// \! 定义网络类型
enum NetWorkType
{
	LUSTER_CLS = 0,  //!< 分类网络
	LUSTER_SEG = 1,  //!< 分割网络
	LUSTET_DET = 2,  //!< 检测网络
	LUSTER_SIM = 3,  //!< 双流相似度网络
	LUSTER_ANOMALY = 4,  //!< 异常检测网络
};

// \! 定义错误类型
enum ErrorCode {
	LY_OK = 0,               // 函数执行成功
	LY_WRONG_CALL = 1,       // 错误的调用，比如说分割模型调用了分类的接口
	LY_WRONG_FILE = 2,       // 文件找不到或者损坏
	LY_WRONG_IMG = 3,        // 输入图像格式错误或者不存在
	LY_WRONG_CUDA = 4,       // cuda指定错误或者显卡不存在
	LY_WRONG_TNAME = 5,      // caffe参数中inputTensorNames或者outputTensorNames设置错误
	LY_WRONG_PLAT = 6,       // 平台类型和模型文件不匹配
	LY_WRONG_BATCH = 7,      // onnx的batch size小于实际使用的
	LY_WRONG_GPUMEM = 8,       // 显存不足，无法申请显卡内存
	LY_UNKNOW_ERROR = 99,    // 未知错误，请联系开发者解决
};

// \! 定义初始化 参数
struct Params
{
	NetWorkType netType = LUSTER_CLS;	            // 网络类型
	std::string onnxFilePath;	                    // onnx文件路径
	std::string engineFilePath;	                    // 引擎文件保存位置，不存在则自动生成，存在则自动加载
	std::string log_path = "./";					// log日志保存路径，设置为空则不保存日志文件
	bool fp16{ false };	                            // 是否使用半精度 使用半精度生成引擎文件比较慢，使用的时候快
	std::vector<float> stdValue{ 1.f, 1.f, 1.f };	// 归一化时用到的方差。superAI训练的模型不需要设置stdValue和meanValue，使用此默认值即可
	std::vector<float> meanValue{ 0.f, 0.0, 0.0 };	// 归一化时用到的均值。其他训练方式可能会需要此参数。
	int gpuId{ 0 };		                            // GPU ID，0表示第一块显卡，1表示第2块显卡....以此类推。
	int maxThread = 8;				                // mContexts的数量，可以理解为一个模型支持同时启动的最多画面数。超过16的话请设置更大的数，否则不需要更改

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
// CoreImage

// \! 异常检测返回结果
// CoreImage