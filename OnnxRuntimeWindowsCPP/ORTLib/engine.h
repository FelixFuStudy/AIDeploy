/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : ONNXRuntime Session
*****************************************************************************/
#pragma once
#include <queue>	// 系统库
#include <opencv2/opencv.hpp> // opencv库
#include <onnxruntime_cxx_api.h>  //ONNXRuntime库
#include "params.h"	// 自定义的数据结构， 程序的接口


// \! ------------------------------------Session Start------------------------------
class ORTSession {
public:
	ORTSession() {};	// 默认构造函数
	ORTSession(const Params& params, int& nErrnoFlag);  // 引擎构造， 传入Params参数，和错误参数nErrorFlag
public:
	Ort::Session m_Session{ nullptr };	// Ort::Session 指针
	// 关于env和session的定义也是有个坑，不要定义在类的构造函数内部，
	//而是要定义成类成员，这样才能全局可用，否则infer()用到session时就会报错:
	Ort::Env m_Env{ ORT_LOGGING_LEVEL_WARNING, "OnnxSession" };	

	// \! 网络输入输出
	std::vector<std::vector<int64_t>> mInputDims;  // 模型输入维度
	std::vector<std::vector<int64_t>> mOutputDims;    // 模型输出维度
	std::vector<std::string> mOutputTensorNames;	// 模型输出的名称
	std::vector<std::string> mInputTensorNames; // 模型输入的名称

};
// \! ------------------------------------Session End------------------------------


// \! 执行上下文结构体
struct ORTCore_ctx
{
	Params params;							// \! 执行上下文的初始参数
	std::shared_ptr<ORTSession> session;	// \! 一个模型对应一个Session，一个Session执行上下文可以同时被多个线程调用
};
