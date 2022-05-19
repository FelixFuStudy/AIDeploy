#include <iostream>
#include <string>

#include "engine.h"
#include "F_log.h"

ORTSession::ORTSession(const Params& params, int& nErrnoFlag) {
	// 初始化Engine
	try {
		// 1. 初始化日志日志文件
		YLog ortLog(YLog::INFO, params.log_path, YLog::ADD);
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "初始化ONNX Engine");

		// 2. 设置Env
		// initialize  enviroment...one enviroment per process
		// enviroment maintains thread pools and other state info
		//environment （设置为VERBOSE（ORT_LOGGING_LEVEL_VERBOSE）时，方便控制台输出时看到是使用了cpu还是gpu执行）
		//Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");

		// 3. 设置SessionOptions
		// initialize session options if needed
		// 使用1个线程执行op,若想提升速度，增加线程数
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(params.sessionThread);
		//	CUDA加速开启(由于onnxruntime的版本太高，
		//  无cuda_provider_factory.h的头文件，加速可以使用onnxruntime V1.8的版本)
		//  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
		// If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
		// session (we also need to include cuda_provider_factory.h above which defines it)
		// #include "cuda_provider_factory.h"
		// OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

		//  ORT_ENABLE_ALL: 启用所有可能的优化
		// Sets graph optimization level
		// Available levels are
		// ORT_DISABLE_ALL -> To disable all optimizations
		// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
		// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
		// ORT_ENABLE_ALL -> To Enable All possible opitmizations
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
		const std::string text = params.onnxFilePath;
		std::wstring szDst;
		{
			int len = MultiByteToWideChar(CP_ACP, 0, (LPCSTR)text.c_str(), -1, NULL, 0);
			wchar_t *wszUtf8 = new wchar_t[len + 1];
			memset(wszUtf8, 0, len * 2 + 2);
			MultiByteToWideChar(CP_ACP, 0, (LPCSTR)text.c_str(), -1, (LPWSTR)wszUtf8, len);
			szDst = wszUtf8;
			delete[]wszUtf8;
		}
		const wchar_t* model_path = szDst.c_str();
#else
		const std::string text = params.onnxFilePath;
		std::wstring szDst;
		{
			int len = MultiByteToWideChar(CP_ACP, 0, (LPCSTR)text.c_str(), -1, NULL, 0);
			wchar_t *wszUtf8 = new wchar_t[len + 1];
			memset(wszUtf8, 0, len * 2 + 2);
			MultiByteToWideChar(CP_ACP, 0, (LPCSTR)text.c_str(), -1, (LPWSTR)wszUtf8, len);
			szDst = wszUtf8;
			delete[]wszUtf8;
		}
		const char_t* model_path = szDst.c_str();
#endif

		// 4. 生成Ort::Session类
		// create session and load model into memory
		// using squeezenet version 1.3
		// URL = https://github.com/onnx/models/tree/master/squeezenet
		m_Session = Ort::Session(m_Env, model_path, session_options);
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "初始化ONNX Session成功");

		// 5. 打印ONNX模型信息
		// 5.1打印输入信息
		//print model input layer (node names, types, shape etc.)
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "--ONNX 模型信息--");
		Ort::AllocatorWithDefaultOptions allocator;
		// print number of model input nodes
		size_t num_input_nodes = m_Session.GetInputCount();
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "输入个数:"+std::to_string(num_input_nodes));
		// iterate over all input nodes
		for (int i = 0; i < num_input_nodes; i++) {
			// print input node names
			char* input_name = m_Session.GetInputName(i, allocator);
			mInputTensorNames.push_back(input_name);
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Input " + std::to_string(i) + "_Name:"+input_name);

			// print input node types
			Ort::TypeInfo type_info = m_Session.GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			ONNXTensorElementDataType type = tensor_info.GetElementType();
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Input " + std::to_string(i) + "_Type:" + std::to_string(type));

			// print input shapes/dims
			std::vector<int64_t> input_node_dims;
			input_node_dims = tensor_info.GetShape();
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Input " + std::to_string(i) + "_SizeNum:" + std::to_string(input_node_dims.size()));
			for (int j = 0; j < input_node_dims.size(); j++)
				ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Input " + std::to_string(i) + "_"+
					std::to_string(j)+":" + std::to_string(input_node_dims[j]));
			mInputDims.push_back(input_node_dims);
		}

		// 5.2打印输出信息
		//print number of model output nodes
		size_t num_output_nodes = m_Session.GetOutputCount();
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "输出个数:" + std::to_string(num_output_nodes));
		// iterate over all outpu nodes
		for (int i = 0; i < num_output_nodes; i++) {
			// print output node names
			char* output_name = m_Session.GetOutputName(i, allocator);
			mOutputTensorNames.push_back(output_name);
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Output " + std::to_string(i) + "_Name:" + output_name);

			// print output node types
			Ort::TypeInfo type_info = m_Session.GetOutputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			ONNXTensorElementDataType type = tensor_info.GetElementType();
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Output " + std::to_string(i) + "_Type:" + std::to_string(type));

			// print output shapes/dims
			std::vector<int64_t> output_node_dims;
			output_node_dims = tensor_info.GetShape();
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Output " + std::to_string(i) + "_SizeNum:" + std::to_string(output_node_dims.size()));
			for (int j = 0; j < output_node_dims.size(); j++)
				ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "Output " + std::to_string(i) + "_" +
					std::to_string(j) + ":" + std::to_string(output_node_dims[j]));
			mOutputDims.push_back(output_node_dims);
		}
	}
	catch (const char* msg) {
		YLog ortLog(YLog::ERR, params.log_path, YLog::ADD);
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTSession", "初始化ONNX Engine失败");
		nErrnoFlag = FF_UNKNOW_ERROR;
	}
	nErrnoFlag = FF_OK;
	

	
}

