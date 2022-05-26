#include "engine.h"

// build和加载引擎的时候需要用到这个
//SampleErrorRecorder gRecorder;
namespace sample
{
	Logger gLogger{ Logger::Severity::kINFO };
	LogStreamConsumer gLogVerbose{ LOG_VERBOSE(gLogger) };
	LogStreamConsumer gLogInfo{ LOG_INFO(gLogger) };
	LogStreamConsumer gLogWarning{ LOG_WARN(gLogger) };
	LogStreamConsumer gLogError{ LOG_ERROR(gLogger) };
	LogStreamConsumer gLogFatal{ LOG_FATAL(gLogger) };

	void setReportableSeverity(Logger::Severity severity)
	{
		gLogger.setReportableSeverity(severity);
		gLogVerbose.setReportableSeverity(severity);
		gLogInfo.setReportableSeverity(severity);
		gLogWarning.setReportableSeverity(severity);
		gLogError.setReportableSeverity(severity);
		gLogFatal.setReportableSeverity(severity);
	}
} // namespace sample

// \!引擎构造
// \@param 传入Params参数
// \@param 错误参数nErrorFlag
TRTEngine::TRTEngine(const Params& params, int& nErrnoFlag) 
{
	// 1.初始化Engine：build或者load
	std::ifstream fin(params.engineFilePath);
	if (fin) {
		nErrnoFlag = loadEngine(params);  // 加载Engine文件
		if (nErrnoFlag != LY_OK)
		{
			LOG_F(INFO, "Loading Engine Failed");
			return;
		}
	}
	else {
		nErrnoFlag = buildONNX(params); // 构建Engine文件，并保存
		if (nErrnoFlag != LY_OK) {
			LOG_F(INFO, "Building ONNX Failed");
			return;
		}
	}
	
	// 2. 获得输入输出维度
	int num_input_output = mEngine->getNbBindings();
	for (int i = 0; i < num_input_output; i++) {
		if (mEngine->bindingIsInput(i)) {//如果是输入节点
			auto inputName_i = mEngine->getBindingName(i); // 获得i的输入节点名称
			mInputTensorNames.push_back(inputName_i);		// 将节点名称放入mInputTensorNames
			auto inputDims_i = mEngine->getBindingDimensions(i); // 获得节点i的维度
			mInputDims.push_back(inputDims_i);
		}
		else {//如果是输出节点
			auto outputName_i = mEngine->getBindingName(i); // 获得i的输出节点名称
			mOutputTensorNames.push_back(outputName_i);		// 将节点名称放入mOutputTensorNames
			auto outputDims_i = mEngine->getBindingDimensions(i); // 获得节点i的维度
			mOutputDims.push_back(outputDims_i);
		}
	}
	nErrnoFlag = LY_OK;
}

// \! 获取ICudaEngine指针
std::shared_ptr<nvinfer1::ICudaEngine> TRTEngine::Get() const
{
	return mEngine;
}

// \! 加载Engine文件
// \@param  params 参数结构体
int TRTEngine::loadEngine(const Params& params)
{
	std::fstream file;
	ICudaEngine* engine;
	file.open(params.engineFilePath, std::ios::binary | std::ios::in);
	if (!file.is_open())
	{
		LOG_F(INFO, (std::string("Can't load Engine file from: ") + params.engineFilePath).c_str());
		return LY_WRONG_FILE;
	}
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);
	file.close();
	SampleUniquePtr<IRuntime> runTime(createInferRuntime(sample::gLogger));
	if (runTime == nullptr) {
		LOG_F(INFO, "CreateInferRuntime error");
		return LY_UNKNOW_ERROR;
	}
	engine = runTime->deserializeCudaEngine(data.get(), length, nullptr);
	if (engine == nullptr) {
		LOG_F(INFO, "DeserializeCudaEngine error");
		return LY_UNKNOW_ERROR;
	}
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());
	return LY_OK;
}

// \! 构建和存储Engine
// \@param  params 参数结构体
int TRTEngine::buildONNX(const Params& params)
{
	LOG_F(INFO, "Building engine from onnx file, this may take few minutes, please wait ...");
	// 1. builder
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		LOG_F(INFO, "CreateInferBuilder error");
		return LY_UNKNOW_ERROR;
	}

	// 2. network
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		LOG_F(INFO, "CreateNetworkV2 error");
		return LY_UNKNOW_ERROR;
	}

	// 3. config
	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		LOG_F(INFO, "createBuilderConfig error");
		return LY_UNKNOW_ERROR;
	}
	config->setMaxWorkspaceSize(2048_MiB);
	if (params.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}

	// 4. parser
	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		LOG_F(INFO, "createParser error");
		return LY_UNKNOW_ERROR;
	}

	// 5. parsed
	auto parsed = parser->parseFromFile(params.onnxFilePath.c_str(),
		static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		LOG_F(INFO, "parse onnx File error");
		return LY_WRONG_FILE;
	}

	// 6. profileStream
	auto profileStream = samplesCommon::makeCudaStream();
	if (!profileStream)
	{
		LOG_F(INFO, "makeCudaStream error");
		return LY_UNKNOW_ERROR;
	}
	config->setProfileStream(*profileStream);

	// 7. plan
	SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan)
	{
		LOG_F(INFO, "builder->buildSerializedNetwork error");
		return LY_UNKNOW_ERROR;
	}

	// 8. runtime
	SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
	if (!runtime)
	{
		LOG_F(INFO, "createInferRuntime error");
		return LY_UNKNOW_ERROR;
	}

	// 9. icudaEngine
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
		runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		LOG_F(INFO, "deserializeCudaEngine error");
		return LY_UNKNOW_ERROR;
	}

	// 10.save
	SampleUniquePtr<IHostMemory> serializedModel(mEngine->serialize());
	std::ofstream p(params.engineFilePath.c_str(), std::ios::binary);
	p.write((const char*)serializedModel->data(), serializedModel->size());
	p.close();
	return LY_OK;
}