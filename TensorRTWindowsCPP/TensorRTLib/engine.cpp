#include "engine.h"
#include "my_logger.h"


TRTEngine::TRTEngine(const Params& params, int& nErrnoFlag) {
	// ³õÊ¼»¯Engine
	std::ifstream fin(params.engineFilePath); // judge whether exist engine file
	if (fin) {
		nErrnoFlag = loadEngine(params);
		if (nErrnoFlag != LY_OK)
		{
			my_logger::log(params.log_path, std::string("Loading Engine Failed"));
			OutputDebugString("Loading Engine Failed");
			return;
		}
	}
	else {
		nErrnoFlag = buildONNX(params);
		if (nErrnoFlag != LY_OK) {
			OutputDebugString("Building ONNX Failed");
			my_logger::log(params.log_path, std::string("Building ONNX Failed"));
			return;
		}
	}
	// get some infomation of dims 
	mInputDims = mEngine->getBindingDimensions(0); // input: n*c*h*w 
	mOutputDims = mEngine->getBindingDimensions(1); // output: n*c*h*w 
	mInputTensorNames = { mEngine->getBindingName(0) };  //.get().getBindingName(0);
	mOutputTensorNames = { mEngine->getBindingName(1) };
	nErrnoFlag = LY_OK;
}

std::shared_ptr<nvinfer1::ICudaEngine> TRTEngine::Get() const
{
	return mEngine;
}

int TRTEngine::loadEngine(const Params& params)
{
	my_logger::log(params.log_path, std::string("Loading Engine from ") + params.engineFilePath);
	std::fstream file;
	ICudaEngine* engine;
	file.open(params.engineFilePath, std::ios::binary | std::ios::in);
	if (!file.is_open())
	{
		OutputDebugString((std::string("Can't load Engine file from: ") + params.engineFilePath).c_str());
		my_logger::log(params.log_path, std::string("Can't load Engine file from: ") + params.engineFilePath);
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
		OutputDebugString("CreateInferRuntime error");
		my_logger::log(params.log_path, "CreateInferRuntime error");
		return LY_UNKNOW_ERROR;
	}
	engine = runTime->deserializeCudaEngine(data.get(), length, nullptr);
	if (engine == nullptr) {
		OutputDebugString("DeserializeCudaEngine error");
		my_logger::log(params.log_path, "DeserializeCudaEngine error");
		return LY_UNKNOW_ERROR;
	}
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());
	return LY_OK;
}

// build and save onnx engine
int TRTEngine::buildONNX(const Params& params)
{
	my_logger::log(params.log_path, std::string("Building engine from onnx file, this may take few minutes, please wait ..."));
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		my_logger::log(params.log_path, "CreateInferBuilder error");
		OutputDebugString("CreateInferBuilder error");
		return LY_UNKNOW_ERROR;
	}
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		my_logger::log(params.log_path, "CreateNetworkV2 error");
		OutputDebugString("CreateNetworkV2 error");
		return LY_UNKNOW_ERROR;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		my_logger::log(params.log_path, "createBuilderConfig error");
		OutputDebugString("createBuilderConfig error");
		return LY_UNKNOW_ERROR;
	}
	config->setMaxWorkspaceSize(2048_MiB);
	if (params.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		my_logger::log(params.log_path, "createParser error");
		OutputDebugString("createParser error");
		return LY_UNKNOW_ERROR;
	}

	auto parsed = parser->parseFromFile(params.onnxFilePath.c_str(),
		static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		my_logger::log(params.log_path, "parse onnx File error");
		OutputDebugString("parse onnx File error");
		return LY_WRONG_FILE;
	}

	auto profileStream = samplesCommon::makeCudaStream();
	if (!profileStream)
	{
		my_logger::log(params.log_path, "makeCudaStream error");
		OutputDebugString("makeCudaStream error");
		return LY_UNKNOW_ERROR;
	}
	config->setProfileStream(*profileStream);

	SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan)
	{
		my_logger::log(params.log_path, "builder->buildSerializedNetwork error");
		OutputDebugString("builder->buildSerializedNetwork error");
		return LY_UNKNOW_ERROR;
	}

	SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
	if (!runtime)
	{
		my_logger::log(params.log_path, "createInferRuntime error");
		OutputDebugString("createInferRuntime error");
		return LY_UNKNOW_ERROR;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
		runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		my_logger::log(params.log_path, "deserializeCudaEngine error");
		OutputDebugString("deserializeCudaEngine error");
		return LY_UNKNOW_ERROR;
	}
	//save
	SampleUniquePtr<IHostMemory> serializedModel(mEngine->serialize());
	std::ofstream p(params.engineFilePath.c_str(), std::ios::binary);
	p.write((const char*)serializedModel->data(), serializedModel->size());
	p.close();
	return LY_OK;
}