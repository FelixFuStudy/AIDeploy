#include "ortCore.h"
#include "engine.h"
#include "F_log.h"
#include "utils.h"
using namespace std;

// \! 初始化ORTCore_ctx执行上下文
std::shared_ptr<ORTCore_ctx> ORTCORE::init(
	const Params& params, 
	int& nErrnoFlag
) {
	// 1. 初始化日志日志文件
	YLog ortLog(YLog::INFO, params.log_path, YLog::ADD);
	try {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Current version: 2022-05-17");

		// 1.使用ONNX模型文件生成ORTSession类
		std::shared_ptr <ORTSession> ortSession(new ORTSession(params, nErrnoFlag));
		if (nErrnoFlag != FF_OK) {
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Can't load onnx file");
			return nullptr;
		}

		// 2. 通过ortSession生成ORTCore_ctx上下文
		std::shared_ptr<ORTCore_ctx> ctx(new ORTCore_ctx{ params, ortSession });
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Init Successfully !!!!");
		return ctx;
	}
	catch (const std::invalid_argument& ex) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Init failed !!!!");
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", std::string(ex.what()));
		nErrnoFlag = FF_UNKNOW_ERROR;
		return nullptr;
	}
}


// \! 分类
int ORTCORE::classify(
	std::shared_ptr<ORTCore_ctx> ctx, 
	const std::vector<cv::Mat> &cvImages, 
	std::vector<std::vector<ClassifyResult>>& outputs
)
{
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);
	// 1. ctx是否初始化成功
	if (ctx == nullptr) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Classification", "Init Failed，Can't call classify function !");
		return FF_ERROR_PNULL;
	}

	// 2. 判断NetType是否正确
	if (ctx.get()->params.netType != FF_CLS)
	{
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Classification", "Illegal calls，please check your NetWorkType");
		return FF_ERROR_NETWORK;
	}

	// 3. session信息与输入信息对比
	std::vector<int64_t> inputDims = ctx.get()->session->mInputDims[0];		// 第一个输入维度（session）
	std::vector<int64_t> outputDims = ctx.get()->session->mOutputDims[0];	// 第一个输出维度（session）
	auto input_name = ctx.get()->session->mInputTensorNames[0];				// 第一个输入名称
	auto output_name = ctx.get()->session->mOutputTensorNames[0];			// 第一个输出名称
	// 获得session信息, input和output
	int session_batch{ 0 }, session_channels{ 0 }, session_height{ 0 }, session_width{ 0 };
	int session_numClass{ 0 };
	session_batch = inputDims[0];	// 获得batchsize（session）
	session_channels = inputDims[1];// 获得channels（session）
	session_height = inputDims[2];// 获得height（session）
	session_width = inputDims[3];// 获得width（session）
	session_numClass = outputDims[1]; // 获得number class
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	output_node_names.push_back(ctx.get()->session->mOutputTensorNames[0].c_str());	// 获得输出名称
	input_node_names.push_back(ctx.get()->session->mInputTensorNames[0].c_str());	// 获得输入名称
	// session信息与输入信息对比
	std::vector<cv::Mat> imgs;	// 存放resize后的图片，在判断时，存储在imgs中
	// 3.1 判断batchsize
	if (cvImages.size() != session_batch) {// batchsize判断
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Classification", "输入图片数组的batchsize与ONNX batchsize不同 ");
		return FF_ERROR_INPUT;
	}
	// 3.2 判断channels和HW
	for (int i = 0; i < cvImages.size(); i++) {// c,h,w判断
		cv::Mat cv_img = cvImages[i].clone();
		if (cv_img.channels() != session_channels) {
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "Classification", "输入图片的通道数与session不符 ");
			return FF_ERROR_INPUT;
		}
		if (session_height != cv_img.cols || session_width != cv_img.rows) {
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "Classification", "输入的图片尺寸与session不相符,自动resize");
			cv::resize(cv_img, cv_img, cv::Size(session_height, session_width), 0, 0, cv::INTER_LINEAR);
		}

		imgs.push_back(cv_img);
	}
	if (imgs.empty()) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Classification", "No images, please check");
		return FF_ERROR_INPUT;
	}
	
	// 4. 预处理图片
	size_t input_tensor_size = session_batch * session_channels * session_height * session_width;  // 获得图片尺寸
	std::vector<float> input_tensor_values(input_tensor_size);	// 存放最终图片
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);// create input tensor object from data values
	normalization(imgs, input_tensor_values, ctx, memory_info);// 归一化图片， 将结果存储在input_tensor_values
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(// 通过input_tensor_values生成input_tensor
		memory_info,
		input_tensor_values.data(),
		input_tensor_size,
		ctx.get()->session->mInputDims[0].data(),
		ctx.get()->session.get()->m_Session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().size());
	assert(input_tensor.IsTensor());	// 判断input_tensor是否是Tensor
	
	// 5. ort_inputs输入
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	// 6. infer推理过程
	auto output_tensors = ctx.get()->session.get()->m_Session.Run(
		Ort::RunOptions{ nullptr },
		input_node_names.data(),
		ort_inputs.data(),
		ort_inputs.size(),
		output_node_names.data(),
		ctx.get()->session.get()->mOutputTensorNames.size());
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	// 7. 后处理
	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	clsPost(floatarr, outputs, session_batch, session_numClass);
	return FF_OK;
}


// \!异常检测
int ORTCORE::anomaly(
	std::shared_ptr<ORTCore_ctx> ctx, // 执行上下文
	const std::vector<cv::Mat> &cvImages, // 输入图片
	std::vector<cv::Mat>& outputs// 输出图片
) 
{
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);//日志对象
	// 1. ctx是否初始化成功
	if (ctx == nullptr) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "anomaly", "ctx有问题");
		return FF_ERROR_PNULL;
	}

	// 2. 判断NetType是否正确
	if (ctx.get()->params.netType != FF_ANOMALY)
	{
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Anomaly", "Illegal calls，please check your NetWorkType");
		return FF_ERROR_NETWORK;
	}

	// 3. session信息与输入信息对比
	std::vector<int64_t> inputDims = ctx.get()->session->mInputDims[0];		// 获得onnx中第一个输入维度（session）
	std::vector<int64_t> outputDims = ctx.get()->session->mOutputDims[0];	// 获得onnx中第一个输出维度（session）
	auto input_name = ctx.get()->session->mInputTensorNames[0];				// 获得onnx中第一个输入名称
	auto output_name = ctx.get()->session->mOutputTensorNames[0];			// 获得onnx中第一个输出名称
	// 3.1 获得输入维度
	int session_batch{ 0 }, session_channels{ 0 }, session_height{ 0 }, session_width{ 0 };
	session_batch = inputDims[0];	// 获得batchsize（session）
	session_channels = inputDims[1];// 获得channels（session）
	session_height = inputDims[2];// 获得height（session）
	session_width = inputDims[3];// 获得width（session）
	// 3.2 获得input和output名称
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	output_node_names.push_back(output_name.c_str());	// 获得输出名称
	input_node_names.push_back(input_name.c_str());	// 获得输入名称
	// 3.3 onnx的batchsize内容与输入内容的batchsize对比
	std::vector<cv::Mat> imgs;	// 存放resize后的图片，在判断时，存储在imgs中
	if (cvImages.size() != session_batch) {// batchsize判断
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Anomaly", "输入图片数组的batchsize与ONNX batchsize不同 ");
		return FF_ERROR_INPUT;
	}
	// 3.4 onnx的c,h,w与输入的图片对比
	for (int i = 0; i < cvImages.size(); i++) {// c,h,w判断
		cv::Mat cv_img = cvImages[i].clone();
		if (cv_img.channels() != session_channels) {// 通道数判断
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "Anomaly", "输入图片的通道数与session不符 ");
			return FF_ERROR_INPUT;
		}
		if (session_height != cv_img.cols || session_width != cv_img.rows) {// h，w判断
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "Anomaly", "输入的图片尺寸与session不相符,自动resize");
			//cv::resize(cv_img, cv_img, cv::Size(session_height, session_width), 0.5, 0.5, cv::INTER_AREA); // 使用此方法与onnx（python）时像对应
			cv::resize(cv_img, cv_img, cv::Size(session_height, session_width), cv::INTER_LINEAR); // 使用此方法与onnx（python）时像对应
		}
		imgs.push_back(cv_img);
	}
	// 3.5 判断图片是否为空
	if (imgs.empty()) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Anomaly", "No images, please check");
		return FF_ERROR_INPUT;
	}

	// 4. 预处理图片
	size_t input_tensor_size = session_batch * session_channels * session_height * session_width;  // 获得图片尺寸
	std::vector<float> input_tensor_values(input_tensor_size);	// 存放最终图片
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);// create input tensor object from data values
	normalization(imgs, input_tensor_values, ctx, memory_info);// 归一化图片， 将结果存储在input_tensor_values
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(// 通过input_tensor_values生成input_tensor
		memory_info,// ort内存类型
		input_tensor_values.data(),// 输入数据
		input_tensor_size,// 数据大小
		ctx.get()->session->mInputDims[0].data(),// onnx输入的shape
		ctx.get()->session.get()->m_Session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().size());
	assert(input_tensor.IsTensor());	// 判断input_tensor是否是Tensor
	// 5.ort_inputs输入
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));

	// 6. infer推理过程
	auto output_tensors = ctx.get()->session.get()->m_Session.Run(
		Ort::RunOptions{ nullptr },
		input_node_names.data(),
		ort_inputs.data(),
		ort_inputs.size(),
		output_node_names.data(),
		ctx.get()->session.get()->mOutputTensorNames.size());
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	// 7. 后处理
	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	anomalyPost(floatarr, outputs, outputDims[0], outputDims[1], outputDims[2]);

	return FF_OK;
}


// \! Segementation
int ORTCORE::segment(
	std::shared_ptr<ORTCore_ctx> ctx, 
	const std::vector<cv::Mat> &cvImages, 
	std::vector<cv::Mat>& outputs
)
{
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);//日志对象
	// 1. ctx是否初始化成功
	if (ctx == nullptr) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "seg", "ctx有问题");
		return FF_ERROR_PNULL;
	}

	// 2. 判断NetType是否正确
	if (ctx.get()->params.netType != FF_SEG)
	{
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "seg", "Illegal calls，please check your NetWorkType");
		return FF_ERROR_NETWORK;
	}

	// 3. session信息与输入信息对比
	std::vector<int64_t> inputDims = ctx.get()->session->mInputDims[0];		// 获得onnx中第一个输入维度（session）
	std::vector<int64_t> outputDims = ctx.get()->session->mOutputDims[0];	// 获得onnx中第一个输出维度（session）
	auto input_name = ctx.get()->session->mInputTensorNames[0];				// 获得onnx中第一个输入名称
	auto output_name = ctx.get()->session->mOutputTensorNames[0];			// 获得onnx中第一个输出名称
	// 获得onnx中第一个输出名称
	// 3.1 获得输入维度
	int session_batch{ 0 }, session_channels{ 0 }, session_height{ 0 }, session_width{ 0 };
	session_batch = inputDims[0];	// 获得batchsize（session）
	session_channels = inputDims[1];// 获得channels（session）
	session_height = inputDims[2];// 获得height（session）
	session_width = inputDims[3];// 获得width（session）
	// 3.2 获得input和output名称
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	output_node_names.push_back(output_name.c_str());	// 获得输出名称
	input_node_names.push_back(input_name.c_str());	// 获得输入名称
	// 3.3 onnx的batchsize内容与输入内容的batchsize对比
	std::vector<cv::Mat> imgs;	// 存放resize后的图片，在判断时，存储在imgs中
	if (cvImages.size() != session_batch) {// batchsize判断
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "seg", "输入图片数组的batchsize与ONNX batchsize不同 ");
		return FF_ERROR_INPUT;
	}
	// 3.4 onnx的c,h,w与输入的图片对比
	for (int i = 0; i < cvImages.size(); i++) {// c,h,w判断
		cv::Mat cv_img = cvImages[i].clone();
		if (cv_img.channels() != session_channels) {// 通道数判断
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "seg", "输入图片的通道数与session不符 ");
			return FF_ERROR_INPUT;
		}
		if (session_height != cv_img.cols || session_width != cv_img.rows) {// h，w判断
			ortLog.W(__FILE__, __LINE__, YLog::INFO, "seg", "输入的图片尺寸与session不相符,自动resize");
			//cv::resize(cv_img, cv_img, cv::Size(session_height, session_width), 0.5, 0.5, cv::INTER_AREA); // 使用此方法与onnx（python）时像对应
			cv::resize(cv_img, cv_img, cv::Size(session_height, session_width), cv::INTER_LINEAR); // 使用此方法与onnx（python）时像对应
		}
		imgs.push_back(cv_img);
	}
	// 3.5 判断图片是否为空
	if (imgs.empty()) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Anomaly", "No images, please check");
		return FF_ERROR_INPUT;
	}

	// 4. 预处理图片
	size_t input_tensor_size = session_batch * session_channels * session_height * session_width;  // 获得图片尺寸
	std::vector<float> input_tensor_values(input_tensor_size);	// 存放最终图片
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);// create input tensor object from data values
	normalization(imgs, input_tensor_values, ctx, memory_info);// 归一化图片， 将结果存储在input_tensor_values
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(// 通过input_tensor_values生成input_tensor
		memory_info,// ort内存类型
		input_tensor_values.data(),// 输入数据
		input_tensor_size,// 数据大小
		ctx.get()->session->mInputDims[0].data(),// onnx输入的shape
		ctx.get()->session.get()->m_Session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().size());
	assert(input_tensor.IsTensor());	// 判断input_tensor是否是Tensor
	// 5.ort_inputs输入
	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));
	// 6. infer推理过程
	auto output_tensors = ctx.get()->session.get()->m_Session.Run(
		Ort::RunOptions{ nullptr },
		input_node_names.data(),
		ort_inputs.data(),
		ort_inputs.size(),
		output_node_names.data(),
		ctx.get()->session.get()->mOutputTensorNames.size());
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	// 7. 后处理
	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	segPost(floatarr, outputs, outputDims[0], outputDims[1], outputDims[2]);
	return FF_OK;
}


// \! Detection
int ORTCORE::detect(
	std::shared_ptr<ORTCore_ctx> ctx, 
	const std::vector<cv::Mat> &cvImages,
	std::vector<std::vector<BBox>>& outputs
)
{
	return FF_OK;
}


// \! 获得input的输入维度
int ORTCORE::getInputDimsK(
	std::shared_ptr<ORTCore_ctx> ctx, 
	int & nBatch, 
	int & nChannels, 
	int & nHeight,
	int & nWidth
)
{
	// 日志文件
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);

	if (ctx == nullptr) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Init failed, can't call getDims");
		return FF_ERROR_PNULL;
	}
	nBatch = ctx->session->mInputDims[0][0];
	nChannels = ctx->session->mInputDims[0][1];
	nHeight = ctx->session->mInputDims[0][2];
	nWidth = ctx->session->mInputDims[0][3];
	return FF_OK;
}

// \! 获得output的输出维度
int ORTCORE::getOutputDimsK(
	std::shared_ptr<ORTCore_ctx> ctx,
	int & nBatch,
	int & nHeight,
	int & nWidth
)
{
	// 日志文件
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);

	if (ctx == nullptr) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Init failed, can't call getDims");
		return FF_ERROR_PNULL;
	}
	nBatch = ctx->session->mOutputDims[0][0];
	nHeight = ctx->session->mOutputDims[0][1];
	nWidth = ctx->session->mOutputDims[0][2];
	return FF_OK;
}

// \! 获得output的输出维度
int ORTCORE::getOutputDimsK(
	std::shared_ptr<ORTCore_ctx> ctx,
	int & nBatch,
	int & numClass
)
{
	// 日志文件
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);

	if (ctx == nullptr) {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "ORTCORE", "Init failed, can't call getDims");
		return FF_ERROR_PNULL;
	}
	nBatch = ctx->session->mOutputDims[0][0];
	numClass = ctx->session->mOutputDims[0][1];
	return FF_OK;
}
