#include "trtCore.h"
#include "engine.h"
#include "Utils.h"

// \! 初始化
// \@param params     初始化参数
// \@param nErrnoFlag 初始化错误码，详情见params.h
std::shared_ptr<TRTCore_ctx> TRTCORE::init(const Params& params, int& nErrnoFlag) 
{
	LOG_F(INFO, "Init Start ......");
	try {
		// 1. 判断此电脑GPU是否兼容代码否
		if (!IsCompatible(params.gpuId)) {
			nErrnoFlag = LY_WRONG_CUDA;
			LOG_F(INFO, "GPU Compatible Error");
			return nullptr;
		}
		std::string log_ = std::string("Using GPU No. ") + std::to_string(params.gpuId);
		LOG_F(INFO, log_.c_str());

		// 2. build Engine. 生成引擎
		std::shared_ptr<TRTEngine> engine_ptr(new TRTEngine(params, nErrnoFlag));
		if (nErrnoFlag != LY_OK) {
			LOG_F(INFO, "Can't build or load engine file");
			return nullptr;
		}
		
		// 3.generate Contexts pools， 通过引擎和一些配置参数，获得执行上下文，线程池
		ContextPool<ExecContext> pool;
		for (int i = 0; i < params.maxThread; ++i)
		{
			std::unique_ptr<ExecContext> context(new ExecContext(params.gpuId, engine_ptr->Get()));
			pool.Push(std::move(context));
		}
		if (pool.Size() == 0) {
			nErrnoFlag = LY_WRONG_CUDA;
			LOG_F(INFO, "No suitable CUDA device");
			return nullptr;
		}

		// 4.产生执行上下文
		std::shared_ptr<TRTCore_ctx> ctx(new TRTCore_ctx{ params,  engine_ptr, std::move(pool) });
		LOG_F(INFO, "Init Successfully !!!!");

		return ctx;
	}
	catch (const std::invalid_argument& ex) {
		LOG_F(INFO, "Init failed !!!!");
		nErrnoFlag = LY_UNKNOW_ERROR;
		return nullptr;
	}
}

// \! 分类
// \@param ctx:执行上下文
// \@param vInCoreImages:输入图像列表，Mat格式
// \@param vvOutClsRes:输出结果，ClassifyResult格式
int TRTCORE::classify(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<cv::Mat> &cvImages,std::vector<std::vector<ClassifyResult>>& outputs)
{
	// 1.ctx是否初始化成功
	if (ctx == nullptr) {
		LOG_F(INFO, "Init Failed，Can't call classify function !");
		return LY_UNKNOW_ERROR;
	}

	// 2. 判断NetType是否正确
	if (ctx.get()->params.netType != LUSTER_CLS)
	{
		LOG_F(INFO, "Illegal calls，please check your NetWorkType");
		return LY_WRONG_CALL;
	}

	// 3. Engine信息与输入信息对比
	int engine_batch, engine_channels, engine_height, engine_width;
	this->getInputDims(ctx, engine_batch, engine_channels, engine_height, engine_width);// 获得输入维度信息
	int engine_output_batch, engine_output_numClass;
	this->getOutputDims2(ctx, engine_output_batch, engine_output_numClass);//获得输出维度信息
	auto engine_input_name = this->getInputNames(ctx);  //获得输入名称
	auto engine_output_name = this->getOutputNames(ctx);//获得输出名称
	// 3.1 batchsize判断
	if (cvImages.size() > engine_batch) {
		LOG_F(INFO, "输入图片数组的batchsize超过Engine预定值 ");
		return LY_WRONG_IMG;
	}
	// 3.2 c,h,w判断
	std::vector<cv::Mat> imgs;
	for (int i = 0; i < cvImages.size(); i++) {
		cv::Mat cv_img = cvImages[i].clone();
		if (cv_img.channels() != engine_channels) {
			LOG_F(INFO, "输入图片的通道数与Engine不符 ");
			return LY_WRONG_IMG;
		}
		if (engine_height != cv_img.cols || engine_width != cv_img.rows) {
			LOG_F(WARNING, "输入的图片尺寸与Engine不相符,自动resize");
			cv::resize(cv_img, cv_img, cv::Size(engine_height, engine_width), 0, 0, cv::INTER_LINEAR);
		}
		imgs.push_back(cv_img);
	}
	if (imgs.empty()) {
		LOG_F(INFO, "No images, please check");
		return LY_WRONG_IMG;
	}

	// 4.预处理
	samplesCommon::BufferManager buffers(ctx.get()->engine->Get());// 分配显存（输入和输出）
	if (LY_OK != normalization(buffers, imgs, ctx.get()->params, engine_input_name))
	{
		LOG_F(INFO, "CPU2GPU 内存拷贝失败");
		return LY_UNKNOW_ERROR;
	}
	buffers.copyInputToDevice();
	
	// 5.执行推理过程
	// mContext->executeV2 是这个项目中最核心的一句代码，模型检测就是这一步。
	ScopedContext<ExecContext> context(ctx->pool);
	auto ctx_ = context->getContext();
	if (!ctx_->executeV2(buffers.getDeviceBindings().data()))
	{
		LOG_F(INFO, "执行推理失败");
		return LY_UNKNOW_ERROR;
	}

	// 6. 后处理
	buffers.copyOutputToHost();
	if (LY_OK != clsPost(buffers, outputs, engine_output_batch, engine_output_numClass, engine_output_name)) {
		LOG_F(INFO, "GPU2CPU 内存拷贝失败");
		return LY_UNKNOW_ERROR;
	}

	return LY_OK;
}

// \! 分割
// \@param ctx: 执行上下文
// \@param vInCoreImages: 输入图片vector，cvImage
// \@param vOutCoreImages:输出图片vector，cvImage
// \@param verbose: 如果为true,return the probability graph, else return the class id image
int TRTCORE::segment(std::shared_ptr<TRTCore_ctx> ctx, const std::vector<cv::Mat> &cvImages, std::vector<cv::Mat>& outputs, bool verbose)
{
	// 1. ctx是否初始化成功
	if (ctx == nullptr) {
		LOG_F(INFO, "Init failed, can't call segment");
		return LY_UNKNOW_ERROR;
	}

	// 2.NetType 是否正确
	if (ctx->params.netType != LUSTER_SEG)
	{
		LOG_F(INFO, "Illegal calls，please check your NetWorkType");
		return LY_WRONG_CALL;
	}

	// 3. Engine信息与输入信息对比
	int engine_batch, engine_channels, engine_height, engine_width;
	this->getInputDims(ctx, engine_batch, engine_channels, engine_height, engine_width);// 获得输入维度信息
	int engine_output_batch, engine_output_height, engine_output_width;
	this->getOutputDims(ctx, engine_output_batch, engine_output_height, engine_output_width);//获得输出维度信息
	auto engine_input_name = this->getInputNames(ctx);  //获得输入名称
	auto engine_output_name = this->getOutputNames(ctx);//获得输出名称
	// 3.1 batchsize判断
	if (cvImages.size() > engine_batch) {
		LOG_F(INFO, "输入图片数组的batchsize超过Engine预定值 ");
		return LY_WRONG_IMG;
	}
	// 3.2 c,h,w判断
	std::vector<cv::Mat> imgs;//存放最终的imgs
	for (int i = 0; i < cvImages.size(); i++) {
		cv::Mat cv_img = cvImages[i].clone();
		if (cv_img.channels() != engine_channels) {
			LOG_F(INFO, "输入图片的通道数与Engine不符 ");
			return LY_WRONG_IMG;
		}
		if (engine_height != cv_img.cols || engine_width != cv_img.rows) {
			LOG_F(WARNING, "输入的图片尺寸与Engine不相符,自动resize");
			cv::resize(cv_img, cv_img, cv::Size(engine_height, engine_width), cv::INTER_LINEAR); // 使用此方法与onnx（python）时像对应
		}
		imgs.push_back(cv_img);
	}
	if (imgs.empty()) {
		LOG_F(INFO, "No images, please check");
		return LY_WRONG_IMG;
	}

	// 4. 预处理图片
	samplesCommon::BufferManager buffers(ctx.get()->engine->Get());// 分配显存（输入和输出）
	// 预处理
	if (LY_OK != normalization(buffers, imgs, ctx.get()->params, engine_input_name))
	{
		LOG_F(INFO, "CPU2GPU 内存拷贝失败");
		return LY_UNKNOW_ERROR;
	}
	buffers.copyInputToDevice();

	// 5. 执行推理过程
	// mContext->executeV2 是这个项目中最核心的一句代码，模型检测就是这一步。
	ScopedContext<ExecContext> context(ctx->pool);
	auto ctx_ = context->getContext();
	if (!ctx_->executeV2(buffers.getDeviceBindings().data()))
	{
		LOG_F(INFO, "执行推理失败");
		return LY_UNKNOW_ERROR;
	}

	// 6. 后处理
	buffers.copyOutputToHost();
	int segPostFlag = segPost(
		buffers,
		outputs,
		engine_output_batch,
		engine_output_height,
		engine_output_width,
		engine_output_name);
	if (LY_OK != segPostFlag) {
		LOG_F(INFO, "GPU2CPU 内存拷贝失败");
		return LY_UNKNOW_ERROR;
	}

	return LY_OK;
}

// \! 目标检测
int TRTCORE::detect(
	std::shared_ptr<TRTCore_ctx> ctx, 
	const std::vector<cv::Mat> &coreImages, 
	std::vector<std::vector<BBox>>& outputs
)
{
	return LY_UNKNOW_ERROR;
}

// \! 异常检测
// \@param ctx:执行上下文
// \@param cvImages:输入图片列表，Mat
// \@param outputs:输出图片数组，Mat
int TRTCORE::anomaly(std::shared_ptr<TRTCore_ctx> ctx,const std::vector<cv::Mat> &cvImages,std::vector<cv::Mat>& outputs)
{
	// 1.ctx是否初始化成功
	if (ctx == nullptr) {
		LOG_F(INFO, "Init failed, can't call anomaly.");
		return LY_UNKNOW_ERROR;
	}

	// 2. NetType 是否正确
	if (ctx->params.netType != LUSTER_ANOMALY)
	{
		LOG_F(INFO, "Illegal calls，please check your NetWorkType");
		return LY_WRONG_CALL;
	}

	// 3.Engine信息与输入信息对比
	int engine_batch, engine_channels, engine_height, engine_width;
	this->getInputDims(ctx, engine_batch, engine_channels, engine_height, engine_width);// 获得输入维度信息
	int engine_output_batch, engine_output_height, engine_output_width;
	this->getOutputDims(ctx, engine_output_batch, engine_output_height, engine_output_width);//获得输出维度信息
	auto engine_input_name = this->getInputNames(ctx);  //获得输入名称
	auto engine_output_name = this->getOutputNames(ctx);//获得输出名称
	// 3.1 batchsize判断
	if (cvImages.size() > engine_batch) {
		LOG_F(INFO, "输入图片数组的batchsize超过Engine预定值 ");
		return LY_WRONG_IMG;
	}
	// 3.2 c,h,w判断
	std::vector<cv::Mat> imgs;//存放最终的imgs
	for (int i = 0; i < cvImages.size(); i++) {
		cv::Mat cv_img = cvImages[i].clone();
		if (cv_img.channels() != engine_channels) {
			LOG_F(ERROR, "输入图片的通道数与Engine不符 ");
			return LY_WRONG_IMG;
		}
		if (engine_height != cv_img.cols || engine_width != cv_img.rows) {
			LOG_F(WARNING, "输入的图片尺寸与Engine不相符,自动resize");
			cv::resize(cv_img, cv_img, cv::Size(engine_height, engine_width), cv::INTER_LINEAR); // 使用此方法与onnx（python）时像对应
		}
		imgs.push_back(cv_img);
	}
	if (imgs.empty()) {
		LOG_F(INFO, "No images, please check");
		return LY_WRONG_IMG;
	}

	// 4. 预处理图片
	samplesCommon::BufferManager buffers(ctx.get()->engine->Get());// 分配显存（输入和输出）
	// 预处理
	if (LY_OK != normalization(buffers, imgs, ctx.get()->params, engine_input_name))
	{
		LOG_F(INFO, "CPU2GPU 内存拷贝失败");
		return LY_UNKNOW_ERROR;
	}
	buffers.copyInputToDevice();
	
	// 5. 执行推理过程
	// mContext->executeV2 是这个项目中最核心的一句代码，模型检测就是这一步。
	ScopedContext<ExecContext> context(ctx->pool);
	auto ctx_ = context->getContext();
	if (!ctx_->executeV2(buffers.getDeviceBindings().data()))
	{
		LOG_F(INFO, "执行推理失败");
		return LY_UNKNOW_ERROR;
	}
	
	// 6. 后处理
	buffers.copyOutputToHost();
	int anomalyPostFlag = anomalyPost(
		buffers,
		outputs, 
		engine_output_batch, 
		engine_output_height, 
		engine_output_width,
		engine_output_name);
	if (LY_OK != anomalyPostFlag) {
		LOG_F(INFO, "GPU2CPU 内存拷贝失败");
		return LY_UNKNOW_ERROR;
	}

	return LY_OK;

}

// \! 获取显卡数量
// \@param ctx:执行上下文
// \@param number:gpu数量
int TRTCORE::getNumberGPU(std::shared_ptr<TRTCore_ctx> ctx,int& number)
{
	cudaError_t st = cudaGetDeviceCount(&number);
	if (st != cudaSuccess) {
		LOG_F(INFO, "Could not list CUDA devices");
		return 0;
	}
	return LY_OK;
}

// \! 获取输入维度
// \@param ctx:执行上下文
// \@param nBatch:batchsize
// \@param nChannels:channels
// \@param nHeight:height
// \@param nWidth:width
// \@param index:第index个输入，加入onnx有多个输入，则通过index来指定
int TRTCORE::getInputDims(std::shared_ptr<TRTCore_ctx> ctx,int & nBatch,int & nChannels,int & nHeight,int & nWidth,int index)
{
	if (ctx == nullptr) {
		LOG_F(INFO, "init failed, can't call getDims");
		return LY_UNKNOW_ERROR;
	}
	auto inputDims = ctx->engine->mInputDims[index];
	nBatch = inputDims.d[0];
	nChannels = inputDims.d[1];
	nHeight = inputDims.d[2];
	nWidth = inputDims.d[3];
	return LY_OK;
}

// \! 获取输出维度
// \@param ctx：执行上下文
// \@param nBatch:batchsize
// \@param nHeight:Height
// \@param nWidth:Width
// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
int TRTCORE::getOutputDims(std::shared_ptr<TRTCore_ctx> ctx,int & nBatch,int & nHeight,int & nWidth,int index)
{
	if (ctx == nullptr) {
		LOG_F(INFO, "init failed, can't call getDims");
		return LY_UNKNOW_ERROR;
	}
	auto outputDims = ctx->engine->mOutputDims[index];
	nBatch = outputDims.d[0];
	nHeight = outputDims.d[1];
	nWidth = outputDims.d[2];
	return LY_OK;
}
// \! 获取输出维度
// \@param ctx：执行上下文
// \@param nBatch:batchsize
// \@param nNumClass:NumClass 类别数，针对分类
// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
int TRTCORE::getOutputDims2(std::shared_ptr<TRTCore_ctx> ctx,int & nBatch,int & nNumClass,int index)
{
	if (ctx == nullptr) {
		LOG_F(INFO, "init failed, can't call getDims");
		return LY_UNKNOW_ERROR;
	}
	auto outputDims = ctx->engine->mOutputDims[index];
	nBatch = outputDims.d[0];
	nNumClass = outputDims.d[1];
	return LY_OK;
}

// \! 获取输入名称
// \@param ctx：执行上下文
// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
std::string TRTCORE::getInputNames(std::shared_ptr<TRTCore_ctx> ctx,int index)
{
	auto engine_input_name = ctx.get()->engine->mInputTensorNames[index];
	return engine_input_name;
}

// \! 获取输出名称
// \@param ctx：执行上下文
// \@param index:第index个输出，假如onnx有多个输出，则通过index来指定
std::string TRTCORE::getOutputNames(std::shared_ptr<TRTCore_ctx> ctx,int index)
{
	auto engine_output_name = ctx.get()->engine->mOutputTensorNames[index];
	return engine_output_name;
}