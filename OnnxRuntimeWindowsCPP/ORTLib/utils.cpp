#include "utils.h"

// \! 分类、异常检测前处理
int normalization(
	std::vector<cv::Mat>& cv_images, //输入图像
	std::vector<float> &input_tensor_values, //目标地址
	std::shared_ptr<ORTCore_ctx> ctx, //执行上下文
	Ort::MemoryInfo &memory_info// ort类型
)
{
	// 1.日志对象初始化
	YLog ortLog(YLog::INFO, ctx.get()->params.log_path, YLog::ADD);

	// 2.从ctx中获得onnx输入信息 和 params
	std::vector<int64_t> inputDims = ctx.get()->session->mInputDims[0];		// 第一个输入维度（session）
	int session_batch{ 0 }, session_channels{ 0 }, session_height{ 0 }, session_width{ 0 };
	session_batch = inputDims[0];	// 获得batchsize（session）
	session_channels = inputDims[1];// 获得channels（session）
	session_height = inputDims[2];// 获得height（session）
	session_width = inputDims[3];// 获得width（session）
	size_t input_tensor_size = session_batch * session_channels * session_height * session_width;  // 获得图片尺寸
	Params params = ctx.get()->params;// 从ctx中获得params

	// 3.read image
	std::vector<float> dst_data;
	if (session_channels == 1) {// 通道数为1
		for (int i = 0; i < cv_images.size(); i++) {
			cv_images[i].convertTo(cv_images[i], CV_32FC1, 1 / 255.0);
			cv_images[i] = (cv_images[i] - params.meanValue[0]) / params.stdValue[0];
			std::vector<float> data = std::vector<float>(cv_images[i].reshape(1, 1));
			dst_data.insert(dst_data.end(), data.begin(), data.end());
		}
	}
	else if (session_channels == 3) {
		// 新的方法，参考：https://blog.csdn.net/juluwangriyue/article/details/123041695
		for (int n = 0, index = 0; n < cv_images.size(); n++)
		{
			cv_images[n].convertTo(cv_images[n], CV_32F, 1.0 / 255);
			std::vector<cv::Mat> bgrChannels(3);//借用来进行HWC->CHW
			cv::split(cv_images[n], bgrChannels);
			for (int i = 0; i < cv_images[n].channels(); i++)
			{
				bgrChannels[i] -= params.meanValue[i];  // mean
				bgrChannels[i] /= params.stdValue[i];   // std
			}
			for (int i = 0; i < cv_images[n].channels(); i++)  // BGR2RGB, HWC->CHW
			{
				std::vector<float> data = std::vector<float>(bgrChannels[2 - i].reshape(1, cv_images[n].cols * cv_images[n].rows));
				dst_data.insert(dst_data.end(), data.begin(), data.end());
			}
		}
	}
	else {
		ortLog.W(__FILE__, __LINE__, YLog::INFO, "Normalization", "不支持的图像类型");
		return FF_ERROR_INPUT;
	}
	// 将dst_data赋值到input_tensor_values
	for (unsigned int i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = dst_data[i];
}


// 比较Pair
bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
	return lhs.first > rhs.first;
}
// 分类任务中返回最大的N类得分类别,输入大小是类别数目c个，输出大小N
std::vector<int> Argmax(const std::vector<float>& v, int N)
{
	// 将v（一个batch的所有类别的softmax值）变成pair对
	std::vector<std::pair<float, int>> pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));

	// 将pairs排序
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	// 将pairs排序后的前N项的索引返回
	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}
// \! 分类后处理
int clsPost(
	float* floatarr, 
	std::vector<std::vector<ClassifyResult>>& outputs, 
	const int batch, 
	const int num_class
)
{
	// 1. Top K
	int N = 3;
	auto K = N > num_class ? num_class : N;

	// 2. softmax
	std::vector<float> output_buffer;
	for (int b = 0; b < batch; b++) {
		// 2.1 求一个batch的sum， 并将每个类的exp数值存放到output_buffer中
		float sum{ 0.0f };
		for (int i = 0; i < num_class; i++) {
			output_buffer.push_back(exp(floatarr[b * num_class + i]));
			sum += output_buffer[b * num_class + i];
		}
		// 2.2 求softmax，output存放一张图片的所有类别的置信度
		std::vector<float> output;
		for (int j = 0; j < num_class; j++) {
			output.push_back(output_buffer[b * num_class + j] / sum);
		}
		// 2.3 求Top K的下标和分值，output topk的index 放入maxN中
		std::vector<int> maxN = Argmax(output, K);
		std::vector<ClassifyResult> classifyResults;
		for (int i = 0; i < K; ++i)
		{
			int idx = maxN[i];
			classifyResults.push_back(std::make_pair(idx, output[idx]));
		}
		outputs.push_back(classifyResults);
	}

	return FF_OK;
}


// \! 异常检测后处理
int anomalyPost(
	float* floatarr,// onnxruntime 推理的结果
	std::vector<cv::Mat>& outputs,// 存储输出的结果
	const int output_batch,// output的batchsize
	const int output_height,//output的高
	const int output_width//output的宽
)
{

	for (int i = 0; i < output_batch; i++)
	{
		cv::Mat res = cv::Mat(output_height, output_width, CV_32FC1, floatarr + output_height * output_width * i).clone();
		outputs.push_back(res);
	}
	return FF_OK;
}

// \! 分割后处理
int segPost(
	float* floatarr,// onnxruntime 推理的结果
	std::vector<cv::Mat>& outputs,// 存储输出的结果
	const int output_batch,// output的batchsize
	const int output_height,//output的高
	const int output_width//output的宽
)
{
	for (int i = 0; i < output_batch; i++){
		cv::Mat res = cv::Mat(output_height, output_width, CV_32FC1, floatarr + output_height * output_width * i).clone();
		outputs.push_back(res);
	}
	return FF_OK;
}


//int segPostOutput(const samplesCommon::BufferManager & buffers, std::vector<cv::Mat>& out_masks, const int numSamples,
//	const Params& params, const nvinfer1::Dims outputDims, bool verbose, const std::string outputName)
//{
//	out_masks.clear();
//	float* output = static_cast<float*>(buffers.getHostBuffer(outputName));
//	int32_t num_class, height, width;
//
//	if (verbose) // 输出概率图
//	{
//		num_class = outputDims.d[1];
//		height = outputDims.d[2];
//		width = outputDims.d[3];
//		for (int i = 0; i < numSamples; i++)
//		{
//			std::vector<cv::Mat> n_channels;
//			//cv::Mat res = cv::Mat(height, width, CV_32FC(num_class), output + height * width * num_class * i).clone();
//			cv::Mat res;
//			for (int j = 0; j < num_class; j++) {
//				cv::Mat tmp = cv::Mat(height, width, CV_32F, output + height * width * j + height * width * i * num_class).clone();
//				n_channels.push_back(tmp);
//			}
//			cv::merge(n_channels, res);
//			out_masks.push_back(res);
//		}
//	}
//	else
//	{
//		if (outputDims.nbDims == 3)  // onnx本身就是输出类别，无需转换。类别数越多，这种方式越体现效率优势
//		{
//			num_class = 0;  // 输出类别图，这里已经没有num_class，num_class在像素值中体现
//			height = outputDims.d[1];
//			width = outputDims.d[2];
//
//			for (int i = 0; i < numSamples; i++)
//			{
//				cv::Mat res = cv::Mat(height, width, CV_32F, output + height * width * i).clone();
//				res.convertTo(res, CV_8UC1);
//				out_masks.push_back(res);
//			}
//		}
//		else  // onnx本身是概率值，需要自己进行转换，效率低，为了兼容之前的才保留此段代码分支。后期将不再支持。
//		{
//			num_class = outputDims.d[1];
//			height = outputDims.d[2];
//			width = outputDims.d[3];
//			if (num_class == 2)
//			{  // 只需要做前景背景二分割的时候，这种形式的代码效率高、可读性强
//				for (int n = 0; n < numSamples; n++)
//				{
//					// 输出两层h*w的矩阵。第一层是bg的概率，第二层是crack的概率。两层的概率相加等于1
//					// 对bg层以0.5为阈值二值化，小于0.5的为crack
//					cv::Mat tmp = cv::Mat(height, width, CV_32F, output + height * width * (2 * n)).clone();
//					cv::threshold(tmp, tmp, 0.5f, 1.f, cv::THRESH_BINARY_INV);
//					tmp.convertTo(tmp, CV_8UC1);
//					out_masks.push_back(tmp);
//				}
//			}
//			else
//			{	// 类别数大于2的情况
//				for (int n = 0; n < numSamples; n++)
//				{
//					std::vector<cv::Mat> n_channels;
//					//cv::Mat res = cv::Mat(height, width, CV_32FC(num_class), output + height * width * num_class * i).clone();
//					cv::Mat res;
//					for (int j = 0; j < num_class; j++) {
//						cv::Mat tmp = cv::Mat(height, width, CV_32F, output + height * width * j + height * width * n * num_class).clone();
//						n_channels.push_back(tmp);
//					}
//					cv::merge(n_channels, res);
//					res.convertTo(res, CV_8UC(num_class));
//
//					cv::Mat tmp = cv::Mat::zeros(height, width, CV_8U);
//					for (int h = 0; h < height; ++h) {
//						for (int w = 0; w < width; ++w) {
//							//uchar *p = res.ptr(h, w); // prob of a point
//							uchar *p = res.ptr(h, w); // prob of a point
//							tmp.at<uchar>(h, w) = (uchar)std::distance(p, std::max_element(p, p + num_class));
//						}
//					}
//					// todo 嵌套循环效率极低。。。
//					//cv::Mat tmp = cv::Mat::zeros(height, width, CV_32F);
//					//for (int row = 0; row < height; row++) {
//					//	float* dataP = tmp.ptr<float>(row);
//					//	for (int col = 0; col < width; col++) {
//					//		std::vector<float> prop(num_class, 0.f);
//					//		for (int indexC = 0; indexC < num_class; indexC++) {
//					//			prop[indexC] = output[n * num_class * height * width + indexC * height * width + row * height + col];
//					//		}
//					//		size_t maxIndex = std::distance(prop.begin(), std::max_element(prop.begin(), prop.end()));
//					//		//size_t maxIndex = argmax(prop.begin(), prop.end());
//					//		dataP[col] = float(maxIndex);
//					//	}
//					//}
//					//tmp.convertTo(tmp, CV_8UC1);
//
//					out_masks.push_back(tmp);
//				}
//			}
//		}
//	}
//
//	return LY_OK;
//}
//

