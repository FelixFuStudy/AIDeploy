#include <opencv2/opencv.hpp> // opencv include
#include <iostream> // system include
#include <Windows.h>

#include "TRTAPI.h"
#include "PARAMS.h"
using namespace std;

// \! 用于计算时间间隔
class TimeTick
{
public:
	TimeTick(void)
	{
		QueryPerformanceFrequency(&mFrequency);
	};  //构造函数

private:
	LARGE_INTEGER mStartTime;
	LARGE_INTEGER mEndTime;
	LARGE_INTEGER mCurrentTime;

	LARGE_INTEGER mFrequency;  // CPU频率 计时的精度跟频率有关，我电脑频率是10e8，计时精度为10纳秒级别

public:
	double mInterval;  // 间隔

public:
	void start()
	{
		QueryPerformanceCounter(&mStartTime);
	};
	void end()
	{
		QueryPerformanceCounter(&mEndTime);

		mInterval = ((double)mEndTime.QuadPart - (double)mStartTime.QuadPart) / (double)mFrequency.QuadPart;  //秒，10e-8级别

	};
	double tick()
	{
		QueryPerformanceCounter(&mCurrentTime);
		return (double)mCurrentTime.QuadPart / (double)mFrequency.QuadPart;
	};
};


int main(int argc, char** argv)
{
	TRTAPI trtAPI;
	// 0 分类测试;
	// 1 分割测试;
	// 2 异常检测测试；
	const int FLAG_TYPE = 0;
	switch (FLAG_TYPE)
	{
	case 0:
	{
		// 1. 模型参数, 9个
		Params params;
		params.onnxFilePath = "E:/AIDeploy/Env/DemoData/classification/onnxs/PZb0b8.onnx";
		params.engineFilePath = "E:/AIDeploy/Env/DemoData/classification/onnxs/PZb0b8.engine";
		params.fp16 = false;
		params.maxThread = 1;
		params.netType = LUSTER_CLS;
		params.log_path = "../../TRT_Log.txt";
		params.meanValue = { 0, 0, 0 };
		params.stdValue = { 1, 1, 1 };
		params.gpuId = 0;

		// 2. 初始化
		int flag;
		auto ctx = trtAPI.init(params, flag);

		// 3. 输入数据
		std::vector<std::string> file_names;
		cv::glob("E:/AIDeploy/Env/DemoData/classification/images/*.bmp", file_names);
		int batch_size, channels, height, width;
		trtAPI.getInputDims(ctx, batch_size, channels, height, width); // 获得onnx中的输入维度

		// 假如图片很多，需要循环多次来处理. 每次处理一个batchsize
		for (int nn = 0; nn < file_names.size() / batch_size; nn++) { 
			std::vector<CoreImage*> inputs1, inputs2; // 线程1, ..., n 的输入
			std::vector<std::vector<ClassifyResult>> outputs1, outputs2;//线程1, ..., n的输出

			std::vector<cv::Mat> inputs1_tmp, inputs2_tmp; // 存放CV::Mat 在此次循环中保留内存
			CoreImage *inputs_core_images1 = new CoreImage[batch_size];// 存放CoreImage的数组，在此次循环中保留在内存中
			//CoreImage *inputs_core_images2 = new CoreImage[batch_size];// 存放CoreImage的数组，在此次循环中保留在内存中

			for (int b = 0; b < batch_size; b++) {  // 每次处理多个batchsize，batchsize大小最好为onnx中的batchsize大小
				inputs1_tmp.push_back(cv::imread(file_names[nn * batch_size + b], cv::IMREAD_GRAYSCALE));
				//inputs2_tmp.push_back(cv::imread(file_names[nn * batch_size + b], cv::IMREAD_GRAYSCALE));

				inputs_core_images1[b].SetValue(inputs1_tmp[b].channels(), inputs1_tmp[b].cols, inputs1_tmp[b].rows, inputs1_tmp[b].step, (unsigned char *)inputs1_tmp[b].data);
				//inputs_core_images2[b].SetValue(inputs2_tmp[b].channels(), inputs2_tmp[b].cols, inputs2_tmp[b].rows,inputs2_tmp[b].step, (unsigned char *)inputs2_tmp[b].data);

				inputs1.push_back(&inputs_core_images1[b]);
				//inputs2.push_back(&inputs_core_images2[b]);
			}

			TimeTick time;
			time.start();
			std::thread obj1(&TRTAPI::classify, std::ref(trtAPI), ctx, std::ref(inputs1), std::ref(outputs1));
			//std::thread obj2(&LUSTERTRT::classify, std::ref(lustertrt), ctx, std::ref(inputs2), std::ref(outputs2));

			obj1.join();
			//obj2.join();

			time.end();
			std::cout << "infer Time : " << time.mInterval * 1000 << "ms" << std::endl;

			for (int b = 0; b < batch_size; b++) {
				std::cout << file_names[nn*batch_size + b] << " ..................... " << "outputs1:::: top1=";
				std::cout << std::to_string(outputs1[b][0].first) << ":" << std::to_string(outputs1[b][0].second);
				std::cout << "    top2=";
				std::cout << std::to_string(outputs1[b][1].first) << ":" << std::to_string(outputs1[b][1].second) << std::endl;
			}
			//for (int b = 0; b < batch_size; b++) { // (int b = 0; b < inputs3.size(); b++)
			//	std::cout << file_names[nn*batch_size + b] << " ..................... " << "outputs2:::: top1=";
			//	std::cout << std::to_string(outputs2[b][0].first) << ":" << std::to_string(outputs2[b][0].second);
			//	std::cout << "    top2=";
			//	std::cout << std::to_string(outputs2[b][1].first) << ":" << std::to_string(outputs2[b][1].second) << std::endl;
			//}
		}// 假如图片很多，需要循环多次来处理
		break;
	}
	case 1:
	{
		// 1.配置参数,9个参数
		Params params;
		params.onnxFilePath = "E:/AIDeploy/Env/DemoData/segmentation/onnxs/PSPNet2_resnet50.onnx";
		params.engineFilePath = "E:/AIDeploy/Env/DemoData/segmentation/onnxs/PSPNet2_resnet50.engine";
		params.netType = LUSTER_SEG;
		params.fp16 = false;
		params.maxThread = 1;
		params.meanValue = { 0.45734706, 0.43338275, 0.40058118 };
		params.stdValue = { 0.23965294, 0.23532275, 0.2398498 };
		params.log_path = "../../TRT_Log.txt";
		params.gpuId = 0;

		// 2.初始化
		int flag;
		auto ctx = trtAPI.init(params, flag);

		// 3. 输入数据
		std::vector<std::string> file_names;
		cv::glob("E:/AIDeploy/Env/DemoData/segmentation/images/*.jpg", file_names);// 获得所有文件名
		int batch_size, channels, height, width;
		trtAPI.getInputDims(ctx, batch_size, channels, height, width); // 获得onnx中的输入维度

		for (int nn = 0; nn < file_names.size() / batch_size; nn++) { // 假如图片很多，需要循环多次来处理
			std::vector<CoreImage*> inputs; // API接口的输入
			std::vector<CoreImage*> outputs; // API接口的输入
			// 这个是真实存放图片的内存空间
			std::vector<cv::Mat> inputsCvImage; // 存放CV::Mat 在此次循环中保留内存
			std::vector<cv::Mat> outputsCvImage; // 存放CV::Mat 在此次循环中保留内存
			// 这个是将CV图片转成CoreImage图片的内存空间
			CoreImage *inputCoreImage = new CoreImage[batch_size];// 存放CoreImage的数组，在此次循环中保留在内存中
			CoreImage *outputsCoreImage = new CoreImage[batch_size];// 存放CoreImage的数组，在此次循环中保留在内存中

			// 将一个batchsize图片读入-->CVImage-->CoreImage，最终存放到inputsCvImage和outputsCvImage中
			for (int b = 0; b < batch_size; b++) {  // 每次处理多个batchsize，batchsize大小 = onnx中的batchsize大小
				// 3.1 处理batch_size中的一个图片--输入
				inputsCvImage.push_back(cv::imread(file_names[nn * batch_size + b], cv::IMREAD_COLOR));// 得到CV格式图片，放入内存
				inputCoreImage[b].SetValue(
					inputsCvImage[b].channels(),
					inputsCvImage[b].cols,
					inputsCvImage[b].rows,
					inputsCvImage[b].step,
					(unsigned char *)inputsCvImage[b].data
				);// 得到CoreImage格式图片，放入内存
				inputs.push_back(&inputCoreImage[b]);//转成输入格式

				// 3.2 处理batch_size中的一个图片--输出
				outputsCvImage.push_back(cv::Mat::zeros(cv::Size(height, width), CV_8UC1));
				outputsCoreImage[b].SetValue(
					outputsCvImage[b].channels(),
					outputsCvImage[b].cols,
					outputsCvImage[b].rows,
					outputsCvImage[b].step,
					(unsigned char *)outputsCvImage[b].data
				);
				outputs.push_back(&outputsCoreImage[b]);
			}

			// 4. 推理
			TimeTick time;
			time.start();
			trtAPI.segment(ctx, inputs, outputs);
			time.end();
			std::cout << "Infer Time : " << time.mInterval * 1000 << "ms" << std::endl;

			// 5. 对结果进行处理
			// 同理得到mask输出的路径
			std::vector<std::string> outputs_files;
			for (auto files : file_names) {
				files.replace(files.find(".jpg"), 4, ".png");
				outputs_files.push_back(files);
			}
			for (int k = 0; k < outputs.size(); k++)
			{
				cv::Mat tmp = cv::Mat(
					outputs[k]->height_,
					outputs[k]->width_,
					CV_8UC1,
					outputs[k]->imagedata_,
					outputs[k]->imagestep_).clone();
				cv::resize(tmp, tmp, cv::Size(inputsCvImage[k].cols, inputsCvImage[k].rows), cv::INTER_NEAREST);
				cv::imwrite(outputs_files[batch_size*nn + k], tmp);
			}
		}// 假如图片很多，需要循环多次来处理
		break;

	}
	case 2:
	{
		// 1. 配置参数
		Params params;
		params.onnxFilePath = "E:/AIDeploy/Env/DemoData/anomaly/onnxs/PaDiM2_b8_56.onnx";
		params.engineFilePath = "E:/AIDeploy/Env/DemoData/anomaly/onnxs/PaDiM2_b8_56.engine";
		params.netType = LUSTER_ANOMALY;
		params.fp16 = false;
		params.maxThread = 1;
		params.meanValue = { 0.335782, 0.335782, 0.335782 };
		params.stdValue = { 0.256730, 0.256730, 0.256730 };
		params.log_path = "../../TRT_Log.txt";
		params.gpuId = 0;

		// 2. 初始化
		int flag;
		auto ctx = trtAPI.init(params, flag);

		// 3. 输入数据
		std::vector<std::string> file_names;
		cv::glob("E:/AIDeploy/Env/DemoData/anomaly/images_16/*.bmp", file_names);
		int batch_size, channels, height, width;
		trtAPI.getInputDims(ctx, batch_size, channels, height, width); // 获得onnx中的输入维度

		for (int nn = 0; nn < file_names.size() / batch_size; nn++) { // 假如图片很多，需要循环多次来处理
			std::vector<CoreImage*> inputs; // API接口的输入
			std::vector<CoreImage*> outputs; // API接口的输入
			// 这个是真实存放图片的内存空间
			std::vector<cv::Mat> inputsCvImage; // 存放CV::Mat 在此次循环中保留内存
			std::vector<cv::Mat> outputsCvImage; // 存放CV::Mat 在此次循环中保留内存
			// 这个是将CV图片转成CoreImage图片的内存空间
			CoreImage *inputCoreImage = new CoreImage[batch_size];// 存放CoreImage的数组，在此次循环中保留在内存中
			CoreImage *outputsCoreImage = new CoreImage[batch_size];// 存放CoreImage的数组，在此次循环中保留在内存中

			// 将一个batchsize图片读入-->CVImage-->CoreImage，最终存放到inputsCvImage和outputsCvImage中
			for (int b = 0; b < batch_size; b++) {  // 每次处理多个batchsize，batchsize大小 = onnx中的batchsize大小
				// 3.1 处理batch_size中的一个图片--输入
				inputsCvImage.push_back(cv::imread(file_names[nn * batch_size + b], cv::IMREAD_COLOR));// 得到CV格式图片，放入内存
				inputCoreImage[b].SetValue(
					inputsCvImage[b].channels(),
					inputsCvImage[b].cols,
					inputsCvImage[b].rows,
					inputsCvImage[b].step,
					(unsigned char *)inputsCvImage[b].data
				);// 得到CoreImage格式图片，放入内存
				inputs.push_back(&inputCoreImage[b]);//转成输入格式

				// 3.2 处理batch_size中的一个图片--输出
				outputsCvImage.push_back(cv::Mat::zeros(cv::Size(224, 224), CV_8UC1));
				outputsCoreImage[b].SetValue(
					outputsCvImage[b].channels(),
					outputsCvImage[b].cols,
					outputsCvImage[b].rows,
					outputsCvImage[b].step,
					(unsigned char *)outputsCvImage[b].data
				);
				outputs.push_back(&outputsCoreImage[b]);
			}

			// 4. 推理
			TimeTick time;
			time.start();
			trtAPI.anomaly(ctx, inputs, outputs, 0.39915153);
			time.end();
			std::cout << "Infer Time : " << time.mInterval * 1000 << "ms" << std::endl;

			// 5. 对结果进行处理
			// 同理得到mask输出的路径
			std::vector<std::string> outputs_files;
			for (auto files : file_names) {
				files.replace(files.find(".bmp"), 4, ".png");
				outputs_files.push_back(files);
			}
			for (int k = 0; k < outputs.size(); k++)
			{
				cv::Mat tmp = cv::Mat(
					outputs[k]->height_,
					outputs[k]->width_,
					CV_8UC1,
					outputs[k]->imagedata_,
					outputs[k]->imagestep_).clone();

				cv::imwrite(outputs_files[batch_size*nn+k], tmp);
			}
		}// 假如图片很多，需要循环多次来处理
		break;
		
	}
	default:
		break;
	}
	return 0;
}
