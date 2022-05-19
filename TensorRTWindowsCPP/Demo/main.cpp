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
	int flag_s = 1;
	switch (flag_s)
	{
	case 0:
	{
		// 模型参数
		Params params;
		params.onnxFilePath = "E:/AIDeploy/Env/DemoData/classification/onnxs/G41J3D_20220418.onnx";
		params.engineFilePath = "E:/AIDeploy/Env/DemoData/classification/onnxs/G41J3D_20220418.engine";
		params.maxThread = 4;
		params.netType = LUSTER_CLS;
		params.log_path = "./";

		// 初始化
		int flag;
		auto ctx = trtAPI.init(params, flag);

		// 输入数据
		std::vector<std::string> file_names;
		cv::glob("E:/AIDeploy/Env/DemoData/classification/images3/*.bmp", file_names);
		int batch_size = 1; //batchsize大小，必须小于模型的batchsize大小

		for (int nn = 0; nn < file_names.size() / batch_size; nn++) { // 假如图片很多，需要循环多次来处理
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

			for (int b = 0; b < batch_size; b++) { // (int b = 0; b < inputs3.size(); b++)
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
		// 配置参数
		Params segParams;
		segParams.onnxFilePath = "E:/AIDeploy/Env/DemoData/segmentation/onnxs/UNet_resnet50ExportFloat.onnx";
		segParams.engineFilePath = "E:/AIDeploy/Env/DemoData/segmentation/onnxs/UNet_resnet50ExportFloat.engine";
		segParams.netType = LUSTER_SEG;
		segParams.fp16 = false;
		segParams.maxThread = 1;
		segParams.meanValue = { 0.45734706, 0.43338275, 0.40058118 };
		segParams.stdValue = { 0.23965294, 0.23532275, 0.2398498 };
		//segParams.meanValue = { 0.22f };
		//segParams.stdValue = { 0.22f };
		segParams.log_path = "./";

		// 初始化
		int flag;
		auto ctx = trtAPI.init(segParams, flag);

		// 获得所有文件名
		vector<string> file_names;
		cv::glob("E:/AIDeploy/Env/DemoData/segmentation/images/*.jpg", file_names);   //get file names
		int batchSize = 8;

		// 将文件路径 以 batchSize为一组放到inputs_files中
		vector<vector<string>> inputs_files;	// batchSize为一组，共多组
		for (int i = 0; i < file_names.size() / batchSize; i++) {	// 可用整除batchSize个数的组
			vector<string> inputs;
			for (int j = 0; j < batchSize; j++)
			{
				inputs.push_back(file_names[i * batchSize + j]);
			}
			inputs_files.push_back(inputs);
		}
		if (file_names.size() % batchSize != 0) {	// 余数组
			vector<string> inputs;
			for (int i = file_names.size() / batchSize * batchSize; i < file_names.size(); i++)	// i的初始值是上一步骤的末尾值
			{
				inputs.push_back(file_names[i]);
			}
			inputs_files.push_back(inputs);
		}

		// 同理得到mask输出的路径
		vector<vector< string>> outputs_files;
		for (auto files : inputs_files) {
			vector< string> out_files;
			for (auto file : files) {
				string tmp = file;
				tmp.replace(file.find(".jpg"), 4, ".png");
				out_files.push_back(tmp);
			}
			outputs_files.push_back(out_files);
		}

		// 转换格式，并推理
		for (int i = 0; i < inputs_files.size(); i++) {
			vector<CoreImage*> inputs;
			vector<CoreImage*> outputs;
			vector<cv::Mat> cv_imgs, des_imgs;
			CoreImage *inputs_core_images = new CoreImage[batchSize];
			CoreImage *outputs_core_images = new CoreImage[batchSize];

			// 读取一个batchSize的图片
			for (int j = 0; j < inputs_files[i].size(); j++) {
				cv::Mat cv_img = cv::imread(inputs_files[i][j], -1); //cv::IMREAD_GRAYSCALE
				cv_imgs.push_back(cv_img);
				des_imgs.push_back(cv_img.clone().setTo(0));
			}
			// 转换Opencv格式的图片到， CoreImage格式
			for (int j = 0; j < inputs_files[i].size(); j++) {
				inputs_core_images[j].SetValue(cv_imgs[j].channels(), cv_imgs[j].cols, cv_imgs[j].rows, cv_imgs[j].step, (unsigned char *)cv_imgs[j].data);
				inputs.push_back(&inputs_core_images[j]);
				outputs_core_images[j].SetValue(des_imgs[j].channels(), des_imgs[j].cols, des_imgs[j].rows, des_imgs[j].step, (unsigned char *)des_imgs[j].data);
				outputs.push_back(&outputs_core_images[j]);
			}

			TimeTick time;
			time.start();
			trtAPI.segment(ctx, inputs, outputs);
			time.end();
			cout << "infer Time : " << time.mInterval * 1000 << "ms" << endl;

			for (int k = 0; k < outputs.size(); k++)
			{
				cv::Mat tmp = cv::Mat(outputs[k]->height_,
					outputs[k]->width_,
					CV_8UC1,
					outputs[k]->imagedata_,
					outputs[k]->imagestep_).clone();

				//cv::resize(tmp, tmp, cv::Size(800, 800));
				//cv::threshold(tmp, tmp, 0.5, 255, cv::THRESH_BINARY);
				tmp *= 10;
				cv::imwrite(outputs_files[i][k], tmp);
			}
			delete[] inputs_core_images;
			delete[] outputs_core_images;

		}
		break;
	}
	default:
		break;
	}
	return 0;
}
