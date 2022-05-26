/*****************************************************************************
* @author : FelixFu
* @date : 2021/10/10 14:40
* @last change :
* @description : TensorRT Engine
*****************************************************************************/
#pragma once
#include <queue>	// 系统库

#include <cuda_runtime.h>  // cuda库
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp> // opencv库

#include "NvInfer.h"  //TensorRT库
#include "NvOnnxParser.h"

#include "common.h"  // TensorRT samples中的函数
#include "buffers.h"
#include "logging.h"

#include "params.h"	// 自定义的数据结构， 程序的接口
#include "loguru.hpp" // https://github.com/emilk/loguru


// \! 定义指针, 管理生成TRT引擎中间类型的独占智能指针
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;



// \! ------------------------------------Engine Start------------------------------
class TRTEngine {
public:
	TRTEngine() {};

	// \!引擎构造
	// \@param 传入Params参数
	// \@param 错误参数nErrorFlag
	TRTEngine(const Params& params, int& nErrnoFlag); 

	std::shared_ptr<nvinfer1::ICudaEngine> Get() const;

public:
	// \! 网络输入输出
	std::vector<nvinfer1::Dims> mInputDims;	                    // The dimensions of the input to the network.
	std::vector<nvinfer1::Dims> mOutputDims;                     // The dimensions of the output to the network.
	std::vector<std::string> mInputTensorNames;	    // 模型输入的名称
	std::vector<std::string> mOutputTensorNames;	// 模型输出的名称

private:
	int buildONNX(const Params& params);			// Build Engine
	int loadEngine(const Params& params);			// 加载Engine

private:
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

};
// \! ------------------------------------Engine End------------------------------


// \! ----------------------Context 执行上下文线程池部分----------------------------

// \! A simple threadsafe queue using a mutex and a condition variable.
template <typename T>
class Queue
{
public:
	Queue() = default;

	Queue(Queue&& other)
	{
		std::unique_lock<std::mutex> lock(other.mutex_);
		queue_ = std::move(other.queue_);
	}

	void Push(T value)
	{
		std::unique_lock<std::mutex> lock(mutex_);
		queue_.push(std::move(value));
		lock.unlock();
		cond_.notify_one();
	}

	T Pop()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		cond_.wait(lock, [this] {return !queue_.empty(); });
		T value = std::move(queue_.front());
		queue_.pop();
		return value;
	}

	size_t Size()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		return queue_.size();
	}

private:
	mutable std::mutex mutex_;
	std::queue<T> queue_;
	std::condition_variable cond_;
};

// \! A pool of available contexts is simply implemented as a queue for our example. 
template <typename Context>
using ContextPool = Queue<std::unique_ptr<Context>>;

// \! A RAII class for acquiring an execution context from a context pool. 
template <typename Context>
class ScopedContext
{
public:
	explicit ScopedContext(ContextPool<Context>& pool)
		: pool_(pool), context_(pool_.Pop())
	{
		context_->Activate();
	}

	~ScopedContext()
	{
		context_->Deactivate();
		pool_.Push(std::move(context_));
	}

	Context* operator->() const
	{
		return context_.get();
	}

private:
	ContextPool<Context>& pool_;
	std::unique_ptr<Context> context_;
};

// \! 执行上下文
class ExecContext
{
public:
	friend ScopedContext<ExecContext>;

	ExecContext(int device, std::shared_ptr<nvinfer1::ICudaEngine> engine) : device_(device)
	{
		cudaError_t st = cudaSetDevice(device_);

		if (st != cudaSuccess)
			throw std::invalid_argument("could not set CUDA device");

		context_.reset(engine->createExecutionContext());
	}

	nvinfer1::IExecutionContext* getContext()
	{
		return context_.get();
	}

private:
	void Activate() {}
	void Deactivate() {}
private:
	int device_;
	SampleUniquePtr<nvinfer1::IExecutionContext> context_;
};
// \! ----------------------Context 执行上下文线程池部分 END---------------------------


// \! 执行上下文结构体
struct TRTCore_ctx
{
	Params params;							// \! 执行上下文的Engine
	std::shared_ptr<TRTEngine> engine;		// \! 一个模型对应一个Engine
	ContextPool<ExecContext> pool;          // \! 一个Engine 对应多个模型
};

