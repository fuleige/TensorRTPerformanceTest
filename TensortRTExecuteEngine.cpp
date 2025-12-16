
#include "TensortRTExecuteEngine.h"

namespace fulei
{

    ExecutionContext::ExecutionContext(nvinfer1::ICudaEngine *engine, const EngineConfig &config)
    {
        // 1. 利用引擎创建新的执行上下文
        context_.reset(engine->createExecutionContext());
        if (!context_)
        {
            FL_LOG_ERROR("创建TensorRT执行上下文失败");
            return;
        }

        // 2. 创建新的cuda流
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        {
            FL_LOG_ERROR("创建CUDA流失败");
            context_.reset();
            return;
        }

        data_type_size_ = config.data_type_size;

        // 3. 预分配缓冲区
        const int num_io_tensors = engine->getNbIOTensors();
        for (int i = 0; i < num_io_tensors; ++i)
        {
            const char *name = engine->getIOTensorName(i);
            size_t memory_size = config.max_batch_size * config.data_type_size;
            auto dims = engine->getTensorShape(name);
            // 输入节点
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                if (config.input_nodes_shape.count(name))
                {
                    dims = config.input_nodes_shape.at(name);
                }
                // 计算输入张量的内存大小
                for (int j = 1; j < dims.nbDims; ++j)
                {
                    memory_size *= dims.d[j];
                }
                input_host_sizes_.emplace_back(dims);
                // 申请输入显存
                input_names.emplace_back(name);
                input_gpu_ptrs_.emplace_back(nullptr);
                input_host_ptrs_.emplace_back(nullptr);
                if (cudaMalloc(&input_gpu_ptrs_.back(), memory_size) != cudaSuccess)
                {
                    FL_LOG_ERROR("为输入张量 %s 分配显存失败，大小: %zu，错误: %s", name, memory_size, cudaGetErrorString(cudaGetLastError()));
                    context_.reset();
                    return;
                }
                if (!context_->setInputTensorAddress(name, input_gpu_ptrs_.back()))
                {
                    FL_LOG_ERROR("为输入张量 %s 设置显存地址失败，错误: %s", name, cudaGetErrorString(cudaGetLastError()));
                    context_.reset();
                    return;
                }
                if (cudaMallocHost(&input_host_ptrs_.back(), memory_size) != cudaSuccess)
                {
                    FL_LOG_ERROR("为输入张量 %s 分配主机内存失败，大小: %zu，错误: %s", name, memory_size, cudaGetErrorString(cudaGetLastError()));
                    context_.reset();
                    return;
                }
            }
            // 输出节点
            else
            {
                output_names.emplace_back(name);

                if (config.output_nodes_shape.count(name))
                {
                    dims = config.output_nodes_shape.at(name);
                }
                // 计算输入张量的内存大小
                for (int j = 1; j < dims.nbDims; ++j)
                {
                    memory_size *= dims.d[j];
                }
                output_host_sizes_.emplace_back(dims);
                output_gpu_ptrs_.emplace_back(nullptr);
                output_host_ptrs_.emplace_back(nullptr);
                // 申请输出显存
                if (cudaMalloc(&output_gpu_ptrs_.back(), memory_size) != cudaSuccess)
                {
                    FL_LOG_ERROR("为输出张量 %s 分配显存失败，大小: %zu，错误: %s", name, memory_size, cudaGetErrorString(cudaGetLastError()));
                    context_.reset();
                    return;
                }
                if (!context_->setOutputTensorAddress(name, output_gpu_ptrs_.back()))
                {
                    FL_LOG_ERROR("为输出张量 %s 设置显存地址失败", name);
                    context_.reset();
                    return;
                }
                if (cudaMallocHost(&output_host_ptrs_.back(), memory_size) != cudaSuccess)
                {
                    FL_LOG_ERROR("为输出张量 %s 分配主机内存失败，大小: %zu，错误: %s", name, memory_size, cudaGetErrorString(cudaGetLastError()));
                    context_.reset();
                    return;
                }
            }
        }
    }

    const char *ExecutionContext::getInputAddress(const std::string &input_name)
    {
        for (size_t i = 0; i < input_names.size(); ++i)
        {
            if (input_names[i] == input_name)
            {
                return input_host_ptrs_[i];
            }
        }
        return nullptr;
    }

    const char *ExecutionContext::getOutputAddress(const std::string &output_name)
    {
        for (size_t i = 0; i < output_names.size(); ++i)
        {
            if (output_names[i] == output_name)
            {
                return output_host_ptrs_[i];
            }
        }
        return nullptr;
    }

    // 执行推理
    void ExecutionContext::executeInference(size_t batch_size)
    {
        if (!context_)
        {
            FL_LOG_ERROR("上下文未初始化");
            return;
        }

        // 拷贝输入数据
        for (size_t i = 0; i < input_names.size(); ++i)
        {
            nvinfer1::Dims input_dim = input_host_sizes_[i];
            input_dim.d[0] = batch_size;
            if (!context_->setInputShape(input_names[i].c_str(), input_dim))
            {
                FL_LOG_ERROR("为输入张量 %s 设置显存地址失败，错误: %s", input_names[i].c_str(), cudaGetErrorString(cudaGetLastError()));
                return;
            }
            // 计算输入张量的内存大小
            size_t input_size = batch_size * data_type_size_;
            for (int j = 1; j < input_dim.nbDims; ++j)
            {
                input_size *= input_dim.d[j];
            }
            if (cudaMemcpyAsync(input_gpu_ptrs_[i], input_host_ptrs_[i], input_size, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            {
                FL_LOG_ERROR("拷贝输入张量 %s 到显存失败，大小: %zu，错误: %s", input_names[i].c_str(), input_size, cudaGetErrorString(cudaGetLastError()));
                return;
            }
        }

        // 执行推理
        if (!context_->enqueueV3(stream_))
        {
            FL_LOG_ERROR("执行推理失败，错误: %s", cudaGetErrorString(cudaGetLastError()));
            return;
        }

        // 拷贝输出数据
        for (size_t i = 0; i < output_names.size(); ++i)
        {
            size_t output_size = batch_size * data_type_size_;
            for (int j = 1; j < output_host_sizes_[i].nbDims; ++j)
            {
                output_size *= output_host_sizes_[i].d[j];
            }
            if (cudaMemcpyAsync(output_host_ptrs_[i], output_gpu_ptrs_[i], output_size, cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
            {
                FL_LOG_ERROR("拷贝输出张量 %s 到主机内存失败，大小: %zu，错误: %s", output_names[i].c_str(), output_size, cudaGetErrorString(cudaGetLastError()));
                return;
            }
        }

        if (cudaStreamSynchronize(stream_) != cudaSuccess)
        {
            FL_LOG_ERROR("同步流失败，错误: %s", cudaGetErrorString(cudaGetLastError()));
            return;
        }
    }

    ExecutionContext::~ExecutionContext()
    {
        if (stream_)
        {
            cudaStreamDestroy(stream_);
        }
        context_.reset();
    }

    void TensorRTLogger::log(Severity severity, const char *msg) noexcept
    {
        switch (severity)
        {
        case Severity::kVERBOSE:
            // FL_LOG_DEBUG("TensorRT verbose: %s", msg);
            break;
        case Severity::kINFO:
            FL_LOG_INFO("TensorRT info: %s", msg);
            break;
        case Severity::kWARNING:
            FL_LOG_WARNING("TensorRT warning: %s", msg);
            break;
        case Severity::kERROR:
            FL_LOG_ERROR("TensorRT error: %s", msg);
            break;
        case Severity::kINTERNAL_ERROR:
            FL_LOG_ERROR("TensorRT 内部错误: %s", msg);
            break;
        default:
            FL_LOG_ERROR("TensorRT 未知日志级别: %d", static_cast<int>(severity));
            break;
        }
    }

    nvinfer1::ILogger::Severity TensorRTLogger::getLogLevel() const
    {
        return log_level_;
    }

    void TensorRTLogger::setLogLevel(nvinfer1::ILogger::Severity level)
    {
        log_level_ = level;
    }

    bool TensorRTEngineManager::loadEngineFromFile(const std::string &engine_path)
    {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good())
        {
            FL_LOG_ERROR("无法打开引擎文件: %s", engine_path.c_str());
            return false;
        }

        // 获取文件大小
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        // 读取文件内容
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        // 创建运行时
        runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
        if (!runtime_)
        {
            FL_LOG_ERROR("创建TensorRT运行时失败");
            return false;
        }

        // 反序列化引擎
        shared_engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!shared_engine_)
        {
            FL_LOG_ERROR("反序列化TensorRT引擎失败");
            return false;
        }

        FL_LOG_INFO("从文件加载TensorRT引擎成功: %s", engine_path.c_str());
        return true;
    }

    bool TensorRTEngineManager::initialize(const std::string &model_path, const EngineConfig &config)
    {
        if (is_initialized_)
        {
            FL_LOG_WARNING("TensorRTEngineManager已经初始化");
            return true;
        }

        config_ = config;

        // 检查文件扩展名，决定是加载ONNX还是TensorRT引擎
        std::string extension = model_path.substr(model_path.find_last_of('.') + 1);
        bool success = false;

        if (extension == "trt" || extension == "engine")
        {
            FL_LOG_INFO("从TensorRT引擎文件加载: %s", model_path.c_str());
            success = loadEngineFromFile(model_path);
        }
        else
        {
            FL_LOG_ERROR("不支持的模型文件格式: %s", extension.c_str());
            return false;
        }

        if (!success)
        {
            FL_LOG_ERROR("加载模型失败: %s", model_path.c_str());
            return false;
        }

        is_initialized_ = true;
        FL_LOG_INFO("TensorRTEngineManager初始化成功");
        return true;
    }

    thread_local std::unique_ptr<ExecutionContext> TensorRTEngineManager::thread_context_ = nullptr;

    /**
     * 为当前线程创建执行上下文 (每个线程需要创建)
     * @return 是否创建成功
     */
    ExecutionContext *TensorRTEngineManager::getThreadContext()
    {
        if (!shared_engine_)
        {
            FL_LOG_ERROR("引擎未初始化");
            return nullptr;
        }

        // 创建执行上下文
        if (!thread_context_)
        {
            thread_context_.reset(new ExecutionContext(shared_engine_.get(), config_));
        }

        return thread_context_->isSuccess() ? thread_context_.get() : nullptr;
    }

    const char *TensorRTEngineManager::getInputAddress(const std::string &input_name)
    {
        auto thread_context = getThreadContext();
        if (!thread_context)
        {
            FL_LOG_ERROR("线程上下文未初始化");
            return nullptr;
        }
        return thread_context->getInputAddress(input_name);
    }

    const char *TensorRTEngineManager::getOutputAddress(const std::string &output_name)
    {
        auto thread_context = getThreadContext();
        if (!thread_context)
        {
            FL_LOG_ERROR("线程上下文未初始化");
            return nullptr;
        }
        return thread_context->getOutputAddress(output_name);
    }

    /**
     * 执行推理
     */
    void TensorRTEngineManager::executeInference(size_t batch_size)
    {
        auto thread_context = getThreadContext();
        if (!thread_context)
        {
            FL_LOG_ERROR("线程上下文未初始化");
            return;
        }
        thread_context->executeInference(batch_size);
    }

    void TensorRTEngineManager::ModelDetail()
    {
        if (!shared_engine_)
        {
            FL_LOG_ERROR("引擎未初始化");
            return;
        }
        // 打印模型详情
        int nb_tensors = shared_engine_->getNbIOTensors();
        for (int i = 0; i < nb_tensors; i++)
        {
            const char *name = shared_engine_->getIOTensorName(i);
            nvinfer1::Dims dims = shared_engine_->getTensorShape(name);
            auto io_mode = shared_engine_->getTensorIOMode(name);
            std::cout << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << " " << "Tensor \"" << name << "\" shape: ";
            for (int d = 0; d < dims.nbDims; d++)
            {
                std::cout << dims.d[d] << " ";
            }
            std::cout << std::endl;
        }
    }

}