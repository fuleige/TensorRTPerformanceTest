
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
            FL_LOG_DEBUG("TensorRT verbose: %s", msg);
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

    bool TensorRTEngineManager::buildEngineFromOnnx(const std::string &onnx_path)
    {
        // 创建builder
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger_));
        if (!builder)
        {
            FL_LOG_ERROR("创建TensorRT builder失败");
            return false;
        }

        // 创建网络定义
        const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network)
        {
            FL_LOG_ERROR("创建网络定义失败");
            return false;
        }

        // 创建ONNX解析器
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trt_logger_));
        if (!parser)
        {
            FL_LOG_ERROR("创建ONNX解析器失败");
            return false;
        }

        // 解析ONNX文件
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(trt_logger_.getLogLevel())))
        {
            FL_LOG_ERROR("解析ONNX文件失败: %s", onnx_path.c_str());
            return false;
        }

        // 创建构建配置
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            FL_LOG_ERROR("创建构建配置失败");
            return false;
        }

        // 设置精度
        // if (config_.precision == Precision::FP16) {
        //     if (builder->platformHasFastFp16()) {
        //         config->setFlag(nvinfer1::BuilderFlag::kFP16);
        //         FL_LOG_INFO("启用FP16精度");
        //     } else {
        //         FL_LOG_WARNING("平台不支持FP16，使用FP32精度");
        //     }
        // } else if (config_.precision == Precision::INT8) {
        //     if (builder->platformHasFastInt8()) {
        //         config->setFlag(nvinfer1::BuilderFlag::kINT8);
        //         FL_LOG_INFO("启用INT8精度");
        //     } else {
        //         FL_LOG_WARNING("平台不支持INT8，使用FP32精度");
        //     }
        // }

        // 构建引擎
        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
        if (!engine)
        {
            FL_LOG_ERROR("构建TensorRT引擎失败");
            return false;
        }

        // 创建运行时
        runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
        if (!runtime_)
        {
            FL_LOG_ERROR("创建TensorRT运行时失败");
            return false;
        }

        // 转移引擎所有权
        shared_engine_.reset(engine.release());

        FL_LOG_INFO("从ONNX构建TensorRT引擎成功");
        return true;
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

        if (extension == "onnx")
        {
            FL_LOG_INFO("从ONNX文件构建TensorRT引擎: %s", model_path.c_str());
            success = buildEngineFromOnnx(model_path);

            // 保存构建的引擎
            // if (success)
            // {
            //     std::string engine_path = model_path.substr(0, model_path.find_last_of('.')) + ".trt";
            //     FL_LOG_INFO("保存TensorRT引擎到文件: %s", engine_path.c_str());
            //     success = saveEngineToFile(engine_path);
            // }
        }
        else if (extension == "trt" || extension == "engine")
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

}