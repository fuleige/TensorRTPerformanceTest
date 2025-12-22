
#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>
#include "utils/log.h"
#include <map>
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>


namespace fulei
{
    struct EngineConfig
    {

        // onnx文件有可能batch_size维度是动态的-1,可以预先指定最大batch_size
        size_t max_batch_size{64};

        // 注意需要考虑预先分配内存重复利用问题, 所以最好处理batch_size, 其它的所有的维度都是固定的
        // 如果非固定的也最好有有个最大值限制
        // 可以利用最大值预先分配内存, 重复分配内存会降低性能
        // 如果输入和输出某些维度有 -1 这种问题, 需要在配置中写明形状, shape的必须为 [batch_size, xxx, ...] (batch_size 在这里可以随意设置)
        std::map<std::string, nvinfer1::Dims> input_nodes_shape;
        std::map<std::string, nvinfer1::Dims> output_nodes_shape;

        // 数据类型大小
        // 如果是float32就是4,float16就是2,int8/uint8就是1
        int data_type_size{4};
    };

    class ExecutionContext
    {
    private:
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
        cudaStream_t stream_;

        // 输入输出主机的 pinned 内存指针
        std::vector<std::string> input_names;
        std::vector<char *> input_host_ptrs_;
        std::vector<char *> output_host_ptrs_;
        std::vector<nvinfer1::Dims> input_host_sizes_;

        // 输入输出显存指针
        std::vector<std::string> output_names;
        std::vector<char *> input_gpu_ptrs_;
        std::vector<char *> output_gpu_ptrs_;
        std::vector<nvinfer1::Dims> output_host_sizes_;

        size_t data_type_size_{4};

    public:
        explicit ExecutionContext(nvinfer1::ICudaEngine *engine, const EngineConfig &config);
        ~ExecutionContext();

        // 禁用拷贝构造和赋值
        ExecutionContext(const ExecutionContext &) = delete;
        ExecutionContext &operator=(const ExecutionContext &) = delete;

        // 获取CUDA流
        cudaStream_t getStream() const { return stream_; }

        // 获取执行上下文
        nvinfer1::IExecutionContext *getContext() const { return context_.get(); }

        // success
        bool isSuccess() const { return context_ != nullptr; }

        // 获取输入输出地址
        const char *getInputAddress(const std::string &input_name);
        const char *getOutputAddress(const std::string &output_name);

        // 执行推理
        void executeInference(size_t batch_size=1);
    };

    class TensorRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char *msg) noexcept override;

        nvinfer1::ILogger::Severity getLogLevel() const;
        void setLogLevel(nvinfer1::ILogger::Severity level);

    private:
        nvinfer1::ILogger::Severity log_level_{nvinfer1::ILogger::Severity::kINFO};
    };

    class TensorRTEngineManager
    {
    private:
        std::shared_ptr<nvinfer1::ICudaEngine> shared_engine_; // 共享引擎
        std::shared_ptr<nvinfer1::IRuntime> runtime_;

        // 线程本地存储
        thread_local static std::unique_ptr<ExecutionContext> thread_context_;

        // 配置信息
        EngineConfig config_;
        bool is_initialized_;

        // 日志记录器
        TensorRTLogger trt_logger_;

        // 导入模型文件,onnx/engine文件
        bool buildEngineFromOnnx(const std::string &onnx_path);
        bool loadEngineFromFile(const std::string &engine_path);

        ExecutionContext *getThreadContext();

    public:
        TensorRTEngineManager() : is_initialized_(false) {}
        ~TensorRTEngineManager() = default;

        // 禁用拷贝构造和赋值
        TensorRTEngineManager(const TensorRTEngineManager &) = delete;
        TensorRTEngineManager &operator=(const TensorRTEngineManager &) = delete;

        /**
         * 初始化引擎管理器
         * @param model_path 模型文件路径
         * @param config 引擎配置
         * @return 是否初始化成功
         */
        bool initialize(const std::string &model_path, const EngineConfig &config = EngineConfig());

        /**
         * 检查引擎是否已初始化
         */
        bool isInitialized() const { return is_initialized_; }

        /**
         * 打印模型详情
         *  打印模型的输入输出节点名称、形状、数据类型等信息
         */
        void ModelDetail();

        /**
         * 获取引擎配置
         */
        const EngineConfig &getConfig() const { return config_; }

        const char *getInputAddress(const std::string &input_name);

        const char *getOutputAddress(const std::string &output_name);

        void executeInference(size_t batch_size=1);
    };

}