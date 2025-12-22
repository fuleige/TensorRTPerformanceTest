
#include <iostream>
#include "utils/ThreadSafeQueue.h"
#include <thread>
#include <string.h>
#include "utils/TimeRecoder.h"
#include "TensortRTExecuteEngine.h"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <atomic>
#include "utils/YoloUtils.h"

void ExampleYolo()
{
    const char *engine_file = "/root/yolo_onnx_models/yolo11m.static.batch1.engine";
    const char *image_path = "/root/My_project/DemoTensorRT/images/bus.jpg";

    cv::Mat image = cv::imread(image_path);

    fulei::TensorRTEngineManager engine_manager;
    fulei::EngineConfig config;
    // 如果是动态输入,则需要设置最大batch size
    config.max_batch_size = 64;
    config.input_nodes_shape["images"] = nvinfer1::Dims4(1, 3, 640, 640);

    engine_manager.initialize(engine_file, config);

    // 1. 预处理图像, 拷贝数据并执行推理
    float scale = 1.0f;
    fulei::ProcessImageToPtr(image, (float *)engine_manager.getInputAddress("images"), 640, 640, scale);

    std::cout << "Input data size: " << image.total() * image.elemSize() << std::endl;
    engine_manager.executeInference(1);

    // 3. 获取输出地址并解析结果
    char *output_address = (char *)engine_manager.getOutputAddress("output0");
    // 解析结果
    // 每个检测框有6个值: x, y, w, h, conf, class_id
    size_t count = 0;
    for (size_t i = 0; i < 300; i++)
    {
        float x1 = ((float *)output_address)[i * 6 + 0] / scale;
        float y1 = ((float *)output_address)[i * 6 + 1] / scale;
        float x2 = ((float *)output_address)[i * 6 + 2] / scale;
        float y2 = ((float *)output_address)[i * 6 + 3] / scale;
        float conf = ((float *)output_address)[i * 6 + 4];
        float class_id = ((float *)output_address)[i * 6 + 5];

        if (conf < 0.2)
        {
            break;
        }

        count++;
        std::cout << "[" << i << "] x: " << x2 << ", y: " << y1 << ", w: " << x2 - x1 << ", h: " << y2 - y1 << ", conf: " << conf << ", class_id: " << class_id << std::endl;
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);
        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, std::to_string(int(class_id)) + ":" + std::to_string(conf), cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    std::cout << "Total count: " << count << std::endl;
    std::cout << "Result saved: " << cv::imwrite("bus_result.jpg", image) << std::endl;
}

void SingleInferenceMultiThread()
{
    const char *static_batch1_engine_file = "/root/yolo_onnx_models/yolo11m.static.batch1.engine";
    const char *image_path = "/root/My_project/DemoTensorRT/images/bus.jpg";
    cv::Mat image = cv::imread(image_path);

    const std::string engine_file = static_batch1_engine_file;

    fulei::TensorRTEngineManager engine_manager;
    fulei::EngineConfig config;
    // 如果是动态输入,则需要设置最大batch size
    config.max_batch_size = 1;
    config.input_nodes_shape["images"] = nvinfer1::Dims4(1, 3, 640, 640);

    engine_manager.initialize(engine_file, config);

    size_t total_size = 1024 * 64;
    size_t print_size = total_size / 10;

    fulei::TimeRecoder recoder;

    std::atomic_uint64_t counter{0};

    // 线程测试
    #pragma omp parallel for num_threads(64) schedule(dynamic, 1)
    for (size_t i = 0; i < total_size; i++)
    {
        // 预处理图像
        float scale = 1.0f;
        fulei::ProcessImageToPtr(image, (float *)engine_manager.getInputAddress("images"), 640, 640, scale);
        // 执行推理
        engine_manager.executeInference(1);

        // 进度打印
        uint64_t counter_val = ++counter;
        if (counter_val % print_size == 0)
        {
            std::cout << "Progress: [" << counter_val << "/" << total_size << "] " << ((double)counter_val / total_size * 100) << "%,cost=" << recoder.cost_ms() << "ms" << std::endl;
        }
    }

    size_t time_ms = recoder.cost_ms();

    std::cout << "total_size=" << total_size << ", cost:" << time_ms << "ms" << std::endl;
    std::cout << "single sample cost:" << ((double)time_ms / total_size) << "ms" << std::endl;
}

void BatchIntferanceMultiThraed()
{
    const char *engine_file = "/root/yolo_onnx_models/yolo11n.dynamic.batch64.engine";
    const char *image_path = "/root/My_project/DemoTensorRT/images/bus.jpg";
    cv::Mat image = cv::imread(image_path);
    const size_t batch_size = 64;

    // 是否多线程测试
    size_t num_test_thread = 6;
    size_t preprocess_num_thread = 4;

    fulei::TensorRTEngineManager engine_manager;
    fulei::EngineConfig config;
    // 如果是动态输入,则需要设置最大batch size
    config.max_batch_size = batch_size;
    config.input_nodes_shape["images"] = nvinfer1::Dims4(1, 3, 640, 640);

    size_t step = 640 * 640 * 3;

    std::atomic_uint64_t counter{0};

    engine_manager.initialize(engine_file, config);

    size_t total_size = 65536;
    size_t loop = total_size / batch_size;
    size_t print_loop = loop / 10;

    fulei::TimeRecoder recoder;

    #pragma omp parallel for num_threads(num_test_thread) schedule(dynamic, 1)
    for (size_t i = 0; i < loop; i++)
    {
        float *input_images_ptr = (float *)engine_manager.getInputAddress("images");
        // 预处理
        #pragma omp parallel for num_threads(preprocess_num_thread) schedule(dynamic, 1)
        for (size_t j = 0; j < batch_size; j++)
        {
            // 预处理图像
            float scale = 1.0f;
            fulei::ProcessImageToPtr(image, input_images_ptr + j * step, 640, 640, scale);
        }

        // 执行推理
        engine_manager.executeInference(batch_size);

        // 定期打印内容
        uint64_t counter_val = ++counter;
        if (counter_val % print_loop == 0)
        {
            std::cout << "Progress: [" << counter_val << "/" << loop << "] " << ((double)counter_val / loop * 100) << "%,cost=" << recoder.cost_ms() << "ms" << std::endl;
        }
    }

    size_t time_ms = recoder.cost_ms();

    std::cout << "total_size=" << total_size << ", cost:" << time_ms << "ms" << std::endl;
    std::cout << "single sample cost:" << ((double)time_ms / total_size) << "ms" << std::endl;
}

int main()
{
    SingleInferenceMultiThread();
    // BatchIntferanceMultiThraed();
    return 0;
}