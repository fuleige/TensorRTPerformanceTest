
#include <iostream>
#include "utils/ThreadSafeQueue.h"
#include <thread>
#include <string.h>
#include "utils/TimeRecoder.h"
#include "TensortRTExecuteEngine.h"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "utils/YoloUtils.h"

void DetectImage()
{
    const char *engine_file = "/root/yolo_onnx_models/yolo11m.engine";
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

void PerformanceTest()
{
    const char *engine_file = "/root/yolo_onnx_models/yolo11m.engine";

    fulei::TensorRTEngineManager engine_manager;
    fulei::EngineConfig config;
    // 如果是动态输入,则需要设置最大batch size
    config.max_batch_size = 64;
    config.input_nodes_shape["images"] = nvinfer1::Dims4(1, 3, 640, 640);

    engine_manager.initialize(engine_file, config);

    fulei::TimeRecoder recorder;

    for (size_t i = 0; i < 1024; i++)
    {
        recorder.reset();
        engine_manager.executeInference(64);
        std::cout << "[" << i << "] Infer time: " << recorder.cost_ms() << " ms" << std::endl;
    }
}

int main()
{
    DetectImage();
    PerformanceTest();
    return 0;
}