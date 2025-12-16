#pragma once
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fulei
{

    void ProcessImageToPtr(const cv::Mat &image, float *out_data, int input_width, int input_height, float &scale)
    {
        // 1. 缩放图像,保持比例
        scale = std::min(input_width * 1.0 / image.cols, input_height * 1.0 / image.rows);
        int new_width = cvRound(image.cols * scale);
        int new_height = cvRound(image.rows * scale);

        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(new_width, new_height));

        cv::Mat padded_image = cv::Mat::zeros(input_height, input_width, CV_8UC3);
        resized_image.copyTo(padded_image(cv::Rect(0, 0, new_width, new_height)));
        cv::imwrite("padded_image.jpg", padded_image);

        // 2. 三个处理, 可以放在一起做
        // 2.1 归一化数据
        // 2.2 数据转置 [H,W,C] -> [C,H,W]
        // 2.3 注意原始数据是BRG的顺序,但是模型输入是RBG格式
        size_t hw = input_width * input_height;
        uint8_t *src_ptr = padded_image.data;
        float *r_buffer = out_data;
        float *g_buffer = r_buffer + hw;
        float *b_buffer = g_buffer + hw;
        for (size_t i = 0; i < hw; i++)
        {
            b_buffer[i] = src_ptr[i * 3 + 0] / 255.0f;
            g_buffer[i] = src_ptr[i * 3 + 1] / 255.0f;
            r_buffer[i] = src_ptr[i * 3 + 2] / 255.0f;
        }
    }

    std::vector<float> ProcessImage(const cv::Mat &image, int input_width, int input_height, float &scale)
    {
        std::vector<float> input_data(input_width * input_height * 3, 0.0f);
        ProcessImageToPtr(image, input_data.data(), input_width, input_height, scale);
        return input_data;
    }

    void ProcessImageRawBufferToPtr(const char *image_buffer, size_t buffer_size, float *out_data, int input_width, int input_height, float &scale)
    {
        cv::Mat image = cv::imdecode(cv::Mat(1, buffer_size, CV_8UC1, const_cast<char *>(image_buffer)), cv::IMREAD_COLOR);
        ProcessImageToPtr(image, out_data, input_width, input_height, scale);
    }

    std::vector<float> ProcessImageRawBuffer(const char *image_buffer, size_t buffer_size, int input_width, int input_height, float &scale)
    {
        cv::Mat image = cv::imdecode(cv::Mat(1, buffer_size, CV_8UC1, const_cast<char *>(image_buffer)), cv::IMREAD_COLOR);
        return ProcessImage(image, input_width, input_height, scale);
    }

    void ProcessImagePathToPtr(const std::string &image_path, float *out_data, int input_width, int input_height, float &scale)
    {
        cv::Mat image = cv::imread(image_path);
        ProcessImageToPtr(image, out_data, input_width, input_height, scale);
    }

    std::vector<float> ProcessImagePath(const std::string &image_path, int input_width, int input_height, float &scale)
    {
        cv::Mat image = cv::imread(image_path);
        return ProcessImage(image, input_width, input_height, scale);
    }

} // namespace fulei