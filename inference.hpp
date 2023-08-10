#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include "logger.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <math.h>
#include <numeric>
#include <Windows.h>
#include "utils.h"

using namespace std;


inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}


inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}


class Inference {
private:
    bool efficient_ad;                      // 是否使用efficient_ad模型
    MetaData meta{};                        // 超参数
    nvinfer1::ICudaEngine* engine;          // model
    nvinfer1::IExecutionContext* context;   // contenx
    cudaStream_t stream;                    // async stream
    void* cudaBuffers[4];                   // 分配显存空间
    std::vector<int> bufferSize;            // 每个显存空间占用大小
    int output_nums;                        // 模型输出个数
    std::vector<float*> outputs;            // 分配输出内存空间

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param efficient_ad  是否使用efficient_ad模型
     */
    Inference(string& model_path, string& meta_path, bool efficient_ad = false) {
        this->efficient_ad = efficient_ad;
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->get_model(model_path);
        // 4.模型预热
        this->warm_up();
    }

    /**
     * get onnx model
     * @param model_path    模型路径
     */
    void get_model(string& model_path) {
        /******************** load engine ********************/
        string cached_engine;
        std::fstream file;
        std::cout << "loading filename from:" << model_path << std::endl;
        file.open(model_path, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            std::cout << "read file error: " << model_path << std::endl;
            cached_engine = "";
        }
        while (file.peek() != EOF) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            cached_engine.append(buffer.str());
        }
        file.close();

        nvinfer1::IRuntime* trtRuntime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
        initLibNvInferPlugins(&sample::gLogger, "");
        this->engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
        std::cout << "deserialize done" << std::endl;
        /******************** load engine ********************/

        /********************** binding **********************/
        this->context = this->engine->createExecutionContext();
        assert(this->context != nullptr);

        //get buffers
        int nbBindings = this->engine->getNbBindings();
        assert(nbBindings <= 4); // 大多数模型是1个输入1个输出,patchcore是1个输入2个输出,efficiented是1个输入3个输出,所以nbBindings最大为4
        this->bufferSize.resize(nbBindings);
        this->output_nums = nbBindings - 1;  // 假设只有1个输入

        for (int i = 0; i < nbBindings; i++) {
            string name = this->engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = this->engine->getTensorIOMode(name.c_str());
            // cout << "mode: " << int(mode) << endl; // 0:input or output  1:input  2:output
            nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
            nvinfer1::Dims dims = this->engine->getTensorShape(name.c_str());
            int totalSize = volume(dims) * getElementSize(dtype);
            bufferSize[i] = totalSize;
            cudaMalloc(&this->cudaBuffers[i], totalSize); // 分配显存空间

            if (int(mode) == 2) {                         // 分配输出内存空间
                int outSize = int(totalSize / sizeof(float));
                float* output = new float[outSize];
                this->outputs.push_back(output);
            }
        }
        /********************** binding **********************/

        // get stream
        cudaStreamCreate(&this->stream);
    }


    /**
     * 模型预热
     */
    void warm_up() {
        // 输入数据
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        // this->infer(input);
    }

    ///**
    // * 推理单张图片
    // * @param image 原始图片
    // * @return      标准化的并所放到原图热力图和得分
    // */
    Result infer(cv::Mat & image) {
        // 1.保存图片原始高宽
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.图片预处理
        cv::Mat resized_image;
        resized_image = pre_process(image, meta, this->efficient_ad);
        resized_image = cv::dnn::blobFromImage(resized_image);

        // 3.infer
        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        cudaMemcpy(this->cudaBuffers[0], resized_image.ptr<float>(), this->bufferSize[0], cudaMemcpyHostToDevice);
        // cudaMemcpyAsync(this->cudaBuffers[0], image.ptr<float>(), this->bufferSize[0], cudaMemcpyHostToDevice, this->stream);  // 异步没有把数据移动上去,很奇怪
        context->executeV2(this->cudaBuffers);
        for (size_t i = 1; i <= this->output_nums; i++) {
            cudaMemcpy(this->outputs[i-1], this->cudaBuffers[i], this->bufferSize[i], cudaMemcpyDeviceToHost);
            // cudaMemcpyAsync(this->outputs[i-1], this->cudaBuffers[i], this->bufferSize[i], cudaMemcpyDeviceToHost, this->stream);
        }
        // cudaStreamSynchronize(stream);

        // 4.将热力图转换为Mat
        cv::Mat anomaly_map;
        cv::Mat pred_score;
        if (this->output_nums == 1) {
            anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, this->outputs[0]);
            double _, maxValue;    // 最大值，最小值
            cv::minMaxLoc(anomaly_map, &_, &maxValue);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);
        }
        else if (this->output_nums == 2) {
            // patchcore的输出[0]为得分,[1]为map
            anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, this->outputs[1]);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, this->outputs[0]);  // {1}
        }
        else if (this->output_nums == 3) {
            // efficient_ad有3个输出结果, [2]才是anomaly_map
            anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, this->outputs[2]);
            double _, maxValue;    // 最大值，最小值
            cv::minMaxLoc(anomaly_map, &_, &maxValue);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);
        }
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 5.后处理:标准化,缩放到原图
        vector<cv::Mat> result = post_process(anomaly_map, pred_score, this->meta);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 6.返回结果
        return Result{ anomaly_map, score };
    }

    /**
     * 推理单张图片
     * @param image 原始图片
     * @return      标准化的并所放到原图热力图和得分
     */
    cv::Mat single(string& image_path, string& save_dir) {
        // 1.读取图片
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 2.推理单张图片
        Result result = this->infer(image);
        cout << "score: " << result.score << endl;

        // 3.生成其他图片(mask,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;

        // 4.保存显示图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        // 拼接图片
        cv::Mat res;
        cv::hconcat(images, res);
        saveScoreAndImages(result.score, res, image_path, save_dir);

        return res;
    }

    /**
     * 多张图片推理
     * @param image_dir 图片文件夹路径
     * @param save_dir  保存路径
     */
    void multi(string& image_dir, string& save_dir) {
        // 1.读取全部图片路径
        vector<cv::String> paths = getImagePaths(image_dir);

        vector<float> times;
        for (auto& image_path : paths) {
            // 2.读取单张图片
            cv::Mat image = readImage(image_path);

            // time
            auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            // 3.推理单张图片
            Result result = this->infer(image);
            cout << "score: " << result.score << endl;

            // 4.图片生成其他图片(mask,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
            // time
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cout << "infer time: " << end - start << " ms" << endl;
            times.push_back(end - start);

            // 5.保存图片
            // 将mask转化为3通道,不然没法拼接图片
            cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
            // 拼接图片
            cv::Mat res;
            cv::hconcat(images, res);
            saveScoreAndImages(result.score, res, image_path, save_dir);
        }

        // 6.统计数据
        double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
        double avgValue = sumValue / times.size();                   // 求均值
        cout << "avg infer time: " << avgValue << " ms" << endl;
    }
};