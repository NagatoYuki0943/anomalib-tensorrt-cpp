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
    return accumulate(d.d, d.d + d.nbDims, 1, multiplies<int64_t>());
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
    throw runtime_error("Invalid DataType.");
    return 0;
}


class Inference {
private:
    bool efficient_ad;                      // 是否使用efficient_ad模型
    bool dynamic_batch;                     // 是否使用dynamic_batch
    int min_batch = 1;                      // 最小支持的dim
    int max_batch = 1;                      // 最大支持的dim
    MetaData meta{};                        // 超参数
    nvinfer1::IRuntime* trtRuntime;         // runtime
    nvinfer1::ICudaEngine* engine;          // model
    nvinfer1::IExecutionContext* context;   // contenx
    cudaStream_t stream;                    // async stream
    void* cudaBuffers[4];                   // 分配显存空间
    vector<int> bufferSize;                 // 每个显存空间占用大小
    int output_nums;                        // 模型输出个数
    vector<float*> outputs;                 // 分配输出内存空间

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param efficient_ad  是否使用efficient_ad模型
     * @param dynamic_batch dynamic_batch模型是否使用dynamic_batch
     */
    Inference(string& model_path, string& meta_path, bool efficient_ad = false, bool dynamic_batch = false) {
        this->efficient_ad = efficient_ad;
        this->dynamic_batch = dynamic_batch;
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->get_model(model_path);
        // 4.模型预热
        this->warm_up();
    }

    ~Inference() {
        // 析构顺序很重要
        // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#ab3ace89a0eb08cd7e4b4cba7bedac5a2
        delete this->context;
        delete this->engine;
        delete this->trtRuntime;

        for (float* fpoint : this->outputs) {
            delete[] fpoint;
        }
    }

    /**
     * get tensorrt model
     * @param model_path    模型路径
     */
    void get_model(string& model_path) {
        int trt_version = nvinfer1::kNV_TENSORRT_VERSION_IMPL;
        cout << "trt_version = " << trt_version << endl; // 8601

        /******************** load engine ********************/
        string cached_engine;
        fstream file;
        cout << "loading filename from:" << model_path << endl;
        file.open(model_path, ios::binary | ios::in);
        if (!file.is_open()) {
            cout << "read file error: " << model_path << endl;
            cached_engine = "";
        }
        while (file.peek() != EOF) {
            stringstream buffer;
            buffer << file.rdbuf();
            cached_engine.append(buffer.str());
        }
        file.close();

        this->trtRuntime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
        initLibNvInferPlugins(&sample::gLogger, "");
        this->engine = this->trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size());
        assert(this->engine != nullptr);
        cout << "deserialize done" << endl;
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
            const char* name;
            int mode;
            nvinfer1::DataType dtype;
            nvinfer1::Dims dims;
            int totalSize;

            if (trt_version < 8500) {
                mode = int(this->engine->bindingIsInput(i));
                name = this->engine->getBindingName(i);
                dtype = this->engine->getBindingDataType(i);
                dims = this->context->getBindingDimensions(i);
                // dynamic batch
                if ((*dims.d == -1) && mode) {
                    nvinfer1::Dims minDims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                    nvinfer1::Dims optDims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
                    nvinfer1::Dims maxDims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                    if (this->dynamic_batch) {
                        this->min_batch = minDims.d[0];
                        this->max_batch = maxDims.d[0];
                        // 设置为最大batch
                        context->setBindingDimensions(i, maxDims);
                    }
                    else {
                        // 设置为batch为1
                        context->setBindingDimensions(i, nvinfer1::Dims4(1, maxDims.d[1], maxDims.d[2], maxDims.d[3]));
                    }
                    dims = this->context->getBindingDimensions(i);
                }

                totalSize = volume(dims) * getElementSize(dtype);
                this->bufferSize[i] = totalSize;
                cudaMalloc(&this->cudaBuffers[i], totalSize);   // 分配显存空间

                if (!mode) {                                    // 分配输出内存空间
                    int outSize = int(totalSize / sizeof(float));
                    float* output = new float[outSize];
                    this->outputs.push_back(output);
                }
            }
            else {
                name = this->engine->getIOTensorName(i);
                mode = int(this->engine->getTensorIOMode(name));
                // cout << "mode: " << mode << endl; // 0:input or output  1:input  2:output
                dtype = this->engine->getTensorDataType(name);
                dims = this->context->getTensorShape(name);

                // dynamic batch
                if ((*dims.d == -1) && (mode == 1)) {
                    nvinfer1::Dims minDims = this->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
                    nvinfer1::Dims optDims = this->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
                    nvinfer1::Dims maxDims = this->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
                    if (this->dynamic_batch) {
                        this->min_batch = minDims.d[0];
                        this->max_batch = maxDims.d[0];
                        // 设置为最大batch
                        context->setInputShape(name, maxDims);
                    }
                    else {
                        // 设置为batch为1
                        context->setInputShape(name, nvinfer1::Dims4(1, maxDims.d[1], maxDims.d[2], maxDims.d[3]));
                    }
                    dims = this->context->getTensorShape(name);
                }

                totalSize = volume(dims) * getElementSize(dtype);
                this->bufferSize[i] = totalSize;
                cudaMalloc(&this->cudaBuffers[i], totalSize);   // 分配显存空间

                if (mode == 2) {                                // 分配输出内存空间
                    int outSize = int(totalSize / sizeof(float));
                    float* output = new float[outSize];
                    this->outputs.push_back(output);
                }
            }

            fprintf(stderr, "name: %s, mode: %d, dims: [%d, %d, %d, %d], totalSize: %d Byte\n", name, mode, dims.d[0], dims.d[1], dims.d[2], dims.d[3], totalSize);
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
        cv::Mat image = cv::Mat(size, CV_8UC3, color);
        if (this->max_batch == 1) {
            this->infer(image);
        }
        else {
            vector<cv::Mat> images(this->max_batch, image);
            this->dynamicBatchInfer(images);
        }
        cout << "warm up finish" << endl;
    }

    ///**
    // * 推理单张图片
    // * @param image 原始图片
    // * @return      标准化的并所放到原图热力图和得分
    // */
    Result infer(cv::Mat & image) {
        // 1.保存图片原始高宽
        this->meta.image_size[0] = image.size().height;
        this->meta.image_size[1] = image.size().width;

        // 2.图片预处理
        cv::Mat resized_image = pre_process(image, this->meta, this->efficient_ad);
        cv::Mat blob = cv::dnn::blobFromImage(resized_image);

        // 3.infer
        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        cudaMemcpy(this->cudaBuffers[0], blob.ptr<float>(), this->bufferSize[0], cudaMemcpyHostToDevice);
        // cudaMemcpyAsync(this->cudaBuffers[0], image.ptr<float>(), this->bufferSize[0], cudaMemcpyHostToDevice, this->stream);  // 异步没有把数据移动上去,很奇怪
        this->context->executeV2(this->cudaBuffers);
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
        vector<cv::Mat> post_mat = post_process(anomaly_map, pred_score, this->meta);
        anomaly_map = post_mat[0];
        float score = post_mat[1].at<float>(0, 0);

        // 6.返回结果
        return Result{ anomaly_map, score };
    }

    /**
     * 单张图片推理
     * @param image    RGB图片
     * @return      标准化的并所放到原图热力图和得分
     */
    Result single(cv::Mat& image) {
        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 1.推理单张图片
        Result result = this->infer(image);
        cout << "score: " << result.score << endl;

        // 2.生成其他图片(mask,mask抠图,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;

        // 3.保存显示图片
        // 拼接图片
        cv::Mat res;
        cv::hconcat(images, res);

        return Result{ res, result.score };
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

            // 4.生成其他图片(mask,mask抠图,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
            // time
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cout << "infer time: " << end - start << " ms" << endl;
            times.push_back(end - start);

            // 5.保存图片
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


    /**
     * 动态batch推理,要保证输入图片的大小都相同
     * 图片前后处理是顺序进行的,推理是batch推理
     *
     * @param image 原始图片
     * @return      标准化的并所放到原图热力图和得分
     */
    vector<Result> dynamicBatchInfer(vector<cv::Mat> images) {
        int images_num = images.size();
        assert(images_num >= this->min_batch && images_num <= this->max_batch);

        // 1.保存图片原始高宽,使用第一张图片,假设图片大小都一致
        this->meta.image_size[0] = images[0].size().height;
        this->meta.image_size[1] = images[0].size().width;

        // 2.图片预处理,图片顺序处理
        vector<cv::Mat> resized_images;
        for (cv::Mat image : images) {
            resized_images.push_back(pre_process(image, this->meta, this->efficient_ad));
        }
        cv::Mat blob = cv::dnn::blobFromImages(resized_images);

        // 3.infer
        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        cudaMemcpy(this->cudaBuffers[0], blob.ptr<float>(), this->bufferSize[0] / this->max_batch * images_num, cudaMemcpyHostToDevice);
        context->executeV2(this->cudaBuffers);
        for (size_t i = 1; i <= this->output_nums; i++) {
            cudaMemcpy(this->outputs[i - 1], this->cudaBuffers[i], this->bufferSize[i] / this->max_batch * images_num, cudaMemcpyDeviceToHost);
        }

        // 4.获取结果
        vector<cv::Mat> anomaly_maps;
        vector<cv::Mat> pred_scores;
        int total_infer_length = this->meta.infer_size[0] * this->meta.infer_size[1] * images_num;
        int infer_length = this->meta.infer_size[0] * this->meta.infer_size[1];

        // vector<float*> temp_results(images_num, new float[infer_length]); // 这样初始化会导致多个结果内存地址相同
        vector<float*> temp_results;
        for (int i = 0; i < images_num; i++) {
            temp_results.push_back(new float[infer_length]);
        }
        if (this->output_nums == 1) {
            for (int i = 0; i < total_infer_length; i++) {
                temp_results[i / infer_length][i % infer_length] = this->outputs[0][i];
            }
            for (int i = 0; i < images_num; i++) {
                cv::Mat temp_anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, temp_results[i]);
                anomaly_maps.push_back(temp_anomaly_map);
                double _, maxValue;    // 最大值，最小值
                cv::minMaxLoc(temp_anomaly_map, &_, &maxValue);
                pred_scores.push_back(cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue));
            }
        }
        else if (this->output_nums == 2) {
            // patchcore的输出[0]为得分,[1]为map
            for (int i = 0; i < total_infer_length; i++) {
                temp_results[i / infer_length][i % infer_length] = this->outputs[1][i];
            }
            for (int i = 0; i < images_num; i++) {
                cv::Mat temp_anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, temp_results[i]);
                anomaly_maps.push_back(temp_anomaly_map);
            }
            cv::Mat pred_scores_ = cv::Mat(cv::Size(1, 1), CV_32FC(images_num), this->outputs[0]);
            cv::split(pred_scores_, pred_scores);
        }
        else if (this->output_nums == 3) {
            // efficient_ad有3个输出结果, [2]才是anomaly_map
            for (int i = 0; i < total_infer_length; i++) {
                temp_results[i / infer_length][i % infer_length] = this->outputs[2][i];
            }
            for (int i = 0; i < images_num; i++) {
                cv::Mat temp_anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, temp_results[i]);
                anomaly_maps.push_back(temp_anomaly_map);
                double _, maxValue;    // 最大值，最小值
                cv::minMaxLoc(temp_anomaly_map, &_, &maxValue);
                pred_scores.push_back(cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue));
            }
        }

        // 5.后处理,每张图片单独处理
        vector<Result> results;
        for (int i = 0; i < images_num; i++) {
            // 后处理
            vector<cv::Mat> post_mat = post_process(anomaly_maps[i], pred_scores[i], this->meta);
            cv::Mat image = images[i];
            cv::Mat anomaly_map = post_mat[0];
            float score = post_mat[1].at<float>(0, 0);

            // 生成其他图片(mask,mask抠图,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, anomaly_map, score);

            // 拼接图片
            cv::Mat res;
            cv::hconcat(images, res);
            results.push_back(Result{ res , score });
        }

        for (float* fpoint : temp_results) {
            delete[] fpoint;
        }

        // 6.返回结果
        return results;
    }
};