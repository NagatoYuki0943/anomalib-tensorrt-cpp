#include <iostream>
#include <string>
#include <array>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;

/**
 * https://blog.csdn.net/qq_44747572/article/details/121467657
 * https://zhuanlan.zhihu.com/p/513777076
 * https://github.com/mgmk2/onnxruntime-cpp-example/blob/main/cpp/main.cpp
 * https://github.com/tenglike1997/onnxruntime-projects/blob/master/Windows/onnx_mobilenet/ort_test/ort_test.cpp
 */


int main() {
    const wchar_t* model_path = L"D:/ai/code/test/onnx/resnet18.onnx";
    string image_path = "D:/ai/bus.jpg";

    cv::Mat image = cv::imread(image_path);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255,
        { 256, 256 },
        { 0, 0, 0 },
        true, false, CV_32F);

    // 获取可用的provider
    auto availableProviders = Ort::GetAvailableProviders();
    for (const auto& provider : availableProviders) {
        cout << provider << " ";
    }
    cout << endl;
    // TensorrtExecutionProvider
    // CUDAExecutionProvider
    // CPUExecutionProvider

    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::RunOptions runOptions;

    // 使用1个线程执行op,若想提升速度，增加线程数
    sessionOptions.SetIntraOpNumThreads(1);
    // ORT_ENABLE_ALL: 启用所有可能的优化
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    string device = "cuda";
    if (device == "cuda" || device == "tensorrt") {
        // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        // https://onnxruntime.ai/docs/api/c/struct_ort_c_u_d_a_provider_options.html
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = (size_t)2 * 1024 * 1024 * 1024; // 2GB
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        if (device == "tensorrt") {
            // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            // https://onnxruntime.ai/docs/api/c/struct_ort_tensor_r_t_provider_options.html
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = 0;
            trt_options.trt_max_workspace_size = (size_t)2 * 1024 * 1024 * 1024; // 2GB
            trt_options.trt_fp16_enable = 0;
            sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
        }
    }

    // create session
    Ort::Session session = Ort::Session(env, model_path, sessionOptions );

    /**
     * model info
     **/
     // 1. 获得模型有多少个输入和输出，一般是指对应网络层的数目, 如果是多输出网络，就会是对应输出的数目
    size_t input_nums = session.GetInputCount();
    size_t output_nums = session.GetOutputCount();
    printf_s("Number of inputs = %zu\n", input_nums); // Number of inputs = 1
    printf_s("Number of output = %zu\n", output_nums);// Number of output = 1

    // 2.获取输入输出name
    // 3.获取维度数量
    vector<const char*> input_node_names;                   // 输入节点名
    vector<Ort::AllocatedStringPtr> input_node_names_ptr;   // 输入节点名指针,保存它防止释放 https://github.com/microsoft/onnxruntime/issues/13651
    vector<vector<int64_t>> input_dims;                     // 输入形状
    for (int i = 0; i < input_nums; i++) {
        // 输入变量名
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        input_node_names_ptr.push_back(move(input_name));

        // 输入形状
        auto input_shape_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        input_dims.push_back(input_shape_info.GetShape());
    }

    vector<const char*> output_node_names;                  // 输出节点名
    vector<Ort::AllocatedStringPtr> output_node_names_ptr;  // 输入节点名指针
    vector<vector<int64_t>> output_dims;                    // 输出形状
    for (int i = 0; i < output_nums; i++) {
        // 输出变量名
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        output_node_names_ptr.push_back(move(output_name));

        // 输出形状
        auto output_shape_info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        output_dims.push_back(output_shape_info.GetShape());
    }

    for (int i = 0; i < input_nums; ++i) {
        cout << "input_dims: ";
        for (const auto j : input_dims[i]) {
            cout << j << " ";
        }
        cout << endl;
    }

    for (int i = 0; i < output_nums; ++i) {
        cout << "output_dims: ";
        for (const auto j : output_dims[i]) {
            cout << j << " ";
        }
        cout << endl;
    }

    // 4.申请内存空间
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 输入值
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims[0].data(), input_dims[0].size());
    // // 输出值
    // array<float, 1000> results_{}; //模型输出，注意和output_shape_对应
    // Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_dims.data(), output_dims.size());
    // // 同时传递输入和输出
    // session.Run(runOptions, input_node_names.data(), &input_tensor, input_nums, output_node_names.data(), &output_tensor, output_nums);
    // 只传递输入
    vector<Ort::Value> output_tensors;
    try {
        output_tensors = session.Run(runOptions, input_node_names.data(), &input_tensor, input_nums, output_node_names.data(), output_nums);
    }
    catch (Ort::Exception& e) {
        cout << e.what() << endl;
        return 1;
    }
     cv::Mat res = cv::Mat(cv::Size(256, 256),
          CV_32FC1, output_tensors[0].GetTensorMutableData<float>());
    //cv::normalize(res, res, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8UC3);
    //cout << res << endl;
    //cv::imshow("test", res);
    //cv::waitKey(0);


    return 0;
}

