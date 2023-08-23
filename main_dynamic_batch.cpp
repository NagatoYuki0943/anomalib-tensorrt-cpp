#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // patchcore模型训练配置文件调整center_crop为 `center_crop: null`
    // trtexec --onnx=model.onnx --saveEngine=model.engine 转换模型
    // 动态batch,model.onnx的batch为动态的                             input为输入名字, 1, 4, 8要手动指定
    // trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=input:1x3x256x256 --optShapes=input:4x3x256x256 --maxShapes=input:8x3x256x256
    string model_path = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/model.engine";
    string meta_path  = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_dir = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    bool efficient_ad = true;   // 是否使用efficient_ad模型
    int dynamic_batch_size = 4; // 显式指定batch,要在最小和最大batch之间

    // 创建推理器
    auto inference = Inference(model_path, meta_path, efficient_ad, dynamic_batch_size);

    // 读取全部图片路径
    vector<cv::String> paths = getImagePaths(image_dir);
    // batch为几就输入几张图片
    vector<cv::Mat> images;
    for (int i = 0; i < dynamic_batch_size; i++) {
        cv::Mat image = cv::imread(paths[i]);
        cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
        images.push_back(image);
    }

    // 推理
    vector<Result> results = inference.dynamicBatchInfer(images);

    // 查看结果
    for (int i = 0; i < dynamic_batch_size; i++) {
        cout << results[i].score << endl;
        cv::resize(results[i].anomaly_map, results[i].anomaly_map, { 1500, 500 });
        cv::imshow(std::to_string(i), results[i].anomaly_map);
    }

    cv::waitKey(0);

    return 0;
}
