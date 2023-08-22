#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // patchcore模型训练配置文件删除了center_crop
    // trtexec --onnx=model.onnx --saveEngine=model.engine 转换模型
    string model_path = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/model.engine";
    string meta_path  = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_path1 = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/001.png";
    string image_path2 = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/002.png";
    bool efficient_ad = true; // 是否使用efficient_ad模型
    int dynamic_batch_size = 1;

    // 创建推理器
    auto inference = Inference(model_path, meta_path, efficient_ad, dynamic_batch_size);

    // 读取一张图片,复制batch份,假装多张图片
    cv::Mat image = cv::imread(image_path);
    // std::vector<cv::Mat> images(dynamic_batch_size, image);
    std::vector<cv::Mat> images{ image};
    std::vector<Result> results = inference.dynamicBatchInfer(images);
    for (int i = 0; i < dynamic_batch_size; i++) {
        cout << results[i].score << endl;
        cv::resize(results[i].anomaly_map, results[i].anomaly_map, { 1500, 500 });
        cv::imshow(std::to_string(i), results[i].anomaly_map);
    }

    cv::waitKey(0);

    return 0;
}
