# 说明

> 适用于anomalib导出的onnx格式的模型
>
> 测试了patchcore,fastflow,efficient_ad模型

```yaml
# 模型配置文件中设置为onnx,导出openvino会导出onnx
optimization:
  export_mode: onnx # options: torch, onnx, openvino
```

> 使用 `trtexec` 转换模型

```shell
trtexec --onnx=model.onnx --saveEngine=model.engine

# 动态batch,model.onnx的batch为动态的                             input为输入名字, 1, 4, 8要手动指定
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=input:1x3x256x256 --optShapes=input:4x3x256x256 --maxShapes=input:8x3x256x256
```

# 其他推理方式

> [anomalib-onnxruntime-cpp](https://github.com/NagatoYuki0943/anomalib-onnxruntime-cpp)
>
> [anomalib-openvino-cpp](https://github.com/NagatoYuki0943/anomalib-openvino-cpp)
>
> [anomalib-tensorrt-cpp](https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp)

# example

## 固定batch

```C++
#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // patchcore模型训练配置文件调整center_crop为 `center_crop: null`
    // trtexec --onnx=model.onnx --saveEngine=model.engine 转换模型
    string model_path = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/model.engine";
    string meta_path  = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-tensorrt-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    bool efficient_ad = true; // 是否使用efficient_ad模型

    // 创建推理器
    auto inference = Inference(model_path, meta_path, efficient_ad);

    // 单张图片推理
    cv::Mat image = readImage(image_path);
    Result result = inference.single(image);
    saveScoreAndImages(result.score, result.anomaly_map, image_path, save_dir);
    cv::resize(result.anomaly_map, result.anomaly_map, { 1500, 500 });
    cv::imshow("result", result.anomaly_map);
    cv::waitKey(0);

    // 多张图片推理
    inference.multi(image_dir, save_dir);
    return 0;
}
```

## 动态batch

### 前置条件,需要导出的onnx的batch为动态的

1. 方法1(需要重新训练或导出onnx)

> ​	在导出代码中添加一行https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/deploy/export.py#L155
>
> ```python
>     torch.onnx.export(
>         model.model,
>         torch.zeros((1, 3, *input_size)).to(model.device),
>         str(onnx_path),
>         opset_version=11,
>         input_names=["input"],
>         output_names=["output"],
>         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} # add this line to support dynamic batch
>     )
> ```

2. 方法2(不需要重新训练模型,但不保证成功)

> 使用 [onnx-modifier](https://github.com/ZhangGe6/onnx-modifier) 将模型的batch调整为动态的

### 导出engine

```sh
# 动态batch,model.onnx的batch为动态的                             input为输入名字, 1, 4, 8要手动指定,256为输出尺寸
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=input:1x3x256x256 --optShapes=input:4x3x256x256 --maxShapes=input:8x3x256x256
```

### 运行

> 需要显式指定 `dynamic_batch_size`，范围在设定的`minShapes`和`maxShapes`之间

```c++
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
```



# 下载tensorrt和opencv

> 下载安装cuda,cudnn,tensorrt
>
> [CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)
>
> [cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)
>
> [NVIDIA Developer Program Membership Required | NVIDIA Developer](https://developer.nvidia.com/nvidia-tensorrt-download)
>
> 测试tensorrt版本:8.6.1
> 

> https://opencv.org

## 配置环境变量

```yaml
# opencv
$opencv_path\build\x64\vc16\bin

# tensorrt
$tensorrt\bin
$tensorrt\lib
```

# 关于include文件夹

> include文件夹是rapidjson的文件，用来解析json

# Cmake

> cmake版本要设置 `CMakeLists.txt` 中 opencv，tensorrt 路径为自己的路径

# 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
