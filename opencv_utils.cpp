// Copyright (c) OpenMMLab. All rights reserved.

#include "opencv_utils.h"
#include <map>


cv::Mat Resize(const cv::Mat& src, int dst_height, int dst_width,
               const std::string& interpolation) {
    cv::Mat dst(dst_height, dst_width, src.type());
    if (interpolation == "bilinear") {
        cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
    } else if (interpolation == "nearest") {
        cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_NEAREST);
    } else if (interpolation == "area") {
        cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_AREA);
    } else if (interpolation == "bicubic") {
        cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_CUBIC);
    } else if (interpolation == "lanczos") {
        cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LANCZOS4);
    } else {
        assert(0);
    }
    return dst;
}

cv::Mat Crop(const cv::Mat& src, int top, int left, int bottom, int right) {
    return src(cv::Range(top, bottom + 1), cv::Range(left, right + 1)).clone();
}

cv::Mat Divide(const cv::Mat& src, float divide) {
    cv::Mat dst;
    src.convertTo(dst, CV_32FC3, 1.0 / divide);
    return dst;
}

cv::Mat Normalize(cv::Mat& src, const std::vector<float>& mean, const std::vector<float>& std,
                  bool to_rgb, bool inplace) {
    assert(src.channels() == mean.size());
    assert(mean.size() == std.size());

    cv::Mat dst;

    if (src.depth() == CV_32F) {
        dst = inplace ? src : src.clone();
    } else {
        src.convertTo(dst, CV_32FC(src.channels()));
    }

    if (to_rgb && dst.channels() == 3) {
        cv::cvtColor(dst, dst, cv::ColorConversionCodes::COLOR_BGR2RGB);
    }

    auto _mean = mean;
    auto _std = std;
    for (auto i = mean.size(); i < 3; ++i) {
        _mean.push_back(0.);
        _std.push_back(1.0);
    }
    cv::Scalar mean_scalar(_mean[0], _mean[1], _mean[2]);
    cv::Scalar std_scalar(1.0 / _std[0], 1.0 / _std[1], 1.0 / _std[2]);

    cv::subtract(dst, mean_scalar, dst);
    cv::multiply(dst, std_scalar, dst);
    return dst;
}

cv::Mat Transpose(const cv::Mat& src) {
    cv::Mat _src{ src.rows * src.cols, src.channels(), CV_MAKETYPE(src.depth(), 1), src.data };
    cv::Mat _dst;
    cv::transpose(_src, _dst);
    return _dst;
}

cv::Mat Pad(const cv::Mat& src, int top, int left, int bottom, int right, int border_type,
            float val) {
    cv::Mat dst;
    cv::Scalar scalar = { val, val, val, val };
    cv::copyMakeBorder(src, dst, top, bottom, left, right, border_type, scalar);
    return dst;
}
