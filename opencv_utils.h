//from https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv

#pragma once

#include "opencv2/opencv.hpp"

/**
 * @brief resize an image to specified size
 *
 * @param src input image
 * @param dst_height output image's height
 * @param dst_width output image's width
 * @return output image if success, error code otherwise
 */
cv::Mat Resize(const cv::Mat& src, int dst_height, int dst_width,
               const std::string& interpolation);

/**
 * @brief crop an image
 *
 * @param src input image
 * @param top
 * @param left
 * @param bottom
 * @param right
 * @return cv::Mat
 */
cv::Mat Crop(const cv::Mat& src, int top, int left, int bottom, int right);

/**
 * @brief 0~255 -> 0~1
 */
cv::Mat Divide(const cv::Mat& src, float divide=255.0);

/**
 * @brief Do normalization to an image
 *
 * @param src input image. It is assumed to be BGR if the channel is 3
 * @param mean
 * @param std
 * @param to_rgb
 * @param inplace
 * @return cv::Mat
 */
cv::Mat Normalize(cv::Mat& src, const std::vector<float>& mean,
                  const std::vector<float>& std, bool to_rgb = false, bool inplace = true);

/**
 * @brief tranpose an image, from {h, w, c} to {c, h, w}
 *
 * @param src input image
 * @return
 */
cv::Mat Transpose(const cv::Mat& src);


/**
 *
 * @param src
 * @param top
 * @param left
 * @param bottom
 * @param right
 * @param border_type
 * @param val
 * @return
 */
cv::Mat Pad(const cv::Mat& src, int top, int left, int bottom, int right,
            int border_type, float val);

/**
 * @brief compare two images
 *
 * @param src1 one input image
 * @param src2 the other input image
 * @return bool true means the images are the same
 */
bool Compare(const cv::Mat& src1, const cv::Mat& src2);
