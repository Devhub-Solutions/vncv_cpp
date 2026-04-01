#pragma once

#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

namespace vncv {

/**
 * Text-orientation classifier.
 *
 * Replicates the Python Classification class:
 *   - Accepts a batch of BGR cropped text images
 *   - Normalises each to (3, 48, 192) with padding
 *   - Runs ONNX session
 *   - Returns (images_possibly_rotated, labels_and_scores)
 *
 * Images predicted as "180°" with confidence > threshold are rotated 180°
 * in-place before being returned.
 */
class Classification {
public:
    explicit Classification(const std::string& onnx_path);

    /**
     * Classify and, if needed, rotate each image.
     *
     * @param images  In/out: a vector of BGR cv::Mat images.
     * @return        A vector of (label, score) pairs; label is "0" or "180".
     */
    std::vector<std::pair<std::string, float>>
    operator()(std::vector<cv::Mat>& images);

    static constexpr int INPUT_C  = 3;
    static constexpr int INPUT_H  = 48;
    static constexpr int INPUT_W  = 192;
    static const int BATCH_SIZE   = 6;
    float threshold               = 0.98f;

private:
    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;

    /** Resize and normalise a single image to (INPUT_C, INPUT_H, INPUT_W). */
    static std::vector<float> preprocess_single(const cv::Mat& image);
};

} // namespace vncv
