#pragma once

#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "ctc_decoder.h"

namespace vncv {

/**
 * English text recognition via CTC.
 *
 * Replicates the Python EnglishRecognition class:
 *   - Input shape (3, 48, 320) with dynamic width
 *   - Batched inference with BATCH_SIZE=6
 *   - CTC decoding
 */
class RecognitionEn {
public:
    explicit RecognitionEn(const std::string& onnx_path);

    /**
     * Recognise text in a batch of BGR images.
     *
     * @return (texts, per-character confidence lists)
     */
    std::pair<std::vector<std::string>, std::vector<std::vector<float>>>
    operator()(const std::vector<cv::Mat>& images);

    static constexpr int INPUT_C = 3;
    static constexpr int INPUT_H = 48;
    static constexpr int INPUT_W = 320;
    static const int BATCH_SIZE  = 6;

private:
    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;
    bool dynamic_width_{false};
    CTCDecoder ctc_;

    std::vector<float> preprocess_single(const cv::Mat& image,
                                         int padded_w) const;
};

} // namespace vncv
