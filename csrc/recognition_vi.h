#pragma once

#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "vocab.h"

namespace vncv {

/**
 * Vietnamese OCR recognition using VietOCR Transformer.
 *
 * Replicates VietOCROnnxEngine + VietOCRRecognition (seq_modeling=transformer):
 *   - Encoder: (1, 3, H, W) → memory (T, 1, D)
 *   - Decoder: greedy loop (tgt_inp, memory) → token ids
 *   - Decodes with Vocab to UTF-8 string
 */
class RecognitionVi {
public:
    RecognitionVi(const std::string& encoder_path,
                  const std::string& decoder_path,
                  const std::string& vocab_path);

    /**
     * Recognise Vietnamese text in a batch of BGR images.
     *
     * @return (texts, average_probabilities)
     */
    std::pair<std::vector<std::string>, std::vector<float>>
    operator()(const std::vector<cv::Mat>& images);

    // Image pre-processing parameters (match VietOCR defaults)
    int image_height    = 32;
    int image_min_width = 32;
    int image_max_width = 512;
    int max_seq_length  = 128;

private:
    Ort::Env env_;
    Ort::Session enc_session_;
    Ort::Session dec_session_;
    Vocab vocab_;

    std::string enc_input_name_;
    std::string dec_tgt_name_;
    std::string dec_mem_name_;

    /** Resize and normalise one image to (1, 3, H, new_W). */
    std::vector<float> preprocess(const cv::Mat& bgr_image,
                                  int& out_h, int& out_w) const;

    /** Compute (new_w, H) keeping aspect ratio; width rounded up to 10. */
    static std::pair<int, int> compute_size(int w, int h,
                                            int expected_h,
                                            int min_w, int max_w);
};

} // namespace vncv
