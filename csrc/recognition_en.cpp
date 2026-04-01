#include "recognition_en.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace vncv {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

RecognitionEn::RecognitionEn(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RecognitionEn"),
      session_(env_, onnx_path.c_str(), Ort::SessionOptions{})
{
    Ort::AllocatorWithDefaultOptions alloc;
    auto name_ptr = session_.GetInputNameAllocated(0, alloc);
    input_name_ = std::string(name_ptr.get());

    // Check whether the model has a dynamic width dimension
    auto shape = session_.GetInputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    // shape = {batch, C, H, W}; W<0 means dynamic
    if (shape.size() == 4 && shape[3] < 0) {
        dynamic_width_ = true;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> RecognitionEn::preprocess_single(const cv::Mat& image,
                                                     int padded_w) const
{
    float ratio = static_cast<float>(image.cols) / static_cast<float>(image.rows);
    int resized_w = std::min(static_cast<int>(std::ceil(INPUT_H * ratio)), padded_w);
    resized_w = std::max(resized_w, 1);

    cv::Mat resized;
    cv::resize(image, resized, {resized_w, INPUT_H});

    cv::Mat flt;
    resized.convertTo(flt, CV_32FC3, 1.0 / 255.0);
    flt = (flt - 0.5f) / 0.5f;

    // (C, H, padded_w) – zero-padded
    std::vector<float> out(INPUT_C * INPUT_H * padded_w, 0.0f);
    std::vector<cv::Mat> channels(3);
    cv::split(flt, channels);
    for (int ch = 0; ch < 3; ++ch) {
        const float* src = channels[ch].ptr<float>();
        float* dst = out.data() + ch * INPUT_H * padded_w;
        for (int row = 0; row < INPUT_H; ++row) {
            std::memcpy(dst + row * padded_w,
                        src + row * resized_w,
                        resized_w * sizeof(float));
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Operator()
// ─────────────────────────────────────────────────────────────────────────────

std::pair<std::vector<std::string>, std::vector<std::vector<float>>>
RecognitionEn::operator()(const std::vector<cv::Mat>& images)
{
    int num = static_cast<int>(images.size());
    std::vector<std::string>       texts(num);
    std::vector<std::vector<float>> confs(num);

    // Sort by aspect ratio
    std::vector<int> indices(num);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        float ra = static_cast<float>(images[a].cols) / images[a].rows;
        float rb = static_cast<float>(images[b].cols) / images[b].rows;
        return ra < rb;
    });

    for (int i = 0; i < num; i += BATCH_SIZE) {
        int batch_end = std::min(num, i + BATCH_SIZE);
        int batch_sz  = batch_end - i;

        // Max aspect ratio in this batch → compute padded_w
        float max_ratio = 0.f;
        for (int j = i; j < batch_end; ++j) {
            const auto& img = images[indices[j]];
            float r = static_cast<float>(img.cols) / img.rows;
            max_ratio = std::max(max_ratio, r);
        }
        float base_ratio = static_cast<float>(INPUT_W) / INPUT_H;
        max_ratio = std::max(max_ratio, base_ratio);
        int padded_w = static_cast<int>(INPUT_H * max_ratio);
        // If model has fixed width, use that
        if (!dynamic_width_) padded_w = INPUT_W;

        // Build batch tensor
        std::vector<float> batch_data;
        batch_data.reserve(batch_sz * INPUT_C * INPUT_H * padded_w);
        for (int j = i; j < batch_end; ++j) {
            auto pre = preprocess_single(images[indices[j]], padded_w);
            batch_data.insert(batch_data.end(), pre.begin(), pre.end());
        }

        std::array<int64_t, 4> shape = {batch_sz, INPUT_C, INPUT_H, padded_w};
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_ort = Ort::Value::CreateTensor<float>(
            mem_info, batch_data.data(), batch_data.size(),
            shape.data(), shape.size());

        const char* input_names[]  = {input_name_.c_str()};
        const char* output_names[] = {"output"};
        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    input_names, &input_ort, 1,
                                    output_names, 1);

        // Output: (batch, T, num_classes)
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int T = static_cast<int>(out_shape[1]);
        int C = static_cast<int>(out_shape[2]);
        const float* out = outputs[0].GetTensorData<float>();

        for (int k = 0; k < batch_sz; ++k) {
            auto [text, conf] = ctc_.decode_single(out + k * T * C, T);
            texts[indices[i + k]] = text;
            confs[indices[i + k]] = conf;
        }
    }
    return {texts, confs};
}

} // namespace vncv
