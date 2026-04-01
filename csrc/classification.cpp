#include "classification.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace vncv {

namespace {
Ort::Session make_session(Ort::Env& env, const std::string& onnx_path) {
    Ort::SessionOptions opts;
#ifdef _WIN32
    const std::wstring wpath(onnx_path.begin(), onnx_path.end());
    return Ort::Session(env, wpath.c_str(), opts);
#else
    return Ort::Session(env, onnx_path.c_str(), opts);
#endif
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

Classification::Classification(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Classification"),
      session_(make_session(env_, onnx_path))
{
    Ort::AllocatorWithDefaultOptions alloc;
    auto name_ptr = session_.GetInputNameAllocated(0, alloc);
    input_name_ = std::string(name_ptr.get());
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> Classification::preprocess_single(const cv::Mat& image)
{
    const float input_h = static_cast<float>(INPUT_H);
    const float input_w = static_cast<float>(INPUT_W);

    float ratio = static_cast<float>(image.cols) / static_cast<float>(image.rows);
    int resized_w = (std::ceil(input_h * ratio) > input_w)
                    ? static_cast<int>(input_w)
                    : static_cast<int>(std::ceil(input_h * ratio));
    resized_w = std::max(resized_w, 1);

    cv::Mat resized;
    cv::resize(image, resized, {resized_w, INPUT_H});

    // (H, W, C) BGR float32 → normalise to [-1, 1]
    cv::Mat flt;
    resized.convertTo(flt, CV_32FC3, 1.0 / 255.0);
    flt = (flt - 0.5f) / 0.5f;

    // Transpose to (C, H, W) and pad to (C, INPUT_H, INPUT_W)
    std::vector<float> padded(INPUT_C * INPUT_H * INPUT_W, 0.0f);
    std::vector<cv::Mat> channels(3);
    cv::split(flt, channels);
    for (int ch = 0; ch < 3; ++ch) {
        const float* src = channels[ch].ptr<float>();
        float* dst = padded.data() + ch * INPUT_H * INPUT_W;
        std::memcpy(dst, src, INPUT_H * resized_w * sizeof(float));
    }
    return padded;
}

// ─────────────────────────────────────────────────────────────────────────────
// Operator()
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, float>>
Classification::operator()(std::vector<cv::Mat>& images)
{
    int num = static_cast<int>(images.size());
    std::vector<std::pair<std::string, float>> results(num, {"0", 0.0f});

    // Sort by aspect ratio (width/height) to minimise padding waste in a batch
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

        // Build batch tensor
        std::vector<float> batch_data;
        batch_data.reserve(batch_sz * INPUT_C * INPUT_H * INPUT_W);
        for (int j = i; j < batch_end; ++j) {
            auto pre = preprocess_single(images[indices[j]]);
            batch_data.insert(batch_data.end(), pre.begin(), pre.end());
        }

        std::array<int64_t, 4> shape = {batch_sz, INPUT_C, INPUT_H, INPUT_W};
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

        const float* out = outputs[0].GetTensorData<float>();
        // out shape: (batch_sz, 2)  – logits for ["0", "180"]
        for (int k = 0; k < batch_sz; ++k) {
            float l0 = out[k * 2 + 0];
            float l1 = out[k * 2 + 1];
            // Softmax over 2 classes
            float max_l = std::max(l0, l1);
            float e0 = std::exp(l0 - max_l);
            float e1 = std::exp(l1 - max_l);
            float sum = e0 + e1;
            float p0 = e0 / sum;
            float p1 = e1 / sum;

            int best = (p1 > p0) ? 1 : 0;
            float score = (best == 1) ? p1 : p0;
            std::string label = (best == 1) ? "180" : "0";

            int img_idx = indices[i + k];
            results[img_idx] = {label, score};

            if (label == "180" && score > threshold) {
                cv::rotate(images[img_idx], images[img_idx], cv::ROTATE_180);
            }
        }
    }
    return results;
}

} // namespace vncv
