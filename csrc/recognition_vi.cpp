#include "recognition_vi.h"

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

RecognitionVi::RecognitionVi(const std::string& encoder_path,
                             const std::string& decoder_path,
                             const std::string& vocab_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RecognitionVi"),
      enc_session_(make_session(env_, encoder_path)),
      dec_session_(make_session(env_, decoder_path)),
      vocab_(vocab_path)
{
    Ort::AllocatorWithDefaultOptions alloc;

    auto enc_name = enc_session_.GetInputNameAllocated(0, alloc);
    enc_input_name_ = std::string(enc_name.get());

    // Decoder inputs: "tgt_inp" and "memory"
    for (size_t i = 0; i < dec_session_.GetInputCount(); ++i) {
        auto n = dec_session_.GetInputNameAllocated(i, alloc);
        std::string nm(n.get());
        if (nm == "tgt_inp") dec_tgt_name_ = nm;
        else if (nm == "memory") dec_mem_name_ = nm;
    }
    if (dec_tgt_name_.empty()) dec_tgt_name_ = "tgt_inp";
    if (dec_mem_name_.empty()) dec_mem_name_ = "memory";
}

// ─────────────────────────────────────────────────────────────────────────────
// Size helpers
// ─────────────────────────────────────────────────────────────────────────────

std::pair<int, int> RecognitionVi::compute_size(int w, int h,
                                                int expected_h,
                                                int min_w, int max_w)
{
    int new_w = static_cast<int>(expected_h * static_cast<float>(w) / h);
    const int round_to = 10;
    new_w = static_cast<int>(std::ceil(static_cast<float>(new_w) / round_to)) * round_to;
    new_w = std::max(new_w, min_w);
    new_w = std::min(new_w, max_w);
    return {new_w, expected_h};
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing
// ─────────────────────────────────────────────────────────────────────────────

std::vector<float> RecognitionVi::preprocess(const cv::Mat& bgr_image,
                                             int& out_h, int& out_w) const
{
    // Convert BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);

    auto [new_w, new_h] = compute_size(rgb.cols, rgb.rows,
                                       image_height,
                                       image_min_width, image_max_width);

    cv::Mat resized;
    cv::resize(rgb, resized, {new_w, new_h}, 0, 0, cv::INTER_LANCZOS4);

    out_h = new_h;
    out_w = new_w;

    // (H, W, C) uint8 → float32 [0,1] → (1, C, H, W)
    cv::Mat flt;
    resized.convertTo(flt, CV_32FC3, 1.0 / 255.0);

    std::vector<float> tensor(3 * new_h * new_w);
    std::vector<cv::Mat> channels(3);
    cv::split(flt, channels);
    for (int ch = 0; ch < 3; ++ch) {
        std::memcpy(tensor.data() + ch * new_h * new_w,
                    channels[ch].ptr<float>(),
                    new_h * new_w * sizeof(float));
    }
    return tensor; // batch dimension added by caller
}

// ─────────────────────────────────────────────────────────────────────────────
// Operator()
// ─────────────────────────────────────────────────────────────────────────────

std::pair<std::vector<std::string>, std::vector<float>>
RecognitionVi::operator()(const std::vector<cv::Mat>& images)
{
    int num = static_cast<int>(images.size());
    std::vector<std::string> texts(num);
    std::vector<float> probs(num, 0.f);

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (int idx = 0; idx < num; ++idx) {
        // ── Pre-process ──────────────────────────────────────────────────
        int H, W;
        auto input_data = preprocess(images[idx], H, W);

        std::array<int64_t, 4> enc_shape = {1, 3, H, W};
        auto enc_input = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            enc_shape.data(), enc_shape.size());

        // ── Encoder ─────────────────────────────────────────────────────
        const char* enc_in_names[]  = {enc_input_name_.c_str()};
        const char* enc_out_names[] = {"memory"};
        auto enc_outputs = enc_session_.Run(Ort::RunOptions{nullptr},
                                            enc_in_names, &enc_input, 1,
                                            enc_out_names, 1);

        // memory shape: (T, 1, D) or (T, B, D)
        auto mem_shape = enc_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto* mem_data = enc_outputs[0].GetTensorData<float>();
        size_t mem_size = 1;
        for (auto s : mem_shape) mem_size *= static_cast<size_t>(s > 0 ? s : 1);

        // ── Greedy decode ────────────────────────────────────────────────
        // translated[step] = token_id  (batch=1)
        std::vector<int64_t> translated = {Vocab::SOS};
        std::vector<float>   char_probs = {1.0f};

        const char* dec_out_names[] = {"output"};

        for (int step = 0; step <= max_seq_length; ++step) {
            bool has_eos = false;
            for (int64_t tok : translated)
                if (tok == Vocab::EOS) { has_eos = true; break; }
            if (has_eos) break;

            int tgt_len = static_cast<int>(translated.size());

            // tgt_inp: (tgt_len, 1) int64
            std::array<int64_t, 2> tgt_shape = {tgt_len, 1};
            auto tgt_inp = Ort::Value::CreateTensor<int64_t>(
                mem_info,
                translated.data(), translated.size(),
                tgt_shape.data(), tgt_shape.size());

            // memory: keep same tensor
            auto mem_inp = Ort::Value::CreateTensor<float>(
                mem_info,
                const_cast<float*>(mem_data), mem_size,
                mem_shape.data(), mem_shape.size());

            std::vector<const char*> dec_in_names = {
                dec_tgt_name_.c_str(), dec_mem_name_.c_str()
            };
            std::vector<Ort::Value> dec_inputs;
            dec_inputs.push_back(std::move(tgt_inp));
            dec_inputs.push_back(std::move(mem_inp));

            auto dec_outputs = dec_session_.Run(
                Ort::RunOptions{nullptr},
                dec_in_names.data(), dec_inputs.data(), 2,
                dec_out_names, 1);

            // output: (1, tgt_len, vocab_size)
            auto out_shape = dec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int V = static_cast<int>(out_shape[2]);
            const float* out = dec_outputs[0].GetTensorData<float>();

            // Last step logits: out[(tgt_len-1) * V .. (tgt_len-1)*V + V-1]
            const float* last = out + (tgt_len - 1) * V;

            // Softmax
            float max_v = *std::max_element(last, last + V);
            std::vector<float> prob(V);
            float sum = 0.f;
            for (int v = 0; v < V; ++v) {
                prob[v] = std::exp(last[v] - max_v);
                sum += prob[v];
            }
            for (float& p : prob) p /= sum;

            int best = static_cast<int>(
                std::max_element(prob.begin(), prob.end()) - prob.begin());

            translated.push_back(static_cast<int64_t>(best));
            char_probs.push_back(prob[best]);
        }

        // ── Decode tokens → string ────────────────────────────────────
        std::vector<int> ids;
        ids.reserve(translated.size());
        for (int64_t t : translated) ids.push_back(static_cast<int>(t));
        texts[idx] = vocab_.decode(ids);

        // Average confidence (skip special tokens ≤ 3)
        float sum_p = 0.f;
        int count   = 0;
        for (size_t s = 0; s < translated.size(); ++s) {
            if (translated[s] > 3) {
                sum_p += char_probs[s];
                ++count;
            }
        }
        probs[idx] = count > 0 ? sum_p / count : 0.f;
    }

    return {texts, probs};
}

} // namespace vncv
