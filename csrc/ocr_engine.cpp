#include "ocr_engine.h"

#include <filesystem>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace vncv {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

OcrEngine::OcrEngine(const std::string& weights_dir)
    : weights_dir_(weights_dir)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy getters
// ─────────────────────────────────────────────────────────────────────────────

std::string OcrEngine::weight_path(const std::string& filename) const
{
    return (fs::path(weights_dir_) / filename).string();
}

Detection& OcrEngine::get_detection()
{
    if (!detection_)
        detection_ = std::make_unique<Detection>(weight_path("detection.onnx"));
    return *detection_;
}

Classification& OcrEngine::get_classification()
{
    if (!classification_)
        classification_ = std::make_unique<Classification>(
            weight_path("classification.onnx"));
    return *classification_;
}

RecognitionEn& OcrEngine::get_rec_en()
{
    if (!rec_en_)
        rec_en_ = std::make_unique<RecognitionEn>(weight_path("recognition.onnx"));
    return *rec_en_;
}

RecognitionVi& OcrEngine::get_rec_vi()
{
    if (!rec_vi_)
        rec_vi_ = std::make_unique<RecognitionVi>(
            weight_path("model_encoder.onnx"),
            weight_path("model_decoder.onnx"),
            weight_path("vocab.json"));
    return *rec_vi_;
}

// ─────────────────────────────────────────────────────────────────────────────
// extract
// ─────────────────────────────────────────────────────────────────────────────

std::vector<OcrResult> OcrEngine::extract(const std::string& filepath,
                                           const std::string& lang,
                                           bool save_annotated,
                                           const std::string& annotated_path)
{
    cv::Mat frame = cv::imread(filepath);
    if (frame.empty())
        throw std::runtime_error("Cannot read image: " + filepath);

    cv::Mat annotated = frame.clone();

    // Convert BGR → RGB for detection (matches Python: cv2.cvtColor BGR→RGB)
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    // ── Detection ────────────────────────────────────────────────────────────
    auto raw_points = get_detection()(rgb);

    // Convert to float polygons for sort_polygon
    std::vector<std::vector<cv::Point2f>> float_polys;
    float_polys.reserve(raw_points.size());
    for (const auto& box : raw_points) {
        std::vector<cv::Point2f> fbox;
        for (const auto& pt : box)
            fbox.push_back({static_cast<float>(pt.x), static_cast<float>(pt.y)});
        float_polys.push_back(fbox);
    }
    float_polys = sort_polygon(float_polys);

    // Draw boxes on annotated image
    for (const auto& poly : float_polys) {
        std::vector<cv::Point> ipts;
        for (const auto& p : poly)
            ipts.push_back({static_cast<int>(p.x), static_cast<int>(p.y)});
        cv::polylines(annotated, {ipts}, true, {0, 255, 0}, 2);
    }

    // ── Crop ─────────────────────────────────────────────────────────────────
    std::vector<cv::Mat> crops;
    crops.reserve(float_polys.size());
    for (const auto& poly : float_polys) {
        crops.push_back(crop_image(rgb, poly));
    }

    std::vector<OcrResult> results;
    if (crops.empty()) return results;

    // ── Classification ───────────────────────────────────────────────────────
    get_classification()(crops);  // crops may be rotated in-place

    // ── Recognition ──────────────────────────────────────────────────────────
    std::vector<std::string> texts;
    std::vector<float>       confidences;

    if (lang == "vi") {
        auto [t, c] = get_rec_vi()(crops);
        texts = std::move(t);
        confidences = std::move(c);
    } else if (lang == "en") {
        auto [t, c] = get_rec_en()(crops);
        texts = std::move(t);
        // Flatten per-char confidences to mean confidence
        for (const auto& cv : c) {
            if (cv.empty()) {
                confidences.push_back(0.f);
            } else {
                float s = 0.f;
                for (float x : cv) s += x;
                confidences.push_back(s / static_cast<float>(cv.size()));
            }
        }
    } else {
        throw std::runtime_error("Unsupported language: " + lang);
    }

    // ── Annotate & collect results ───────────────────────────────────────────
    for (size_t i = 0; i < float_polys.size(); ++i) {
        const auto& poly = float_polys[i];
        const std::string& text = (i < texts.size()) ? texts[i] : "";
        float conf = (i < confidences.size()) ? confidences[i] : 0.f;

        // Bounding rect for text label
        std::vector<cv::Point> ipts;
        for (const auto& p : poly)
            ipts.push_back({static_cast<int>(p.x), static_cast<int>(p.y)});
        auto br = cv::boundingRect(ipts);
        cv::putText(annotated, text,
                    {br.x, std::max(br.y - 2, 0)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    {200, 200, 0}, 1, cv::LINE_AA);

        OcrResult res;
        res.text       = text;
        res.confidence = conf;
        for (const auto& p : poly)
            res.box.push_back({static_cast<int>(p.x), static_cast<int>(p.y)});
        results.push_back(std::move(res));
    }

    // ── Optional annotated image save ────────────────────────────────────────
    if (save_annotated) {
        std::string out_path = annotated_path;
        if (out_path.empty()) {
            fs::path p(filepath);
            out_path = (p.parent_path() /
                        (p.stem().string() + "_annotated.png")).string();
        }
        cv::imwrite(out_path, annotated);
    }

    return results;
}

} // namespace vncv
