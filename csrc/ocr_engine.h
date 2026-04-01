#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detection.h"
#include "classification.h"
#include "recognition_en.h"
#include "recognition_vi.h"
#include "utils.h"

namespace vncv {

/**
 * Result of recognising one text region.
 */
struct OcrResult {
    std::string text;
    float       confidence{0.f};
    /** Four (x, y) integer corner points of the bounding box */
    std::vector<std::array<int, 2>> box;
};

/**
 * Unified OCR pipeline: Detection → Classification → Recognition.
 *
 * Mirrors extract_text() from Python ocr.py.
 * Engines are created lazily on first use for the requested language so that
 * loading a model not required by the caller is avoided.
 */
class OcrEngine {
public:
    /**
     * @param weights_dir  Directory containing all *.onnx and vocab.json files.
     */
    explicit OcrEngine(const std::string& weights_dir);

    /**
     * Run the full OCR pipeline on an image file.
     *
     * @param filepath       Path to input image (anything OpenCV can read).
     * @param lang           "vi" (VietOCR) or "en" (CTC).
     * @param save_annotated If true, write annotated image to `annotated_path`.
     * @param annotated_path Destination for annotated image (uses
     *                       "<stem>_annotated.png" next to input if empty).
     * @return               One OcrResult per detected text region.
     */
    std::vector<OcrResult> extract(const std::string& filepath,
                                   const std::string& lang = "vi",
                                   bool save_annotated = false,
                                   const std::string& annotated_path = "");

private:
    std::string weights_dir_;

    // Lazily-initialised model instances
    std::unique_ptr<Detection>       detection_;
    std::unique_ptr<Classification>  classification_;
    std::unique_ptr<RecognitionEn>   rec_en_;
    std::unique_ptr<RecognitionVi>   rec_vi_;

    Detection&      get_detection();
    Classification& get_classification();
    RecognitionEn&  get_rec_en();
    RecognitionVi&  get_rec_vi();

    std::string weight_path(const std::string& filename) const;
};

} // namespace vncv
