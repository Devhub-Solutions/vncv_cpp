#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

namespace vncv {

/**
 * DBNet text detection.
 *
 * Replicates the Python Detection class:
 *   - Resizes the input image so the longest side ≤ max_size, dims ≡ 0 (mod 32)
 *   - Normalises with ImageNet mean/std
 *   - Runs ONNX session
 *   - Extracts contours → min-area rects → polygon expansion via Clipper2
 *   - Returns filtered bounding-box polygons (each polygon = 4 corners, int32)
 */
class Detection {
public:
    explicit Detection(const std::string& onnx_path);

    /**
     * Run detection on a BGR image.
     * Returns a vector of polygons, each with exactly 4 (x,y) corners.
     */
    std::vector<std::vector<cv::Point2i>> operator()(const cv::Mat& bgr_image);

    // ── Static helpers (also used by unit tests) ──────────────────────────

    /** Order four points clockwise: TL, TR, BR, BL */
    static std::vector<cv::Point2f> clockwise_order(
        const std::vector<cv::Point2f>& points);

    /** Clip each coordinate to [0, w-1] / [0, h-1] */
    static std::vector<cv::Point2i> clip_points(
        std::vector<cv::Point2i> points, int h, int w);

    /** Compute mean bitmap value inside a contour mask */
    static float box_score(const cv::Mat& bitmap,
                           const std::vector<cv::Point>& contour);

    /** Get the 4-corner box and min side of the minimum area rect of a contour */
    static std::pair<std::vector<cv::Point2f>, float> get_min_boxes(
        const std::vector<cv::Point>& contour);

    /** Resize so longest side ≤ max_size and dims ≡ 0 (mod 32) */
    cv::Mat resize(const cv::Mat& image) const;

    /** Zero-pad to at least 32×32 */
    static cv::Mat zero_pad(const cv::Mat& image);

    int max_size{960};
    int min_size{3};
    float box_thresh{0.8f};
    float mask_thresh{0.8f};

private:
    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;

    // ImageNet normalisation parameters
    // mean = [123.675, 116.28, 103.53]  std = 1/[58.395, 57.12, 57.375]
    float mean_[3]{123.675f, 116.28f, 103.53f};
    float inv_std_[3]{1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f};

    std::pair<std::vector<std::vector<cv::Point2i>>, std::vector<float>>
    boxes_from_bitmap(const cv::Mat& output,
                      const cv::Mat& mask,
                      int dest_width, int dest_height);

    std::vector<std::vector<cv::Point2i>> filter_polygon(
        const std::vector<std::vector<cv::Point2i>>& boxes,
        int img_h, int img_w);
};

} // namespace vncv
