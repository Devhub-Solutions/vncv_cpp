#include "detection.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

// Clipper2 – polygon offsetting (replaces shapely + pyclipper)
#include <clipper2/clipper.h>

namespace vncv {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

Detection::Detection(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Detection"),
      session_(env_, onnx_path.c_str(), Ort::SessionOptions{})
{
    Ort::AllocatorWithDefaultOptions alloc;
    auto name_ptr = session_.GetInputNameAllocated(0, alloc);
    input_name_ = std::string(name_ptr.get());
}

// ─────────────────────────────────────────────────────────────────────────────
// Operator()
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<cv::Point2i>> Detection::operator()(const cv::Mat& bgr_image)
{
    int orig_h = bgr_image.rows;
    int orig_w = bgr_image.cols;

    cv::Mat img = bgr_image;
    if (orig_h + orig_w < 64) {
        img = zero_pad(img);
    }
    img = resize(img);

    // Normalise: subtract mean, multiply by 1/std
    cv::Mat flt;
    img.convertTo(flt, CV_32FC3);
    for (int r = 0; r < flt.rows; ++r) {
        auto* row = flt.ptr<float>(r);
        for (int c = 0; c < flt.cols; ++c) {
            // OpenCV stores BGR; ImageNet mean/std also BGR order
            for (int ch = 0; ch < 3; ++ch) {
                float& v = row[c * 3 + ch];
                v = (v - mean_[ch]) * inv_std_[ch];
            }
        }
    }

    // CHW layout for ONNX: (1, 3, H, W)
    int H = flt.rows, W = flt.cols;
    std::vector<float> input_tensor(3 * H * W);
    std::vector<cv::Mat> channels(3);
    cv::split(flt, channels);
    for (int ch = 0; ch < 3; ++ch) {
        std::memcpy(input_tensor.data() + ch * H * W,
                    channels[ch].ptr<float>(),
                    H * W * sizeof(float));
    }

    // Run ONNX
    std::array<int64_t, 4> shape = {1, 3, H, W};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_ort = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor.data(), input_tensor.size(),
        shape.data(), shape.size());

    const char* input_names[]  = {input_name_.c_str()};
    const char* output_names[] = {"output"};

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names, &input_ort, 1,
                                output_names, 1);

    // Output: (1, 1, H, W) float32
    auto* out_data = outputs[0].GetTensorData<float>();
    int out_h = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2]);
    int out_w = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetShape()[3]);

    cv::Mat output_mat(out_h, out_w, CV_32FC1,
                       const_cast<float*>(out_data));

    cv::Mat mask;
    cv::threshold(output_mat, mask, mask_thresh, 1.0f, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1, 255.0);

    auto [boxes, scores] =
        boxes_from_bitmap(output_mat, mask, orig_w, orig_h);

    return filter_polygon(boxes, orig_h, orig_w);
}

// ─────────────────────────────────────────────────────────────────────────────
// boxes_from_bitmap
// ─────────────────────────────────────────────────────────────────────────────

std::pair<std::vector<std::vector<cv::Point2i>>, std::vector<float>>
Detection::boxes_from_bitmap(const cv::Mat& output,
                              const cv::Mat& mask,
                              int dest_width, int dest_height)
{
    int height = mask.rows;
    int width  = mask.cols;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point2i>> boxes;
    std::vector<float> scores;

    for (const auto& contour : contours) {
        auto [corners, min_side] = get_min_boxes(contour);
        if (min_side < static_cast<float>(min_size)) continue;

        float score = box_score(output, contour);
        if (score < box_thresh) continue;

        // Polygon area / perimeter → expansion distance
        // Use Clipper2 PathD for floating-point polygon offset
        Clipper2Lib::PathD path;
        path.reserve(corners.size());
        for (const auto& pt : corners) {
            path.push_back({static_cast<double>(pt.x),
                            static_cast<double>(pt.y)});
        }

        double area      = std::abs(Clipper2Lib::Area(path));
        // Perimeter (polygon length)
        double perimeter = 0.0;
        for (size_t i = 0; i < path.size(); ++i) {
            const auto& p1 = path[i];
            const auto& p2 = path[(i + 1) % path.size()];
            double dx = p2.x - p1.x, dy = p2.y - p1.y;
            perimeter += std::sqrt(dx * dx + dy * dy);
        }
        if (perimeter < 1e-6) continue;

        double distance = area / perimeter;

        // Expand polygon
        Clipper2Lib::PathsD expanded =
            Clipper2Lib::InflatePaths({path},
                                      distance * 1.5,
                                      Clipper2Lib::JoinType::Round,
                                      Clipper2Lib::EndType::Polygon);

        if (expanded.empty()) continue;

        // Convert back to contour and get min-box again
        std::vector<cv::Point> exp_contour;
        exp_contour.reserve(expanded[0].size());
        for (const auto& pt : expanded[0]) {
            exp_contour.push_back({static_cast<int>(std::round(pt.x)),
                                   static_cast<int>(std::round(pt.y))});
        }
        auto [box, box_min_side] = get_min_boxes(exp_contour);
        if (box_min_side < static_cast<float>(min_size + 2)) continue;

        // Scale back to original image coordinates
        std::vector<cv::Point2i> scaled_box;
        scaled_box.reserve(4);
        for (const auto& pt : box) {
            int x = static_cast<int>(std::clamp(
                std::round(pt.x / width  * dest_width),
                0.0f, static_cast<float>(dest_width)));
            int y = static_cast<int>(std::clamp(
                std::round(pt.y / height * dest_height),
                0.0f, static_cast<float>(dest_height)));
            scaled_box.push_back({x, y});
        }
        boxes.push_back(scaled_box);
        scores.push_back(score);
    }

    return {boxes, scores};
}

// ─────────────────────────────────────────────────────────────────────────────
// filter_polygon
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<cv::Point2i>> Detection::filter_polygon(
    const std::vector<std::vector<cv::Point2i>>& boxes,
    int img_h, int img_w)
{
    std::vector<std::vector<cv::Point2i>> filtered;
    for (const auto& box : boxes) {
        // Re-order clockwise
        std::vector<cv::Point2f> fbox;
        fbox.reserve(box.size());
        for (const auto& pt : box) fbox.push_back({static_cast<float>(pt.x),
                                                    static_cast<float>(pt.y)});
        auto ordered_f = clockwise_order(fbox);

        // Clip to image bounds
        std::vector<cv::Point2i> ordered;
        ordered.reserve(ordered_f.size());
        for (const auto& pt : ordered_f)
            ordered.push_back({static_cast<int>(pt.x), static_cast<int>(pt.y)});
        ordered = clip_points(ordered, img_h, img_w);

        // Check minimum dimensions
        int w = static_cast<int>(cv::norm(
            cv::Point2f(ordered[0]) - cv::Point2f(ordered[1])));
        int h = static_cast<int>(cv::norm(
            cv::Point2f(ordered[0]) - cv::Point2f(ordered[3])));
        if (w <= 3 || h <= 3) continue;

        filtered.push_back(ordered);
    }
    return filtered;
}

// ─────────────────────────────────────────────────────────────────────────────
// Static helpers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<cv::Point2f> Detection::clockwise_order(
    const std::vector<cv::Point2f>& points)
{
    // min sum → TL, max sum → BR
    int min_s = 0, max_s = 0;
    float min_val = points[0].x + points[0].y;
    float max_val = min_val;
    for (int i = 1; i < static_cast<int>(points.size()); ++i) {
        float s = points[i].x + points[i].y;
        if (s < min_val) { min_val = s; min_s = i; }
        if (s > max_val) { max_val = s; max_s = i; }
    }

    std::vector<cv::Point2f> tmp;
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (i != min_s && i != max_s) tmp.push_back(points[i]);
    }

    // min diff → TR, max diff → BL
    int tr = 0, bl = 1;
    float d0 = tmp[0].x - tmp[0].y;
    float d1 = tmp[1].x - tmp[1].y;
    if (d0 > d1) { tr = 0; bl = 1; }
    else         { tr = 1; bl = 0; }

    std::vector<cv::Point2f> poly(4);
    poly[0] = points[min_s];    // TL
    poly[1] = tmp[tr];           // TR
    poly[2] = points[max_s];    // BR
    poly[3] = tmp[bl];           // BL
    return poly;
}

std::vector<cv::Point2i> Detection::clip_points(
    std::vector<cv::Point2i> points, int h, int w)
{
    for (auto& pt : points) {
        pt.x = std::clamp(pt.x, 0, w - 1);
        pt.y = std::clamp(pt.y, 0, h - 1);
    }
    return points;
}

float Detection::box_score(const cv::Mat& bitmap,
                            const std::vector<cv::Point>& contour)
{
    int bh = bitmap.rows, bw = bitmap.cols;
    std::vector<cv::Point> cont = contour;
    // Bounding rect of contour
    int x1 = bw, y1 = bh, x2 = 0, y2 = 0;
    for (const auto& pt : cont) {
        x1 = std::min(x1, std::clamp(pt.x, 0, bw - 1));
        y1 = std::min(y1, std::clamp(pt.y, 0, bh - 1));
        x2 = std::max(x2, std::clamp(pt.x, 0, bw - 1));
        y2 = std::max(y2, std::clamp(pt.y, 0, bh - 1));
    }
    if (x2 <= x1 || y2 <= y1) return 0.f;

    cv::Mat mask_roi = cv::Mat::zeros(y2 - y1 + 1, x2 - x1 + 1, CV_8UC1);
    // Shift contour by (x1, y1)
    std::vector<std::vector<cv::Point>> shifted(1);
    shifted[0].reserve(cont.size());
    for (auto& pt : cont) {
        shifted[0].push_back({pt.x - x1, pt.y - y1});
    }
    cv::fillPoly(mask_roi, shifted, cv::Scalar(1));

    cv::Scalar mean = cv::mean(bitmap(cv::Rect(x1, y1,
                                               x2 - x1 + 1,
                                               y2 - y1 + 1)),
                               mask_roi);
    return static_cast<float>(mean[0]);
}

std::pair<std::vector<cv::Point2f>, float> Detection::get_min_boxes(
    const std::vector<cv::Point>& contour)
{
    // cv::minAreaRect needs a vector<Point>
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f pts[4];
    rect.points(pts);

    // Sort by x ascending
    std::vector<cv::Point2f> sorted(pts, pts + 4);
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) {
                  return a.x < b.x;
              });

    // Left two: smaller y → index_1, larger y → index_4
    int idx1, idx4;
    if (sorted[1].y > sorted[0].y) { idx1 = 0; idx4 = 1; }
    else                            { idx1 = 1; idx4 = 0; }

    int idx2, idx3;
    if (sorted[3].y > sorted[2].y) { idx2 = 2; idx3 = 3; }
    else                            { idx2 = 3; idx3 = 2; }

    std::vector<cv::Point2f> box = {
        sorted[idx1], sorted[idx2], sorted[idx3], sorted[idx4]
    };
    float min_side = std::min(rect.size.width, rect.size.height);
    return {box, min_side};
}

cv::Mat Detection::resize(const cv::Mat& image) const
{
    int h = image.rows, w = image.cols;
    float ratio = (std::max(h, w) > max_size)
                  ? static_cast<float>(max_size) / static_cast<float>(std::max(h, w))
                  : 1.0f;

    int rh = std::max(static_cast<int>(std::round(h * ratio / 32.f) * 32), 32);
    int rw = std::max(static_cast<int>(std::round(w * ratio / 32.f) * 32), 32);

    cv::Mat resized;
    cv::resize(image, resized, {rw, rh});
    return resized;
}

cv::Mat Detection::zero_pad(const cv::Mat& image)
{
    int ph = std::max(32, image.rows);
    int pw = std::max(32, image.cols);
    cv::Mat pad = cv::Mat::zeros(ph, pw, image.type());
    image.copyTo(pad(cv::Rect(0, 0, image.cols, image.rows)));
    return pad;
}

} // namespace vncv
