#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vncv {

std::vector<std::vector<cv::Point2f>> sort_polygon(
    std::vector<std::vector<cv::Point2f>> points)
{
    // Primary sort: ascending y of the first corner point
    std::sort(points.begin(), points.end(),
              [](const std::vector<cv::Point2f>& a,
                 const std::vector<cv::Point2f>& b) {
                  return a[0].y < b[0].y;
              });

    // Bubble pass: polygons in the same row (|Δy| < 10) are sorted by x
    for (int i = 0; i < static_cast<int>(points.size()) - 1; ++i) {
        for (int j = i; j >= 0; --j) {
            if (std::abs(points[j + 1][0].y - points[j][0].y) < 10.0f &&
                points[j + 1][0].x < points[j][0].x) {
                std::swap(points[j], points[j + 1]);
            } else {
                break;
            }
        }
    }
    return points;
}

cv::Mat crop_image(const cv::Mat& image,
                   const std::vector<cv::Point2f>& pts)
{
    assert(pts.size() == 4 && "shape of points must be 4*2");

    float w1 = static_cast<float>(cv::norm(pts[0] - pts[1]));
    float w2 = static_cast<float>(cv::norm(pts[2] - pts[3]));
    float h1 = static_cast<float>(cv::norm(pts[0] - pts[3]));
    float h2 = static_cast<float>(cv::norm(pts[1] - pts[2]));

    int crop_width  = static_cast<int>(std::max(w1, w2));
    int crop_height = static_cast<int>(std::max(h1, h2));

    // Protect against degenerate crops
    if (crop_width <= 0)  crop_width  = 1;
    if (crop_height <= 0) crop_height = 1;

    std::vector<cv::Point2f> dst = {
        {0.f,                        0.f},
        {static_cast<float>(crop_width), 0.f},
        {static_cast<float>(crop_width), static_cast<float>(crop_height)},
        {0.f,                        static_cast<float>(crop_height)}
    };

    cv::Mat matrix = cv::getPerspectiveTransform(pts, dst);
    cv::Mat result;
    cv::warpPerspective(image, result, matrix,
                        {crop_width, crop_height},
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    // Rotate tall text strips to landscape
    if (result.rows > 0 && result.cols > 0) {
        if (static_cast<float>(result.rows) / static_cast<float>(result.cols) >= 1.5f) {
            cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
    }
    return result;
}

} // namespace vncv
