#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace vncv {

/**
 * Sort polygon boxes top-to-bottom, left-to-right.
 * Points whose y-coordinates differ by < 10 pixels are treated as the same row
 * and sorted by ascending x.
 *
 * Each polygon is represented as a vector of cv::Point2f with at least one
 * element (the first element is used as the reference coordinate).
 */
std::vector<std::vector<cv::Point2f>> sort_polygon(
    std::vector<std::vector<cv::Point2f>> points);

/**
 * Perspective-crop a four-corner region from `image`.
 * `pts` must have exactly 4 points in clockwise order (TL, TR, BR, BL).
 * If the cropped region is taller than 1.5× its width it is rotated 90°
 * counter-clockwise so text lines are always landscape.
 */
cv::Mat crop_image(const cv::Mat& image,
                   const std::vector<cv::Point2f>& pts);

} // namespace vncv
