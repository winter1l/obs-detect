#ifndef FRAME_UTILS_H
#define FRAME_UTILS_H

#include <cstdint>

#include <opencv2/core.hpp>

cv::Mat copyMappedBGRA(const uint8_t *data, uint32_t linesize, uint32_t width, uint32_t height);

#endif // FRAME_UTILS_H
