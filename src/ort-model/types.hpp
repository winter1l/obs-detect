#ifndef EDGEYOLO_TYPES_HPP
#define EDGEYOLO_TYPES_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

#include <deque>

#ifdef _WIN32
#define file_name_t std::wstring
#else
#define file_name_t std::string
#endif

struct Object {
	cv::Rect_<float> rect;
	int label;
	float prob;
	uint64_t id = 0;
	uint64_t unseenFrames = 0;
	cv::KalmanFilter kf;
	std::deque<cv::Rect_<float>> lastVisibleRects;
	int hitFrames = 0;
	bool isExempt = false;
	bool isUnconfirmed = false;
	std::string customText = "";
	cv::Point2f landmarks[5];
};

struct GridAndStride {
	int grid0;
	int grid1;
	int stride;
};

#endif
