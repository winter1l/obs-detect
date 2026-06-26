#pragma once

#include "ort-model/ONNXRuntimeModel.h"
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace sface {

class SFaceONNX : public ONNXRuntimeModel {
public:
	SFaceONNX(file_name_t path_to_model, int intra_op_num_threads = 1,
		  int inter_op_num_threads = 1, const std::string &use_gpu_ = "cpu",
		  int device_id = 0, bool use_parallel = false);

	std::vector<Object> inference(const cv::Mat &frame) override;

	// Returns a 128D normalized feature vector for the given face crop.
	// landmarks must contain 5 points: left eye, right eye, nose, left mouth, right mouth
	std::vector<float> inference(const cv::Mat &frame, const cv::Point2f landmarks[5]);

	static float match(const std::vector<float> &feature1, const std::vector<float> &feature2);

private:
	cv::Mat alignCrop(const cv::Mat &src, const cv::Point2f landmarks[5]);
};

} // namespace sface
