#include "SFace.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace sface {

SFaceONNX::SFaceONNX(file_name_t path_to_model, int intra_op_num_threads,
		     int inter_op_num_threads, const std::string &use_gpu_, int device_id,
		     bool use_parallel)
	: ONNXRuntimeModel(path_to_model, intra_op_num_threads, 1, inter_op_num_threads, use_gpu_,
			   device_id, use_parallel, 0.4f, 0.5f)
{
}

std::vector<Object> SFaceONNX::inference(const cv::Mat &frame)
{
	return {};
}

cv::Mat SFaceONNX::alignCrop(const cv::Mat &src, const cv::Point2f landmarks[5])
{
	cv::Point2f srcPoints[3];
	srcPoints[0] = landmarks[0]; // Left eye
	srcPoints[1] = landmarks[1]; // Right eye
	srcPoints[2] = landmarks[2]; // Nose

	cv::Point2f dstPoints[3];
	// Reference points for 112x112 SFace alignment
	dstPoints[0] = cv::Point2f(30.2946f, 51.6963f);
	dstPoints[1] = cv::Point2f(65.5318f, 51.5014f);
	dstPoints[2] = cv::Point2f(48.0252f, 71.7366f);

	cv::Mat M = cv::getAffineTransform(srcPoints, dstPoints);
	cv::Mat aligned;
	cv::warpAffine(src, aligned, M, cv::Size(112, 112), cv::INTER_LINEAR);
	
	return aligned;
}

std::vector<float> SFaceONNX::inference(const cv::Mat &frame, const cv::Point2f landmarks[5])
{
	cv::Mat aligned = alignCrop(frame, landmarks);
	
	// Convert to float, normalize if needed? ONNXRuntimeModel does /255.0f internally.
	// But ONNXRuntimeModel::inference() does letterbox which we DON'T want here.
	// We need to bypass the letterbox from ONNXRuntimeModel or override inference.
	// Wait, ONNXRuntimeModel::inference() resizes and adds padding. 
	// For SFace, aligned is exactly 112x112, so letterbox will just copy it.
	ONNXRuntimeModel::inference(aligned, 0);

	const float *output_ptr = this->output_tensor_[0].GetTensorData<float>();
	std::vector<float> feature(output_ptr, output_ptr + 128);

	// L2 Normalize
	float norm = 0.0f;
	for (float v : feature) {
		norm += v * v;
	}
	norm = std::sqrt(norm);
	if (norm > 1e-6f) {
		for (float &v : feature) {
			v /= norm;
		}
	}

	return feature;
}

float SFaceONNX::match(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
	if (feature1.size() != 128 || feature2.size() != 128) {
		return 0.0f;
	}

	float dot = 0.0f;
	for (size_t i = 0; i < 128; ++i) {
		dot += feature1[i] * feature2[i];
	}
	// Since both vectors are L2 normalized, dot product is cosine similarity.
	return dot;
}

} // namespace sface
