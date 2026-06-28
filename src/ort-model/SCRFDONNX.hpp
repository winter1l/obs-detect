#ifndef _SCRFD_ONNX_HPP
#define _SCRFD_ONNX_HPP

#include "ONNXRuntimeModel.h"

class SCRFDONNX : public ONNXRuntimeModel {
public:
	SCRFDONNX(file_name_t path_to_model, int intra_op_num_threads,
		  int inter_op_num_threads = 1, const std::string &use_gpu_ = "",
		  int device_id = 0, bool use_parallel = false, float nms_th = 0.4f,
		  float conf_th = 0.3f);
	std::vector<Object> inference(const cv::Mat &frame) override;

private:
	std::vector<int> strides;
};

#endif
