#ifndef _YOLOV8_FACE_ONNX_HPP
#define _YOLOV8_FACE_ONNX_HPP

#include "ONNXRuntimeModel.h"

class YOLOv8FaceONNX : public ONNXRuntimeModel {
public:
	YOLOv8FaceONNX(file_name_t path_to_model, int intra_op_num_threads,
		       int inter_op_num_threads = 1, const std::string &use_gpu_ = "",
		       int device_id = 0, bool use_parallel = false, float nms_th = 0.45f,
		       float conf_th = 0.3f);
	std::vector<Object> inference(const cv::Mat &frame) override;
#ifdef _WIN32
	std::vector<Object> inference_gpu() override;
#endif
};

#endif
