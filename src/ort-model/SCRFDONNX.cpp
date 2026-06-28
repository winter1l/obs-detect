#include "SCRFDONNX.hpp"
#include <algorithm>
#include <cmath>

SCRFDONNX::SCRFDONNX(file_name_t path_to_model, int intra_op_num_threads,
		     int inter_op_num_threads, const std::string &use_gpu_,
		     int device_id, bool use_parallel, float nms_th, float conf_th)
	: ONNXRuntimeModel(path_to_model, intra_op_num_threads, 1, inter_op_num_threads,
			   use_gpu_, device_id, use_parallel, nms_th, conf_th),
	  strides({8, 16, 32})
{
}

std::vector<Object> SCRFDONNX::inference(const cv::Mat &frame)
{
	cv::Mat pr_img = static_resize(frame, 0);
	float *blob_data = (float *)(this->input_buffer_[0].get());
	blobFromImage(pr_img, blob_data);

	size_t input_size = input_w_[0] * input_h_[0] * 3;
	for (size_t i = 0; i < input_size; ++i) {
		blob_data[i] = (blob_data[i] - 127.5f) / 128.0f;
	}

	std::vector<const char *> input_names;
	for (size_t i = 0; i < this->input_name_.size(); i++) {
		input_names.push_back(this->input_name_[i].c_str());
	}
	std::vector<const char *> output_names;
	for (size_t i = 0; i < this->output_name_.size(); i++) {
		output_names.push_back(this->output_name_[i].c_str());
	}

	Ort::RunOptions run_options;
	this->session_.Run(run_options, input_names.data(), this->input_tensor_.data(),
			   this->input_tensor_.size(), output_names.data(),
			   this->output_tensor_.data(), this->output_tensor_.size());

	float scale = std::fminf((float)input_w_[0] / (float)frame.cols,
				 (float)input_h_[0] / (float)frame.rows);

	std::vector<Object> faces;

	for (size_t i = 0; i < strides.size(); ++i) {
		const float stride = (float)strides[i];
		int feat_h = input_h_[0] / (int)stride;
		int feat_w = input_w_[0] / (int)stride;

		const float *score_v = (const float *)this->output_buffer_[i].get();
		const float *bbox_v = (const float *)this->output_buffer_[i + 3].get();
		const float *kps_v = (const float *)this->output_buffer_[i + 6].get();

		for (int r = 0; r < feat_h; ++r) {
			for (int c = 0; c < feat_w; ++c) {
				for (int a = 0; a < 2; ++a) {
					size_t idx = (r * feat_w + c) * 2 + a;

					float score = score_v[idx];
					if (score < this->bbox_conf_thresh_)
						continue;

					float cx = c * stride;
					float cy = r * stride;

					float x1 = (cx - bbox_v[idx * 4 + 0] * stride) / scale;
					float y1 = (cy - bbox_v[idx * 4 + 1] * stride) / scale;
					float x2 = (cx + bbox_v[idx * 4 + 2] * stride) / scale;
					float y2 = (cy + bbox_v[idx * 4 + 3] * stride) / scale;

					x1 = std::max(0.f, x1);
					y1 = std::max(0.f, y1);
					x2 = std::min((float)frame.cols - 1.f, x2);
					y2 = std::min((float)frame.rows - 1.f, y2);

					Object face;
					face.rect = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
					face.prob = score;
					face.label = 0;

					for (int n = 0; n < 5; ++n) {
						face.landmarks[n].x = (cx + kps_v[idx * 10 + 2 * n] * stride) / scale;
						face.landmarks[n].y = (cy + kps_v[idx * 10 + 2 * n + 1] * stride) / scale;
					}

					faces.push_back(face);
				}
			}
		}
	}

	ONNXRuntimeModel::qsort_descent_inplace(faces);
	std::vector<int> picked;
	ONNXRuntimeModel::nms_sorted_bboxes(faces, picked, this->nms_thresh_);

	std::vector<Object> final_faces;
	for (size_t i = 0; i < picked.size(); ++i) {
		final_faces.push_back(faces[picked[i]]);
	}

	return final_faces;
}
