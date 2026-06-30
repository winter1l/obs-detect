#include "YOLOv8FaceONNX.hpp"

#ifdef _WIN32
#include <d3d12.h>
#include <obs.h>
#include "plugin-support.h"
#include <stdexcept>
#endif

YOLOv8FaceONNX::YOLOv8FaceONNX(file_name_t path_to_model, int intra_op_num_threads,
			       int inter_op_num_threads, const std::string &use_gpu_,
			       int device_id, bool use_parallel, float nms_th, float conf_th)
	: ONNXRuntimeModel(path_to_model, intra_op_num_threads, 1, inter_op_num_threads,
			   use_gpu_, device_id, use_parallel, nms_th, conf_th)
{
	this->mean_ = {0.0f, 0.0f, 0.0f};
	this->std_ = {1.0f, 1.0f, 1.0f};
}

std::vector<Object> YOLOv8FaceONNX::inference(const cv::Mat &frame)
{
	cv::Mat rgb_frame;
	cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
	cv::Mat pr_img = static_resize(rgb_frame, 0);
	float *blob_data = (float *)(this->input_buffer_[0].get());
	blobFromImage(pr_img, blob_data);

	size_t input_size = input_w_[0] * input_h_[0] * 3;
	for (size_t i = 0; i < input_size; ++i) {
		blob_data[i] /= 255.0f;
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
	auto output_tensors = this->session_.Run(
		run_options, input_names.data(), this->input_tensor_.data(),
		this->input_tensor_.size(), output_names.data(), output_names.size());

	float scale = std::fminf((float)input_w_[0] / (float)frame.cols,
				 (float)input_h_[0] / (float)frame.rows);

	const float *out = output_tensors[0].GetTensorData<float>();
	auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

	int dim1 = (int)shape[1];
	int dim2 = (int)shape[2];

	std::vector<Object> objects;

	if (dim1 == 5) {
		// Lindevs' model format [1, 5, 8400]: cx, cy, w, h, score
		int num_anchors = dim2;
		for (int i = 0; i < num_anchors; i++) {
			float score = out[4 * num_anchors + i];
			if (score < this->bbox_conf_thresh_)
				continue;

			float cx = out[0 * num_anchors + i] / scale;
			float cy = out[1 * num_anchors + i] / scale;
			float w = out[2 * num_anchors + i] / scale;
			float h = out[3 * num_anchors + i] / scale;

			Object obj;
			obj.rect = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
			obj.prob = score;
			obj.label = 0;

			for (int j = 0; j < 5; j++) {
				obj.landmarks[j].x = cx;
				obj.landmarks[j].y = cy;
			}
			objects.push_back(obj);
		}

		ONNXRuntimeModel::qsort_descent_inplace(objects);
		std::vector<int> picked;
		ONNXRuntimeModel::nms_sorted_bboxes(objects, picked, this->nms_thresh_);

		std::vector<Object> final_objects;
		for (size_t i = 0; i < picked.size(); ++i) {
			final_objects.push_back(objects[picked[i]]);
		}
		return final_objects;

	} else {
		// Yakhyo's model format [1, 300, 21] (NMS baked-in)
		int num_boxes = dim1;
		int num_features = dim2;

		for (int i = 0; i < num_boxes; i++) {
			float score = out[i * num_features + 4];
			if (score < this->bbox_conf_thresh_)
				continue;

			float x1 = out[i * num_features + 0] / scale;
			float y1 = out[i * num_features + 1] / scale;
			float x2 = out[i * num_features + 2] / scale;
			float y2 = out[i * num_features + 3] / scale;

			Object obj;
			obj.rect = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
			obj.prob = score;
			obj.label = 0;

			for (int j = 0; j < 5; j++) {
				obj.landmarks[j].x = out[i * num_features + 6 + j * 3] / scale;
				obj.landmarks[j].y = out[i * num_features + 6 + j * 3 + 1] / scale;
			}
			objects.push_back(obj);
		}
		return objects;
	}
}

#ifdef _WIN32
#include <obs.h>

std::vector<Object> YOLOv8FaceONNX::inference_gpu()
{
	if (!gpu_pipeline_ready_) {
		throw std::runtime_error("GPU pipeline is not ready");
	}

	try {
		// 1. Reset command allocator & list
		HRESULT hr = d3d12Allocator_->Reset();
		if (FAILED(hr)) {
			throw std::runtime_error("Failed to reset D3D12 Command Allocator");
		}

		hr = d3d12CommandList_->Reset(d3d12Allocator_, pipelineState_);
		if (FAILED(hr)) {
			throw std::runtime_error("Failed to reset D3D12 Command List");
		}

		// 2. Acquire D3D12 Keyed Mutex (Key 1)
		hr = d3d12KeyedMutex_->AcquireSync(1, 33); // 33ms timeout
		if (FAILED(hr)) {
			if (hr == WAIT_TIMEOUT) {
				throw std::runtime_error("D3D12 Keyed Mutex acquire timeout");
			} else {
				throw std::runtime_error("Failed to acquire D3D12 Keyed Mutex");
			}
		}

		// 3. Record preprocessing dispatch commands
		d3d12CommandList_->SetComputeRootSignature(rootSignature_);
		d3d12CommandList_->SetDescriptorHeaps(1, &srvUavHeap_);
		d3d12CommandList_->SetComputeRootConstantBufferView(0, constantBuffer_->GetGPUVirtualAddress());

		D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
		UINT incrementSize = d3d12Device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		d3d12CommandList_->SetComputeRootDescriptorTable(1, gpuHandle); // t0
		gpuHandle.ptr += incrementSize;
		d3d12CommandList_->SetComputeRootDescriptorTable(2, gpuHandle); // u0

		uint32_t dispatchX = (input_w_[0] + 15) / 16;
		uint32_t dispatchY = (input_h_[0] + 15) / 16;
		d3d12CommandList_->Dispatch(dispatchX, dispatchY, 1);

		// 4. Release D3D12 Keyed Mutex (Key 0)
		hr = d3d12CommandList_->Close();
		if (FAILED(hr)) {
			d3d12KeyedMutex_->ReleaseSync(0);
			throw std::runtime_error("Failed to close D3D12 Command List");
		}

		ID3D12CommandList* cmdLists[] = { d3d12CommandList_ };
		d3d12Queue_->ExecuteCommandLists(1, cmdLists);

		d3d12KeyedMutex_->ReleaseSync(0);

		// 5. DirectML Inference using IoBinding
		ioBinding_.ClearBoundOutputs();
		ioBinding_.BindOutput(output_name_[0].c_str(), output_tensor_[0]);

		Ort::RunOptions run_options;
		session_.Run(run_options, ioBinding_);

		// 6. Post-processing
		const float* out = output_tensor_[0].GetTensorData<float>();
		auto shape = output_tensor_[0].GetTensorTypeAndShapeInfo().GetShape();

		int dim1 = (int)shape[1];
		int dim2 = (int)shape[2];

		float scale = std::fminf((float)input_w_[0] / (float)sharedWidth_,
					 (float)input_h_[0] / (float)sharedHeight_);

		std::vector<Object> objects;
		if (dim1 == 5) {
			int num_anchors = dim2;
			for (int i = 0; i < num_anchors; i++) {
				float score = out[4 * num_anchors + i];
				if (score < this->bbox_conf_thresh_)
					continue;

				float cx = out[0 * num_anchors + i] / scale;
				float cy = out[1 * num_anchors + i] / scale;
				float w = out[2 * num_anchors + i] / scale;
				float h = out[3 * num_anchors + i] / scale;

				Object obj;
				obj.rect = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
				obj.prob = score;
				obj.label = 0;

				for (int j = 0; j < 5; j++) {
					obj.landmarks[j].x = cx;
					obj.landmarks[j].y = cy;
				}
				objects.push_back(obj);
			}

			ONNXRuntimeModel::qsort_descent_inplace(objects);
			std::vector<int> picked;
			ONNXRuntimeModel::nms_sorted_bboxes(objects, picked, this->nms_thresh_);

			std::vector<Object> final_objects;
			for (size_t i = 0; i < picked.size(); ++i) {
				final_objects.push_back(objects[picked[i]]);
			}
			return final_objects;

		} else {
			int num_boxes = dim1;
			int num_features = dim2;

			for (int i = 0; i < num_boxes; i++) {
				float score = out[i * num_features + 4];
				if (score < this->bbox_conf_thresh_)
					continue;

				float x1 = out[i * num_features + 0] / scale;
				float y1 = out[i * num_features + 1] / scale;
				float x2 = out[i * num_features + 2] / scale;
				float y2 = out[i * num_features + 3] / scale;

				Object obj;
				obj.rect = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
				obj.prob = score;
				obj.label = 0;

				for (int j = 0; j < 5; j++) {
					obj.landmarks[j].x = out[i * num_features + 6 + j * 3] / scale;
					obj.landmarks[j].y = out[i * num_features + 6 + j * 3 + 1] / scale;
				}
				objects.push_back(obj);
			}
			return objects;
		}

	} catch (const std::exception& e) {
		obs_log(LOG_ERROR, "GPU Zero-Copy Inference exception: %s. Falling back to CPU.", e.what());
		gpu_pipeline_ready_ = false;
		release_gpu_resources();
		throw;
	}
}
#endif

