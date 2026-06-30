#ifndef ONNXRUNTIME_MODEL_H
#define NOMINMAX
#define ONNXRUNTIME_MODEL_H

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <array>
#include <string>
#include <tuple>
#include <atomic>


#include "types.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11_1.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

// Fallback ID3D12KeyedMutex interface in case it is missing in the SDK
#ifndef __ID3D12KeyedMutex_INTERFACE_DEFINED__
#define __ID3D12KeyedMutex_INTERFACE_DEFINED__
MIDL_INTERFACE("d189fb70-5db6-4654-b5f7-64906f3ef21f")
ID3D12KeyedMutex : public ID3D12DeviceChild
{
public:
    virtual HRESULT STDMETHODCALLTYPE AcquireSync( 
        /* [in] */ UINT64 Key,
        /* [in] */ DWORD dwMilliseconds) = 0;
    
    virtual HRESULT STDMETHODCALLTYPE ReleaseSync( 
        /* [in] */ UINT64 Key) = 0;
};
#endif

#endif

struct gs_texture;
typedef struct gs_texture gs_texture_t;

// Generic class for ONNXRuntime models
class ONNXRuntimeModel {
public:
	// default constructor
	ONNXRuntimeModel() {}
	// constructor with parameters
	ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads, int num_classes,
			 int inter_op_num_threads = 1, const std::string &use_gpu_ = "",
			 int device_id = 0, bool use_parallel = false, float nms_th = 0.45f,
			 float conf_th = 0.3f);
	// virtual destructor
	virtual ~ONNXRuntimeModel() {
#ifdef _WIN32
		release_gpu_resources();
#endif
	}

	void setBBoxConfThresh(float thresh) { this->bbox_conf_thresh_ = thresh; }
	void setNmsThresh(float thresh) { this->nms_thresh_ = thresh; }

	virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

#ifdef _WIN32
	virtual std::vector<Object> inference_gpu() {
		return std::vector<Object>();
	}
	bool is_gpu_pipeline_ready() const { return gpu_pipeline_ready_; }
	bool is_gpu_pipeline_initialized() const { return gpu_pipeline_initialized_; }
	bool initialize_gpu(ID3D11Device* d3d11Device, int width, int height);
	bool copy_d3d11_texture(gs_texture_t *obsSourceTexture);
#endif

protected:
	cv::Mat static_resize(const cv::Mat &img, const int input_index);
	void blobFromImage(const cv::Mat &img, float *blob_data);
	void blobFromImage_nhwc(const cv::Mat &img, float *blob_data);
	float intersection_area(const Object &a, const Object &b);
	void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right);
	void qsort_descent_inplace(std::vector<Object> &objects);
	void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked,
			       const float nms_threshold);

	// run inference on the model with the given frame that should go in the input index
	void inference(const cv::Mat &frame, const int input_index);

	std::vector<int> input_w_;
	std::vector<int> input_h_;
	float nms_thresh_;
	float bbox_conf_thresh_;
	int num_classes_;
	bool use_parallel_;
	int inter_op_num_threads_;
	int intra_op_num_threads_;
	int device_id_;
	std::string use_gpu;
	std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
	std::vector<float> std_ = {0.229f, 0.224f, 0.225f};

	Ort::Session session_{nullptr};
	Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Default"};

	std::vector<Ort::Value> input_tensor_;
	std::vector<Ort::Value> output_tensor_;
	std::vector<std::string> input_name_;
	std::vector<std::string> output_name_;
	std::vector<std::unique_ptr<uint8_t[]>> input_buffer_;
	std::vector<std::unique_ptr<uint8_t[]>> output_buffer_;
	std::vector<Ort::ShapeInferContext::Ints> output_shapes_;

#ifdef _WIN32
	std::atomic<bool> gpu_pipeline_ready_{false};
	bool gpu_pipeline_initialized_ = false;

	// D3D11 Resources
	ID3D11Device* d3d11Device_ = nullptr;
	ID3D11Texture2D* d3d11SharedTexture_ = nullptr;
	IDXGIKeyedMutex* d3d11KeyedMutex_ = nullptr;
	HANDLE sharedHandle_ = nullptr;
	int sharedWidth_ = 0;
	int sharedHeight_ = 0;

	// D3D12 Resources
	ID3D12Device* d3d12Device_ = nullptr;
	ID3D12CommandQueue* d3d12Queue_ = nullptr;
	ID3D12Resource* d3d12SharedResource_ = nullptr;
	ID3D12KeyedMutex* d3d12KeyedMutex_ = nullptr;
	ID3D12CommandAllocator* d3d12Allocator_ = nullptr;
	ID3D12GraphicsCommandList* d3d12CommandList_ = nullptr;
	ID3D12Fence* d3d12Fence_ = nullptr;
	HANDLE d3d12FenceEvent_ = nullptr;
	uint64_t d3d12FenceValue_ = 0;

	// Compute Shader Resources
	ID3D12RootSignature* rootSignature_ = nullptr;
	ID3D12PipelineState* pipelineState_ = nullptr;
	ID3D12Resource* constantBuffer_ = nullptr;
	ID3D12Resource* uavBuffer_ = nullptr;
	ID3D12DescriptorHeap* srvUavHeap_ = nullptr;

	// DirectML Resources
	HMODULE dmlDll_ = nullptr;
	void* dmlDevice_ = nullptr; // IDMLDevice*
	Ort::IoBinding ioBinding_{nullptr};
	void* inputGpuAllocation_ = nullptr; // OrtOpaqueValue*

	void release_gpu_resources();
#endif
};

#endif // ONNXRUNTIME_MODEL_H
