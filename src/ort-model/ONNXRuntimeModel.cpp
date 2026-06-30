#include "ONNXRuntimeModel.h"

#ifdef _WIN32
#include <dml_provider_factory.h>
#endif

#include "plugin-support.h"

#include <obs.h>

ONNXRuntimeModel::ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads,
				   int num_classes, int inter_op_num_threads,
				   const std::string &use_gpu_, int device_id, bool use_parallel,
				   float nms_th, float conf_th)
	: intra_op_num_threads_(intra_op_num_threads),
	  inter_op_num_threads_(inter_op_num_threads),
	  use_gpu(use_gpu_),
	  device_id_(device_id),
	  use_parallel_(use_parallel),
	  nms_thresh_(nms_th),
	  bbox_conf_thresh_(conf_th),
	  num_classes_(num_classes)
{
	try {
		Ort::SessionOptions session_options;

		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		if (this->use_parallel_) {
			session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
			session_options.SetInterOpNumThreads(this->inter_op_num_threads_);
		} else {
			session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
		}
		session_options.SetIntraOpNumThreads(this->intra_op_num_threads_);

#ifdef _WIN32
		if (this->use_gpu == "cuda") {
			OrtCUDAProviderOptions cuda_option;
			cuda_option.device_id = this->device_id_;
			session_options.AppendExecutionProvider_CUDA(cuda_option);
		}
		if (this->use_gpu == "dml") {
			auto &api = Ort::GetApi();
			OrtDmlApi *dmlApi = nullptr;
			Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION,
								      (const void **)&dmlApi));
			Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(
				session_options, 0));
		}
#endif

		this->session_ = Ort::Session(this->env_, path_to_model.c_str(), session_options);
	} catch (std::exception &e) {
		obs_log(LOG_ERROR, "Cannot load model: %s", e.what());
		throw e;
	}

	Ort::AllocatorWithDefaultOptions ort_alloc;

	// number of inputs
	size_t num_input = this->session_.GetInputCount();

	for (size_t i = 0; i < num_input; i++) {
		auto input_info = this->session_.GetInputTypeInfo(i);
		auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
		auto input_shape = input_shape_info.GetShape();
		auto input_tensor_type = input_shape_info.GetElementType();

		// assume input shape is NCHW
		if (input_shape.size() > 2 && input_shape[2] <= 0) {
			input_shape[2] = 640;
		}
		if (input_shape.size() > 3 && input_shape[3] <= 0) {
			input_shape[3] = 640;
		}

		this->input_h_.push_back((int)(input_shape[2]));
		this->input_w_.push_back((int)(input_shape[3]));

		// Allocate input memory buffer
		this->input_name_.push_back(
			std::string(this->session_.GetInputNameAllocated(i, ort_alloc).get()));

		size_t element_count = 1;
		for (auto dim : input_shape) {
			if (dim <= 0) dim = 1;
			element_count *= dim;
		}
		size_t input_byte_count = sizeof(float) * element_count;

		std::unique_ptr<uint8_t[]> input_buffer =
			std::make_unique<uint8_t[]>(input_byte_count);
		auto input_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this->input_tensor_.push_back(Ort::Value::CreateTensor(
			input_memory_info, input_buffer.get(), input_byte_count, input_shape.data(),
			input_shape.size(), input_tensor_type));
		this->input_buffer_.push_back(std::move(input_buffer));

		obs_log(LOG_INFO, "Input name: %s", this->input_name_[i].c_str());
		obs_log(LOG_INFO, "Input shape: %d %d %d %d", input_shape[0],
			input_shape.size() > 1 ? input_shape[1] : 0,
			input_shape.size() > 2 ? input_shape[2] : 0,
			input_shape.size() > 3 ? input_shape[3] : 0);
	}

	// number of outputs
	size_t num_output = this->session_.GetOutputCount();

	for (size_t i = 0; i < num_output; i++) {
		auto output_info = this->session_.GetOutputTypeInfo(i);
		auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
		auto output_shape = output_shape_info.GetShape();
		auto output_tensor_type = output_shape_info.GetElementType();

		this->output_shapes_.push_back(output_shape);

		// Allocate output memory buffer
		size_t output_byte_count = sizeof(float) * output_shape_info.GetElementCount();
		std::unique_ptr<uint8_t[]> output_buffer =
			std::make_unique<uint8_t[]>(output_byte_count);
		auto output_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this->output_tensor_.push_back(Ort::Value::CreateTensor(
			output_memory_info, output_buffer.get(), output_byte_count,
			output_shape.data(), output_shape.size(), output_tensor_type));
		this->output_buffer_.push_back(std::move(output_buffer));

		this->output_name_.push_back(
			std::string(this->session_.GetOutputNameAllocated(i, ort_alloc).get()));

		obs_log(LOG_INFO, "Output name: %s", this->output_name_[i].c_str());
		obs_log(LOG_INFO, "Output shape: %d %d %d %d", output_shape[0],
			output_shape.size() > 1 ? output_shape[1] : 0,
			output_shape.size() > 2 ? output_shape[2] : 0,
			output_shape.size() > 3 ? output_shape[3] : 0);
	}
}

cv::Mat ONNXRuntimeModel::static_resize(const cv::Mat &img, const int input_index)
{
	float r = std::fminf((float)input_w_[input_index] / (float)img.cols,
			     (float)input_h_[input_index] / (float)img.rows);
	// r = std::min(r, 1.0f);
	int unpad_w = (int)(r * (float)img.cols);
	int unpad_h = (int)(r * (float)img.rows);
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	cv::Mat out(input_h_[input_index], input_w_[input_index], CV_8UC3,
		    cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
	return out;
}

// for NCHW
void ONNXRuntimeModel::blobFromImage(const cv::Mat &img, float *blob_data)
{
	size_t channels = 3;
	size_t img_h = img.rows;
	size_t img_w = img.cols;
	for (size_t c = 0; c < channels; ++c) {
		for (size_t h = 0; h < img_h; ++h) {
			for (size_t w = 0; w < img_w; ++w) {
				blob_data[(int)(c * img_w * img_h + h * img_w + w)] =
					(float)img.ptr<cv::Vec3b>((int)h)[(int)w][(int)c];
			}
		}
	}
}

// for NHWC
void ONNXRuntimeModel::blobFromImage_nhwc(const cv::Mat &img, float *blob_data)
{
	size_t channels = 3;
	size_t img_h = img.rows;
	size_t img_w = img.cols;
	for (size_t i = 0; i < img_h * img_w; ++i) {
		for (size_t c = 0; c < channels; ++c) {
			blob_data[i * channels + c] = (float)img.data[i * channels + c];
		}
	}
}

float ONNXRuntimeModel::intersection_area(const Object &a, const Object &b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j) {
		while (faceobjects[i].prob > p)
			++i;

		while (faceobjects[j].prob < p)
			--j;

		if (i <= j) {
			std::swap(faceobjects[i], faceobjects[j]);

			++i;
			--j;
		}
	}
	if (left < j)
		qsort_descent_inplace(faceobjects, left, j);
	if (i < right)
		qsort_descent_inplace(faceobjects, i, right);
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector<Object> &objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, (int)(objects.size() - 1));
}

void ONNXRuntimeModel::nms_sorted_bboxes(const std::vector<Object> &objects,
					 std::vector<int> &picked, const float nms_threshold)
{
	picked.clear();

	const size_t n = objects.size();

	std::vector<float> areas(n);
	for (size_t i = 0; i < n; ++i) {
		areas[i] = objects[i].rect.area();
	}

	for (size_t i = 0; i < n; ++i) {
		const Object &a = objects[i];
		const size_t picked_size = picked.size();

		int keep = 1;
		for (size_t j = 0; j < picked_size; ++j) {
			const Object &b = objects[picked[j]];

			// intersection over union
			float inter_area = this->intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back((int)i);
	}
}

void ONNXRuntimeModel::inference(const cv::Mat &frame, const int input_index)
{
	// preprocess
	cv::Mat pr_img = this->static_resize(frame, input_index);

	float *blob_data = (float *)(this->input_buffer_[input_index].get());
	blobFromImage(pr_img, blob_data);

	// input names
	std::vector<const char *> input_names;
	for (size_t i = 0; i < this->input_name_.size(); i++) {
		input_names.push_back(this->input_name_[i].c_str());
	}

	// output names
	std::vector<const char *> output_names;
	for (size_t i = 0; i < this->output_name_.size(); i++) {
		output_names.push_back(this->output_name_[i].c_str());
	}

	// Inference
	Ort::RunOptions run_options;
	this->session_.Run(run_options, input_names.data(), this->input_tensor_.data(),
			   this->input_tensor_.size(), output_names.data(),
			   this->output_tensor_.data(), this->output_tensor_.size());
}

#ifdef _WIN32
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <DirectML.h>
#include <dml_provider_factory.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// Preprocess constant buffer alignment
struct GPUPreprocessConfig {
	uint32_t SrcWidth;
	uint32_t SrcHeight;
	uint32_t DstWidth;
	uint32_t DstHeight;
	uint32_t UnpadWidth;
	uint32_t UnpadHeight;
	float Padding0; // alignment (Dummy0.x)
	float Padding1; // alignment (Dummy0.y)
	float MeanR;    // Mean.x
	float MeanG;    // Mean.y
	float MeanB;    // Mean.z
	float Padding2; // alignment (Dummy1)
	float StdR;     // Std.x
	float StdG;     // Std.y
	float StdB;     // Std.z
	float Padding3; // alignment (Dummy2)
};

static const char* preprocess_shader_source = R"(
Texture2D<float4> InputTexture : register(t0);
SamplerState BilinearSampler : register(s0);
RWStructuredBuffer<float> OutputBuffer : register(u0);

cbuffer PreprocessParams : register(b0)
{
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;
    uint UnpadWidth;
    uint UnpadHeight;
    float2 Dummy0;
    float3 Mean;
    float Dummy1;
    float3 Std;
    float Dummy2;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x;
    uint y = dtid.y;

    if (x >= DstWidth || y >= DstHeight)
        return;

    float3 pixel;

    if (x < UnpadWidth && y < UnpadHeight)
    {
        float u = (x + 0.5f) / (float)UnpadWidth;
        float v = (y + 0.5f) / (float)UnpadHeight;

        float4 color = InputTexture.SampleLevel(BilinearSampler, float2(u, v), 0);
        pixel = color.rgb;
    }
    else
    {
        pixel = float3(114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f);
    }

    float3 normalized = (pixel - Mean) / Std;

    uint channelSize = DstWidth * DstHeight;
    uint baseIndex = y * DstWidth + x;

    // Input format is B8G8R8A8_UNORM, so color.r = B, color.g = G, color.b = R
    // We want the output tensor to be in RGB format (NCHW)
    OutputBuffer[0 * channelSize + baseIndex] = normalized.b; // Red
    OutputBuffer[1 * channelSize + baseIndex] = normalized.g; // Green
    OutputBuffer[2 * channelSize + baseIndex] = normalized.r; // Blue
}
)";

bool ONNXRuntimeModel::initialize_gpu(ID3D11Device* d3d11Device, int width, int height)
{
	if (gpu_pipeline_initialized_) return gpu_pipeline_ready_;
	gpu_pipeline_initialized_ = true;

	try {
		// 1. Create D3D12 Device
		HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device_));
		if (FAILED(hr)) {
			obs_log(LOG_WARNING, "GPU Zero-Copy: Failed to create D3D12 device (hr=0x%08X). Falling back to CPU.", hr);
			return false;
		}

		// 2. Create Command Queue
		D3D12_COMMAND_QUEUE_DESC queueDesc = {};
		queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		hr = d3d12Device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&d3d12Queue_));
		if (FAILED(hr)) return false;

		// 3. Create Shared D3D11 Texture
		D3D11_TEXTURE2D_DESC d3d11Desc = {};
		d3d11Desc.Width = width;
		d3d11Desc.Height = height;
		d3d11Desc.MipLevels = 1;
		d3d11Desc.ArraySize = 1;
		d3d11Desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		d3d11Desc.SampleDesc.Count = 1;
		d3d11Desc.SampleDesc.Quality = 0;
		d3d11Desc.Usage = D3D11_USAGE_DEFAULT;
		d3d11Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		d3d11Desc.CPUAccessFlags = 0;
		d3d11Desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

		hr = d3d11Device->CreateTexture2D(&d3d11Desc, nullptr, &d3d11SharedTexture_);
		if (FAILED(hr)) {
			obs_log(LOG_WARNING, "GPU Zero-Copy: Failed to create shared D3D11 texture (hr=0x%08X).", hr);
			return false;
		}

		hr = d3d11SharedTexture_->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&d3d11KeyedMutex_);
		if (FAILED(hr)) return false;

		IDXGIResource1* dxgiResource = nullptr;
		hr = d3d11SharedTexture_->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgiResource);
		if (FAILED(hr)) return false;

		hr = dxgiResource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &sharedHandle_);
		dxgiResource->Release();
		if (FAILED(hr)) return false;

		// 4. Open Shared Handle on D3D12
		hr = d3d12Device_->OpenSharedHandle(sharedHandle_, IID_PPV_ARGS(&d3d12SharedResource_));
		if (FAILED(hr)) {
			obs_log(LOG_WARNING, "GPU Zero-Copy: OpenSharedHandle failed in D3D12 (hr=0x%08X).", hr);
			return false;
		}

		hr = d3d12SharedResource_->QueryInterface(IID_PPV_ARGS(&d3d12KeyedMutex_));
		if (FAILED(hr)) return false;

		// 5. Create Command List and Allocator
		hr = d3d12Device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&d3d12Allocator_));
		if (FAILED(hr)) return false;

		hr = d3d12Device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, d3d12Allocator_, nullptr, IID_PPV_ARGS(&d3d12CommandList_));
		if (FAILED(hr)) return false;
		d3d12CommandList_->Close();

		// 6. Fence setup
		hr = d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d3d12Fence_));
		if (FAILED(hr)) return false;
		d3d12FenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
		if (!d3d12FenceEvent_) return false;

		// 7. Compile Compute Shader
		ID3DBlob* shaderBlob = nullptr;
		ID3DBlob* errorBlob = nullptr;
		hr = D3DCompile(preprocess_shader_source, strlen(preprocess_shader_source), nullptr, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &shaderBlob, &errorBlob);
		if (FAILED(hr)) {
			if (errorBlob) {
				obs_log(LOG_ERROR, "Shader compile error: %s", (char*)errorBlob->GetBufferPointer());
				errorBlob->Release();
			}
			return false;
		}

		// 8. Create Root Signature
		D3D12_DESCRIPTOR_RANGE descriptorRanges[2] = {};
		descriptorRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		descriptorRanges[0].NumDescriptors = 1;
		descriptorRanges[0].BaseShaderRegister = 0; // t0
		descriptorRanges[0].RegisterSpace = 0;
		descriptorRanges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		descriptorRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		descriptorRanges[1].NumDescriptors = 1;
		descriptorRanges[1].BaseShaderRegister = 0; // u0
		descriptorRanges[1].RegisterSpace = 0;
		descriptorRanges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

		D3D12_ROOT_PARAMETER rootParameters[3] = {};
		// rootParameters[0]: Constant Buffer View (b0)
		rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
		rootParameters[0].Descriptor.ShaderRegister = 0; // b0
		rootParameters[0].Descriptor.RegisterSpace = 0;
		rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		// rootParameters[1]: SRV Table (t0)
		rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
		rootParameters[1].DescriptorTable.pDescriptorRanges = &descriptorRanges[0];
		rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		// rootParameters[2]: UAV Table (u0)
		rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
		rootParameters[2].DescriptorTable.pDescriptorRanges = &descriptorRanges[1];
		rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		// Static Sampler for Bilinear Filter
		D3D12_STATIC_SAMPLER_DESC staticSampler = {};
		staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
		staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		staticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
		staticSampler.MipLODBias = 0.0f;
		staticSampler.MaxAnisotropy = 1;
		staticSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
		staticSampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
		staticSampler.MinLOD = 0.0f;
		staticSampler.MaxLOD = D3D12_FLOAT32_MAX;
		staticSampler.ShaderRegister = 0; // s0
		staticSampler.RegisterSpace = 0;
		staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
		rootSigDesc.NumParameters = 3;
		rootSigDesc.pParameters = rootParameters;
		rootSigDesc.NumStaticSamplers = 1;
		rootSigDesc.pStaticSamplers = &staticSampler;
		rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

		ID3DBlob* serializedRootSig = nullptr;
		ID3DBlob* rootSigErrorBlob = nullptr;
		hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSig, &rootSigErrorBlob);
		if (FAILED(hr)) return false;

		hr = d3d12Device_->CreateRootSignature(0, serializedRootSig->GetBufferPointer(), serializedRootSig->GetBufferSize(), IID_PPV_ARGS(&rootSignature_));
		serializedRootSig->Release();
		if (FAILED(hr)) return false;

		// 9. Create Pipeline State Object (PSO)
		D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.pRootSignature = rootSignature_;
		psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
		psoDesc.CS.BytecodeLength = shaderBlob->GetBufferSize();
		psoDesc.NodeMask = 0;
		psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

		hr = d3d12Device_->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState_));
		shaderBlob->Release();
		if (FAILED(hr)) return false;

		// 10. Create Constant Buffer
		D3D12_HEAP_PROPERTIES cbHeapProp = {};
		cbHeapProp.Type = D3D12_HEAP_TYPE_UPLOAD;
		D3D12_RESOURCE_DESC cbDesc = {};
		cbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		cbDesc.Alignment = 0;
		cbDesc.Width = (sizeof(GPUPreprocessConfig) + 255) & ~255;
		cbDesc.Height = 1;
		cbDesc.DepthOrArraySize = 1;
		cbDesc.MipLevels = 1;
		cbDesc.Format = DXGI_FORMAT_UNKNOWN;
		cbDesc.SampleDesc.Count = 1;
		cbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		cbDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

		hr = d3d12Device_->CreateCommittedResource(&cbHeapProp, D3D12_HEAP_FLAG_NONE, &cbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&constantBuffer_));
		if (FAILED(hr)) return false;

		// 11. Create UAV Buffer for CS Output (NCHW float format)
		uint32_t dstW = input_w_[0];
		uint32_t dstH = input_h_[0];
		uint32_t uavBufferSize = dstW * dstH * 3 * sizeof(float);

		D3D12_HEAP_PROPERTIES uavHeapProp = {};
		uavHeapProp.Type = D3D12_HEAP_TYPE_DEFAULT;
		D3D12_RESOURCE_DESC uavDesc = {};
		uavDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uavDesc.Alignment = 0;
		uavDesc.Width = uavBufferSize;
		uavDesc.Height = 1;
		uavDesc.DepthOrArraySize = 1;
		uavDesc.MipLevels = 1;
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.SampleDesc.Count = 1;
		uavDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		uavDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		hr = d3d12Device_->CreateCommittedResource(&uavHeapProp, D3D12_HEAP_FLAG_NONE, &uavDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uavBuffer_));
		if (FAILED(hr)) return false;

		// 12. Create Descriptor Heap (for SRV and UAV table)
		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = 2; // 1 SRV, 1 UAV
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		hr = d3d12Device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&srvUavHeap_));
		if (FAILED(hr)) return false;

		D3D12_CPU_DESCRIPTOR_HANDLE handle = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
		UINT incrementSize = d3d12Device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// SRV descriptor
		D3D12_SHADER_RESOURCE_VIEW_DESC srvViewDesc = {};
		srvViewDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		srvViewDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvViewDesc.Texture2D.MipLevels = 1;
		srvViewDesc.Texture2D.MostDetailedMip = 0;
		d3d12Device_->CreateShaderResourceView(d3d12SharedResource_, &srvViewDesc, handle);

		// UAV descriptor
		handle.ptr += incrementSize;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavViewDesc = {};
		uavViewDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavViewDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavViewDesc.Buffer.FirstElement = 0;
		uavViewDesc.Buffer.NumElements = dstW * dstH * 3;
		uavViewDesc.Buffer.StructureByteStride = sizeof(float);
		d3d12Device_->CreateUnorderedAccessView(uavBuffer_, nullptr, &uavViewDesc, handle);

		// 13. Map Constant Buffer settings
		float r = std::fminf((float)dstW / (float)width, (float)dstH / (float)height);
		uint32_t unpadW = (uint32_t)(r * (float)width);
		uint32_t unpadH = (uint32_t)(r * (float)height);

		GPUPreprocessConfig cbData = {};
		cbData.SrcWidth = (uint32_t)width;
		cbData.SrcHeight = (uint32_t)height;
		cbData.DstWidth = dstW;
		cbData.DstHeight = dstH;
		cbData.UnpadWidth = unpadW;
		cbData.UnpadHeight = unpadH;
		cbData.Padding0 = 0.0f;
		cbData.Padding1 = 0.0f;
		cbData.MeanR = mean_[0];
		cbData.MeanG = mean_[1];
		cbData.MeanB = mean_[2];
		cbData.Padding2 = 0.0f;
		cbData.StdR = std_[0];
		cbData.StdG = std_[1];
		cbData.StdB = std_[2];
		cbData.Padding3 = 0.0f;

		void* cbMapped = nullptr;
		hr = constantBuffer_->Map(0, nullptr, &cbMapped);
		if (SUCCEEDED(hr)) {
			memcpy(cbMapped, &cbData, sizeof(cbData));
			constantBuffer_->Unmap(0, nullptr);
		} else {
			return false;
		}

		// 14. Initialize DirectML using session device and queue
		dmlDll_ = LoadLibraryA("DirectML.dll");
		if (!dmlDll_) {
			obs_log(LOG_WARNING, "GPU Zero-Copy: DirectML.dll not found. Falling back to CPU.");
			return false;
		}

		typedef HRESULT (WINAPI* PFN_DML_CREATE_DEVICE)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
		PFN_DML_CREATE_DEVICE dmlCreateDevice = (PFN_DML_CREATE_DEVICE)GetProcAddress(dmlDll_, "DMLCreateDevice");
		if (!dmlCreateDevice) return false;

		hr = dmlCreateDevice(d3d12Device_, DML_CREATE_DEVICE_FLAG_NONE, __uuidof(IDMLDevice), &dmlDevice_);
		if (FAILED(hr)) return false;

		// 15. Setup ONNX Runtime DirectML EP session and IO Binding
		const OrtApi& ortApi = Ort::GetApi();
		const OrtDmlApi* dmlApi = nullptr;
		OrtStatus* status = ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dmlApi);
		if (status != nullptr) {
			ortApi.ReleaseStatus(status);
			return false;
		}

		// Create GPU Allocation for output NCHW buffer
		status = dmlApi->CreateGPUAllocationFromD3DResource(uavBuffer_, &inputGpuAllocation_);
		if (status != nullptr) {
			ortApi.ReleaseStatus(status);
			obs_log(LOG_WARNING, "GPU Zero-Copy: CreateGPUAllocationFromD3DResource failed.");
			return false;
		}

		// Ort::Value wrapping for IO Binding
		Ort::Value inputTensor((OrtValue*)inputGpuAllocation_);
		ioBinding_ = Ort::IoBinding(session_);
		ioBinding_.BindInput(input_name_[0].c_str(), inputTensor);

		d3d11Device_ = d3d11Device;
		sharedWidth_ = width;
		sharedHeight_ = height;
		gpu_pipeline_ready_ = true;
		obs_log(LOG_INFO, "GPU Zero-Copy pipeline successfully initialized.");
		return true;

	} catch (const std::exception& e) {
		obs_log(LOG_ERROR, "GPU Zero-Copy Initialization exception: %s", e.what());
		release_gpu_resources();
		return false;
	}
}

bool ONNXRuntimeModel::copy_d3d11_texture(gs_texture_t *obsSourceTexture)
{
	if (!gpu_pipeline_ready_ || !d3d11KeyedMutex_ || !d3d11Device_) return false;

	ID3D11Texture2D* obsTex = (ID3D11Texture2D*)gs_texture_get_obj(obsSourceTexture);
	if (!obsTex) return false;

	// Acquire sync write lock (Key 0)
	HRESULT hr = d3d11KeyedMutex_->AcquireSync(0, 33); // 33ms timeout to prevent deadlock
	if (FAILED(hr)) {
		if (hr == WAIT_TIMEOUT) {
			obs_log(LOG_WARNING, "GPU Zero-Copy: D3D11 Keyed Mutex acquire timeout.");
		}
		return false;
	}

	ID3D11DeviceContext* context = nullptr;
	d3d11Device_->GetImmediateContext(&context);
	if (context) {
		context->CopyResource(d3d11SharedTexture_, obsTex);
		context->Release();
	}

	d3d11KeyedMutex_->ReleaseSync(1); // Release for D3D12 read
	return true;
}

void ONNXRuntimeModel::release_gpu_resources()
{
	gpu_pipeline_ready_ = false;

	const OrtApi& ortApi = Ort::GetApi();
	const OrtDmlApi* dmlApi = nullptr;
	if (ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dmlApi) == nullptr) {
		if (dmlApi && inputGpuAllocation_) {
			dmlApi->FreeGPUAllocation(inputGpuAllocation_);
		}
	}
	inputGpuAllocation_ = nullptr;

	if (dmlDevice_) {
		((IUnknown*)dmlDevice_)->Release();
		dmlDevice_ = nullptr;
	}
	if (dmlDll_) {
		FreeLibrary(dmlDll_);
		dmlDll_ = nullptr;
	}

	if (srvUavHeap_) { srvUavHeap_->Release(); srvUavHeap_ = nullptr; }
	if (uavBuffer_) { uavBuffer_->Release(); uavBuffer_ = nullptr; }
	if (constantBuffer_) { constantBuffer_->Release(); constantBuffer_ = nullptr; }
	if (pipelineState_) { pipelineState_->Release(); pipelineState_ = nullptr; }
	if (rootSignature_) { rootSignature_->Release(); rootSignature_ = nullptr; }

	if (d3d12FenceEvent_) { CloseHandle(d3d12FenceEvent_); d3d12FenceEvent_ = nullptr; }
	if (d3d12Fence_) { d3d12Fence_->Release(); d3d12Fence_ = nullptr; }
	if (d3d12CommandList_) { d3d12CommandList_->Release(); d3d12CommandList_ = nullptr; }
	if (d3d12Allocator_) { d3d12Allocator_->Release(); d3d12Allocator_ = nullptr; }
	if (d3d12KeyedMutex_) { d3d12KeyedMutex_->Release(); d3d12KeyedMutex_ = nullptr; }
	if (d3d12SharedResource_) { d3d12SharedResource_->Release(); d3d12SharedResource_ = nullptr; }
	if (d3d12Queue_) { d3d12Queue_->Release(); d3d12Queue_ = nullptr; }
	if (d3d12Device_) { d3d12Device_->Release(); d3d12Device_ = nullptr; }

	if (sharedHandle_) { CloseHandle(sharedHandle_); sharedHandle_ = nullptr; }
	if (d3d11KeyedMutex_) { d3d11KeyedMutex_->Release(); d3d11KeyedMutex_ = nullptr; }
	if (d3d11SharedTexture_) { d3d11SharedTexture_->Release(); d3d11SharedTexture_ = nullptr; }
	
	d3d11Device_ = nullptr;
	gpu_pipeline_initialized_ = false;
}
#endif

