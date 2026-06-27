#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include "ort-model/ONNXRuntimeModel.h"
#include "sort/Sort.h"
#include "yunet/YuNet.h"
#include "sface/SFace.h"
#include <unordered_map>
#include <unordered_set>

/**
  * @brief The filter_data struct
  *
  * This struct is used to store the base data needed for ORT filters.
  *
*/
struct filter_data {
	std::string useGPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;

	int minAreaThreshold;
	int maxExemptPersons;
	int minHitFrames;
	int objectCategory;
	bool enableFaceExclusion;
	std::string referenceFacePath;
	float faceMatchThreshold;
	int personCategory;
	float minFaceAreaRatio;
	int faceInferenceInterval;

	enum FaceStatus { UNKNOWN, CHECKING, IS_ME, NOT_ME };
	std::unordered_map<uint64_t, FaceStatus> faceStatusCache;
	std::vector<std::vector<float>> referenceFaceFeatures;

	gs_texture_t *previewTexture = nullptr;
	gs_texture_t *renderMaskTexture = nullptr;
	uint32_t lastTexWidth = 0;
	uint32_t lastTexHeight = 0;

	std::unique_ptr<yunet::YuNetONNX> yunetModel;
	std::unique_ptr<sface::SFaceONNX> sfaceModel;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
	int maskingDilateIterations;
	bool maskingDynamicExpansion;
	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	bool sortTracking;
	bool showUnseenObjects;
	std::string saveDetectionsPath;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	// create SORT tracker
	Sort tracker;
	std::unordered_set<uint64_t> faceExemptIds;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;
	gs_effect_t *pixelateEffect;

	bool syncMode;

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;
	cv::Mat outputMask;
	std::vector<Object> latestObjects;

	// Debug
	bool debugMode;
	float currentInferenceTimeMs;
	float currentInferenceFPS;

	bool isDisabled;
	bool preview;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	// Inference background thread
	std::thread inferenceThread;
	std::atomic<bool> stopInferenceThread;
	std::mutex inferenceMutex;
	std::condition_variable inferenceCV;
	std::atomic<bool> isInferencing;
	cv::Mat inferenceInputFrame;
	cv::Rect inferenceCropRect;
	bool inferenceFrameReady;
	uint64_t frameCount;

	// Face Inference background thread
	std::thread faceInferenceThread;
	std::atomic<bool> stopFaceInferenceThread;
	std::mutex faceInferenceMutex;
	std::condition_variable faceInferenceCV;
	std::vector<Object> faceInferenceQueue;
	cv::Mat faceInferenceFrame;

	std::unique_ptr<ONNXRuntimeModel> onnxruntimemodel;
	std::vector<std::string> classNames;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
