#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include <util/circlebuf.h>
#include "ort-model/ONNXRuntimeModel.h"
#include "sort/Sort.h"
#include "sort/StateHistory.h"
#include "yunet/YuNet.h"
#include "sface/SFace.h"
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <vector>
#include <map>
#include <mutex>
#include <deque>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>

struct DebugFaceBox {
	cv::Rect2f rect;
	std::chrono::steady_clock::time_point timestamp;
};

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
	std::unordered_map<uint64_t, float> faceSimilarityCache;
	std::vector<std::vector<float>> referenceFaceFeatures;

	gs_texture_t *previewTexture = nullptr;
	gs_texture_t *renderMaskTexture = nullptr;
	uint32_t lastTexWidth = 0;
	uint32_t lastTexHeight = 0;

	std::shared_ptr<yunet::YuNetONNX> yunetModel;
	std::shared_ptr<sface::SFaceONNX> sfaceModel;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
	int maskingFeather;
	int maskingDilateIterations;
	bool maskingDynamicExpansion;
	float maskingDynamicExpansionBase;
	float maskingDynamicExpansionRatio;
	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	bool sortTracking;
	bool showUnseenObjects;
	
	// Face Statistics
	bool enableFaceStats;
	bool enableFaceStatsLog;
	bool enableSimilarityLog;
	std::string faceStatsLogPath;
	std::mutex statsMutex;
	int statTotalChecks = 0;
	int statIsMe = 0;
	int statNotMe = 0;
	float statSumSimilarity = 0.0f;
	int similarityHistogram[10] = {0};
	std::string saveDetectionsPath;
	bool saveDetectionsAppend;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	// create SORT tracker
	Sort tracker;
	std::shared_ptr<StateHistoryManager> stateHistory;
	std::unordered_set<uint64_t> faceExemptIds;

	obs_source_t *source;
	gs_texrender_t *texrender;

	int videoDelayFrames;
	int lookaheadDelayFrames = 10;
	
	std::deque<gs_texture_t*> delayedTextures;
	std::deque<uint64_t> delayedFrameIds;
	uint64_t currentFrameId = 0;
	
	std::vector<gs_texture_t*> texturePool;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;
	gs_effect_t *pixelateEffect;
	gs_texture_t *baseTexture;

	bool useGpuZeroCopyCurrentFrame = false;
	cv::Mat inputBGRA;
	std::mutex inputBGRALock;

	// Audio sync
	struct circlebuf audioBuffers[MAX_AUDIO_CHANNELS];
	std::mutex audioMutex;
	float lastSamples[MAX_AUDIO_CHANNELS][2]; // For smoothing/crossfade
	uint64_t lastAudioPts = 0;
	bool resetAudio = false;

	std::map<uint64_t, cv::Mat> previewHistory;
	cv::Mat outputMask;
	std::vector<Object> latestObjects;

	// Debug
	bool debugMode;
	float currentInferenceTimeMs;
	float currentInferenceFPS;
	bool showYuNetDetections;
	std::vector<DebugFaceBox> debugFaceBoxes;
	std::mutex debugFaceMutex;

	bool isDisabled;
	bool preview;
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
	uint64_t inferenceFrameId;
	uint64_t frameCount;

	// Face Inference background thread
	std::thread faceInferenceThread;
	std::atomic<bool> stopFaceInferenceThread;
	std::mutex faceInferenceMutex;
	std::condition_variable faceInferenceCV;
	std::vector<Object> faceInferenceQueue;
	cv::Mat faceInferenceFrame;

	// File IO background thread
	std::thread ioThread;
	std::atomic<bool> stopIOThread;
	std::mutex ioMutex;
	std::condition_variable ioCV;
	std::vector<std::function<void()>> ioQueue;

	std::unique_ptr<ONNXRuntimeModel> onnxruntimemodel;
	std::vector<std::string> classNames;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */

void detect_filter_video_tick(void *data, float seconds);
void detect_filter_video_render(void *data, gs_effect_t *effect);
struct obs_audio_data *detect_filter_audio(void *data, struct obs_audio_data *audio);
