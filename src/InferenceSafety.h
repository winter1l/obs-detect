#ifndef INFERENCE_SAFETY_H
#define INFERENCE_SAFETY_H

struct InferenceFrameRequirements {
	bool cropEnabled = false;
	bool faceExclusionEnabled = false;
	bool previewEnabled = false;
	bool debugEnabled = false;
	bool faceStatsEnabled = false;
	bool faceDebugEnabled = false;
};

inline bool requiresOwnedInferenceFrame(const InferenceFrameRequirements &requirements)
{
	return requirements.cropEnabled || requirements.faceExclusionEnabled ||
	       requirements.previewEnabled || requirements.debugEnabled ||
	       requirements.faceStatsEnabled || requirements.faceDebugEnabled;
}

inline bool canUseGpuZeroCopy(bool isDirectML, bool modelSupportsZeroCopy, bool pipelineReady,
			      bool runtimeHealthy, const InferenceFrameRequirements &requirements)
{
	return isDirectML && modelSupportsZeroCopy && pipelineReady && runtimeHealthy &&
	       !requiresOwnedInferenceFrame(requirements);
}

#endif // INFERENCE_SAFETY_H
