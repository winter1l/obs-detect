#ifndef FACE_INFERENCE_QUEUE_H
#define FACE_INFERENCE_QUEUE_H

#include "FaceStateStore.h"
#include "ort-model/types.hpp"

#include <opencv2/core/mat.hpp>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

struct FaceInferenceRequest {
	Object object;
	uint64_t checkToken = 0;
};

struct FaceInferenceJob {
	cv::Mat imageBGRA;
	std::vector<FaceInferenceRequest> requests;
	uint64_t frameId = 0;
	bool priority = false;
	float matchThreshold = 0.6f;
	int maxExemptPersons = 1;
	int settingsVersion = 0;
	FaceRetryPolicy retryPolicy;
};

class FaceInferenceQueue {
public:
	explicit FaceInferenceQueue(size_t maxJobs = 4);

	bool tryPush(FaceInferenceJob job);
	bool tryPop(FaceInferenceJob &job);
	bool waitPop(FaceInferenceJob &job, const std::atomic<bool> &stopRequested);
	void clear();
	void notifyAll();
	size_t size() const;

private:
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::deque<FaceInferenceJob> jobs_;
	size_t maxJobs_;
};

#endif // FACE_INFERENCE_QUEUE_H
