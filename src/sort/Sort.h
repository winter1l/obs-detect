#ifndef SORT_H
#define SORT_H

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

#include "ort-model/types.hpp"

class Sort {
public:
	// Constructor
	Sort(size_t maxUnseenFrames = 5);

	// Destructor
	~Sort();

	// Update the tracking with detected objects
	std::vector<Object> update(const std::vector<Object> &detections);

	// Get the current tracked objects and their classes
	std::vector<Object> getTrackedObjects() const;

	// Set Max Unseen Frames
	void setMaxUnseenFrames(size_t maxUnseenFrames_)
	{
		this->maxUnseenFrames = maxUnseenFrames_;
	}

	// Get Max Unseen Frames
	size_t getMaxUnseenFrames() const { return this->maxUnseenFrames; }

	// Set Min Hit Frames
	void setMinHitFrames(int minHitFrames_)
	{
		this->minHitFrames = minHitFrames_;
	}

	// Get Min Hit Frames
	int getMinHitFrames() const { return this->minHitFrames; }

	// Set IoU Threshold
	void setIouThreshold(float iouThreshold_)
	{
		this->iouThreshold = iouThreshold_;
	}

	// Get IoU Threshold
	float getIouThreshold() const { return this->iouThreshold; }

	// Set Instant Track Area Ratio
	void setInstantTrackAreaRatio(float ratio)
	{
		this->instantTrackAreaRatio = ratio;
	}

	// Get Instant Track Area Ratio
	float getInstantTrackAreaRatio() const { return this->instantTrackAreaRatio; }

	// Set Screen Area
	void setScreenArea(float area)
	{
		this->screenArea = area;
	}

	// Get Screen Area
	float getScreenArea() const { return this->screenArea; }

private:
	// Private methods for the Kalman filter and other internal workings
	void initializeKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox);
	cv::Rect_<float> predict(cv::KalmanFilter &kf);
	cv::Rect_<float> updateKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox);

	// Data members for tracking
	std::vector<Object> trackedObjects;
	uint64_t nextTrackID;
	size_t maxUnseenFrames;
	int minHitFrames;
	float iouThreshold = 0.3f;
	float instantTrackAreaRatio = 0.0f;
	float screenArea = 0.0f;
};

#endif
// Path: src/sort-cpp/Sort.cpp
