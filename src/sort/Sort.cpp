#include "Sort.h"

#include "sort/munkres-cpp/matrix_base.h"
#include "sort/munkres-cpp/adapters/matrix_std_2d_vector.h"
#include "sort/munkres-cpp/munkres.h"

#include <cmath>
#include <limits>
#include <algorithm>

#include "plugin-support.h"

#include <obs.h>

#define INF std::numeric_limits<float>::infinity()

// Constructor
Sort::Sort(size_t maxUnseenFrames) : nextTrackID(0), maxUnseenFrames(maxUnseenFrames), minHitFrames(1), ghostRecoveryMultiplier(2.0f)
{
}

// Destructor
Sort::~Sort() {}

// Initialize the Kalman filter for a new object
void Sort::initializeKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox)
{
	// Linear motion model with dimension: [x, y, width, height, dx, dy, dwidth, dheight]
	kf.init(8, 4, 0);

	// State vector: [x, y, width, height, dx, dy, dwidth, dheight]
	kf.statePre.at<float>(0) = bbox.x;
	kf.statePre.at<float>(1) = bbox.y;
	kf.statePre.at<float>(2) = bbox.width;
	kf.statePre.at<float>(3) = bbox.height;

	// Transition matrix
	kf.transitionMatrix = (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
			       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
			       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			       0, 0, 1);

	// Measurement matrix
	kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
	for (int i = 0; i < 4; ++i) {
		kf.measurementMatrix.at<float>(i, i) = 1;
	}

	// Process noise covariance matrix
	const float q = 1e-1f;
	kf.processNoiseCov = (cv::Mat_<float>(8, 8) << 0.25, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.25, 0, 0,
			      0, 0.5, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.25, 0, 0, 0,
			      0.5, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0,
			      0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 1) *
			     q;

	// Measurement noise covariance matrix (Dynamically adjust based on size)
	float noiseMultiplier = 1.0f;
	if (this->screenArea > 0.0f) {
		float area_ratio = bbox.area() / this->screenArea;
		if (this->kalmanAreaThreshold <= 0.0f) {
			noiseMultiplier = 1.0f;
		} else {
			float multiplier = 1.0f / this->kalmanAreaThreshold;
			noiseMultiplier = std::clamp(area_ratio * multiplier, this->kalmanMinNoise, 1.0f);
		}
	}
	
	kf.measurementNoiseCov =
		(cv::Mat_<float>(4, 4) << 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10) * noiseMultiplier;

	// Error covariance matrix
	cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e3));

	// Correct the state vector with the initial measurement
	cv::Mat measurement(4, 1, CV_32F);
	measurement.at<float>(0) = bbox.x;
	measurement.at<float>(1) = bbox.y;
	measurement.at<float>(2) = bbox.width;
	measurement.at<float>(3) = bbox.height;
	kf.correct(measurement);
}

// Predict the next state of the object
cv::Rect_<float> Sort::predict(cv::KalmanFilter &kf)
{
	cv::Mat prediction = kf.predict();
	return cv::Rect_<float>(prediction.at<float>(0), prediction.at<float>(1),
				prediction.at<float>(2), prediction.at<float>(3));
}

// Update the Kalman filter with a new measurement
cv::Rect_<float> Sort::updateKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox)
{
	float noiseMultiplier = 1.0f;
	if (this->screenArea > 0.0f) {
		float area_ratio = bbox.area() / this->screenArea;
		if (this->kalmanAreaThreshold <= 0.0f) {
			noiseMultiplier = 1.0f;
		} else {
			float multiplier = 1.0f / this->kalmanAreaThreshold;
			noiseMultiplier = std::clamp(area_ratio * multiplier, this->kalmanMinNoise, 1.0f);
		}
	}
	kf.measurementNoiseCov =
		(cv::Mat_<float>(4, 4) << 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10) * noiseMultiplier;

	cv::Mat measurement(4, 1, CV_32F);
	measurement.at<float>(0) = bbox.x;
	measurement.at<float>(1) = bbox.y;
	measurement.at<float>(2) = bbox.width;
	measurement.at<float>(3) = bbox.height;
	const auto corrected = kf.correct(measurement);
	return cv::Rect_<float>(corrected.at<float>(0), corrected.at<float>(1),
				corrected.at<float>(2), corrected.at<float>(3));
}

// Compute the Intersection over Union (IoU) between two rectangles
float computeIoU(const cv::Rect_<float> &rect1, const cv::Rect_<float> &rect2)
{
	float intersectionArea = (rect1 & rect2).area();
	float unionArea = rect1.area() + rect2.area() - intersectionArea;
	return intersectionArea / unionArea;
}

// Update the tracking with detected objects
std::vector<Object> Sort::update(uint64_t frameId, const std::vector<Object> &detections)
{
	if (detections.empty()) {
		std::vector<Object> newTrackedObjects;

		// No detections, predict the next state of the existing tracks and update unseen frames
		for (size_t i = 0; i < trackedObjects.size(); ++i) {
			if (trackedObjects[i].lastVisibleRects.size() > 1) {
				const auto &rects = trackedObjects[i].lastVisibleRects;
				int n = (int)rects.size();
				float dx = (rects.back().x - rects.front().x) / (n - 1);
				float dy = (rects.back().y - rects.front().y) / (n - 1);
				float dw = (rects.back().width - rects.front().width) / (n - 1);
				float dh = (rects.back().height - rects.front().height) / (n - 1);
				float max_dx = trackedObjects[i].rect.width * 0.5f;
				float max_dy = trackedObjects[i].rect.height * 0.5f;
				dx = std::max(-max_dx, std::min(max_dx, dx));
				dy = std::max(-max_dy, std::min(max_dy, dy));
				dw = std::max(-max_dx, std::min(max_dx, dw));
				dh = std::max(-max_dy, std::min(max_dy, dh));

				trackedObjects[i].rect.x += dx;
				trackedObjects[i].rect.y += dy;
				trackedObjects[i].rect.width += dw;
				trackedObjects[i].rect.height += dh;
				predict(trackedObjects[i].kf); // update internal KF state
			} else {
				predict(trackedObjects[i].kf); // update internal KF state but keep rect stationary
			}
			trackedObjects[i].unseenFrames++;
			trackedObjects[i].trackingState = "Unseen (" + std::to_string(trackedObjects[i].unseenFrames) + ")";

			bool can_predict = trackedObjects[i].lastVisibleRects.size() > 1;
			// Remove lost tracks
			bool is_stable_and_recent = trackedObjects[i].unseenFrames < this->maxUnseenFrames &&
			                            trackedObjects[i].hitFrames >= this->minHitFrames;
			bool is_new_and_recent = trackedObjects[i].unseenFrames < this->ghostRecoveryMaxUnseen &&
			                         trackedObjects[i].hitFrames < this->minHitFrames;
			if ((is_stable_and_recent || is_new_and_recent) && can_predict) {
				newTrackedObjects.push_back(trackedObjects[i]);
			}
		}
		trackedObjects = newTrackedObjects;
		
		if (stateHistory) {
			std::map<int, TrackedObjectState> current_frame_states;
			for (auto &track : trackedObjects) {
				TrackedObjectState state;
				state.id = (int)track.id;
				state.rect = track.rect;
				state.is_extrapolated = (track.unseenFrames > 0);
				state.hit_streak = track.hitFrames;
				state.is_confirmed = (track.hitFrames >= this->minHitFrames);
				current_frame_states[(int)track.id] = state;
			}
			stateHistory->push_frame_state(frameId, current_frame_states);
		}

		return trackedObjects;
	}

	if (trackedObjects.empty()) {
		// No existing tracks, create new tracks for all detections
		for (const auto &detection : detections) {
			cv::KalmanFilter kf;
			initializeKalmanFilter(kf, detection.rect);
			trackedObjects.push_back(detection);
			trackedObjects.back().kf = kf; // store the Kalman filter in the object
			trackedObjects.back().id = nextTrackID++;
			trackedObjects.back().unseenFrames = 0;
			trackedObjects.back().hitFrames = 1;
			trackedObjects.back().trackingState = "New (1)";
			trackedObjects.back().lastVisibleRects.push_back(detection.rect);
		}
		
		if (stateHistory) {
			std::map<int, TrackedObjectState> current_frame_states;
			for (auto &track : trackedObjects) {
				TrackedObjectState state;
				state.id = (int)track.id;
				state.rect = track.rect;
				state.is_extrapolated = (track.unseenFrames > 0);
				state.hit_streak = track.hitFrames;
				state.is_confirmed = (track.hitFrames >= this->minHitFrames);
				current_frame_states[(int)track.id] = state;
			}
			stateHistory->push_frame_state(frameId, current_frame_states);
		}
		
		return trackedObjects;
	}

	// Predict new locations of existing tracked objects
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		if (trackedObjects[i].lastVisibleRects.size() > 1) {
			const auto &rects = trackedObjects[i].lastVisibleRects;
			int n = (int)rects.size();
			float dx = (rects.back().x - rects.front().x) / (n - 1);
			float dy = (rects.back().y - rects.front().y) / (n - 1);
			float dw = (rects.back().width - rects.front().width) / (n - 1);
			float dh = (rects.back().height - rects.front().height) / (n - 1);
			float max_dx = trackedObjects[i].rect.width * 0.5f;
			float max_dy = trackedObjects[i].rect.height * 0.5f;
			dx = std::max(-max_dx, std::min(max_dx, dx));
			dy = std::max(-max_dy, std::min(max_dy, dy));
			dw = std::max(-max_dx, std::min(max_dx, dw));
			dh = std::max(-max_dy, std::min(max_dy, dh));

			trackedObjects[i].rect.x += dx;
			trackedObjects[i].rect.y += dy;
			trackedObjects[i].rect.width += dw;
			trackedObjects[i].rect.height += dh;
			predict(trackedObjects[i].kf); // update internal KF state
		} else {
			predict(trackedObjects[i].kf); // update internal KF state but keep rect stationary
		}
	}

	// Build the cost matrix for the Hungarian algorithm
	size_t numDetections = detections.size();
	std::vector<std::vector<float>> costMatrix(trackedObjects.size(),
						   std::vector<float>(numDetections, 0));

	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		for (size_t j = 0; j < numDetections; ++j) {
			const float iou = computeIoU(trackedObjects[i].rect, detections[j].rect);
			costMatrix[i][j] = iou >= this->iouThreshold ? 1.0f - iou : 10.0f;
		}
	}

	// Solve the assignment problem using the Hungarian algorithm
	munkres_cpp::matrix_std_2d_vector costMatrixAdapter(costMatrix);
	munkres_cpp::Munkres<float, munkres_cpp::matrix_std_2d_vector> solver(costMatrixAdapter);

	// Update Kalman filters with associated detections
	std::vector<bool> detectionUsed(numDetections, false);
	std::vector<bool> trackedObjectUsed(trackedObjects.size(), false);
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		for (size_t j = 0; j < numDetections; ++j) {
			if (costMatrix[i][j] == 0) {
				const float iou =
					computeIoU(trackedObjects[i].rect, detections[j].rect);
				if (iou < this->iouThreshold) {
					// prevent matching detections with low overlap
					continue;
				}
				if (trackedObjects[i].unseenFrames > 0) {
					trackedObjects[i].lastVisibleRects.clear();
				}
				// update the tracked object with the new detection
				trackedObjects[i].rect = updateKalmanFilter(trackedObjects[i].kf,
									    detections[j].rect);
				trackedObjects[i].unseenFrames = 0;
				trackedObjects[i].hitFrames++;
				if (this->instantTrackAreaRatio > 0.0f && this->screenArea > 0.0f) {
					if (detections[j].rect.area() >= this->screenArea * (this->instantTrackAreaRatio / 100.0f)) {
						trackedObjects[i].hitFrames = std::max(trackedObjects[i].hitFrames, this->minHitFrames);
					}
				}
				if (trackedObjects[i].hitFrames > 5) {
					trackedObjects[i].trackingState = "Stable";
				} else {
					trackedObjects[i].trackingState = "New (" + std::to_string(trackedObjects[i].hitFrames) + ")";
				}
				trackedObjects[i].lastVisibleRects.push_back(trackedObjects[i].rect);
				if (trackedObjects[i].lastVisibleRects.size() > 3) {
					trackedObjects[i].lastVisibleRects.pop_front();
				}
				trackedObjects[i].label = detections[j].label;
				trackedObjects[i].prob = detections[j].prob;
				for (int n = 0; n < 5; ++n) {
					trackedObjects[i].landmarks[n] = detections[j].landmarks[n];
				}
				// mark the detection and the tracked object as used
				detectionUsed[j] = true;
				trackedObjectUsed[i] = true;
				break;
			}
		}
	}

	// Global secondary spatial matching (Ghost Recovery) for all tracks that went missing
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		if (trackedObjectUsed[i]) continue;
		
		float best_dist = std::numeric_limits<float>::max();
		int best_det_idx = -1;
		
		float t_cx = trackedObjects[i].rect.x + trackedObjects[i].rect.width / 2.0f;
		float t_cy = trackedObjects[i].rect.y + trackedObjects[i].rect.height / 2.0f;
		float size_ref = std::max(trackedObjects[i].rect.width, trackedObjects[i].rect.height);
		float max_dist = size_ref * this->ghostRecoveryMultiplier;
		
		for (size_t j = 0; j < numDetections; ++j) {
			if (detectionUsed[j]) continue;
			if (trackedObjects[i].label != detections[j].label) continue;
			
			float d_cx = detections[j].rect.x + detections[j].rect.width / 2.0f;
			float d_cy = detections[j].rect.y + detections[j].rect.height / 2.0f;
			
			float dist = std::sqrt((d_cx - t_cx) * (d_cx - t_cx) + (d_cy - t_cy) * (d_cy - t_cy));
			if (dist < max_dist && dist < best_dist) {
				best_dist = dist;
				best_det_idx = (int)j;
			}
		}
		
		if (best_det_idx != -1) {
			int j = best_det_idx;
			// 칼만 필터의 느린 추적을 우회하고 새 대상의 위치로 즉시 이동 (Instant Teleport)
			trackedObjects[i].rect = detections[j].rect;
			
			// 칼만 필터를 초기화하지 않고(기존 관성과 공분산 유지), 위치 좌표만 강제로 덮어씌움
			trackedObjects[i].kf.statePost.at<float>(0) = detections[j].rect.x;
			trackedObjects[i].kf.statePost.at<float>(1) = detections[j].rect.y;
			trackedObjects[i].kf.statePost.at<float>(2) = detections[j].rect.width;
			trackedObjects[i].kf.statePost.at<float>(3) = detections[j].rect.height;
			trackedObjects[i].kf.statePost.copyTo(trackedObjects[i].kf.statePre);
			
			trackedObjects[i].unseenFrames = 0;
			trackedObjects[i].hitFrames++;
			trackedObjects[i].trackingState = "Recovered";
			
			trackedObjects[i].lastVisibleRects.push_back(trackedObjects[i].rect);
			if (trackedObjects[i].lastVisibleRects.size() > 3) {
				trackedObjects[i].lastVisibleRects.pop_front();
			}
			trackedObjects[i].label = detections[j].label;
			trackedObjects[i].prob = detections[j].prob;
			for (int n = 0; n < 5; ++n) {
				trackedObjects[i].landmarks[n] = detections[j].landmarks[n];
			}
			
			detectionUsed[j] = true;
			trackedObjectUsed[i] = true;
		}
	}

	// Create new tracks for unmatched detections
	for (size_t j = 0; j < numDetections; ++j) {
		if (!detectionUsed[j]) {
			cv::KalmanFilter kf;
			initializeKalmanFilter(kf, detections[j].rect);
			trackedObjects.push_back(detections[j]);
			trackedObjects.back().kf = kf; // store the Kalman filter in the object
			trackedObjects.back().id = nextTrackID++;
			trackedObjects.back().unseenFrames = 0;
			trackedObjects.back().hitFrames = 1;
			if (this->instantTrackAreaRatio > 0.0f && this->screenArea > 0.0f) {
				if (detections[j].rect.area() >= this->screenArea * (this->instantTrackAreaRatio / 100.0f)) {
					trackedObjects.back().hitFrames = this->minHitFrames;
				}
			}
			if (trackedObjects.back().hitFrames > 5) {
				trackedObjects.back().trackingState = "Stable";
			} else {
				trackedObjects.back().trackingState = "New (" + std::to_string(trackedObjects.back().hitFrames) + ")";
			}
			trackedObjects.back().lastVisibleRects.push_back(detections[j].rect);
			// resize trackedObjectUsed to match the new size of trackedObjects
			trackedObjectUsed.resize(trackedObjects.size(), true);
		}
	}

	// Remove lost tracks
	std::vector<Object> newTrackedObjects;
	std::vector<int> newTrackIDs;
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		bool can_predict = trackedObjects[i].lastVisibleRects.size() > 1;
		bool is_stable_and_recent = trackedObjects[i].unseenFrames < this->maxUnseenFrames &&
		                            trackedObjects[i].hitFrames >= this->minHitFrames;
		bool is_new_and_recent = trackedObjects[i].unseenFrames < this->ghostRecoveryMaxUnseen &&
		                         trackedObjects[i].hitFrames < this->minHitFrames;
		
		if (trackedObjectUsed[i] ||
		    ((is_stable_and_recent || is_new_and_recent) && can_predict)) {
			newTrackedObjects.push_back(trackedObjects[i]);
			if (!trackedObjectUsed[i]) {
				newTrackedObjects.back().unseenFrames++;
				newTrackedObjects.back().trackingState = "Unseen (" + std::to_string(newTrackedObjects.back().unseenFrames) + ")";
			}
		}
	}
	trackedObjects = newTrackedObjects;

	if (stateHistory) {
		std::map<int, TrackedObjectState> current_frame_states;
		for (auto &track : trackedObjects) {
			TrackedObjectState state;
			state.id = (int)track.id;
			state.rect = track.rect;
			state.is_extrapolated = (track.unseenFrames > 0);
			state.hit_streak = track.hitFrames;
			
			// Detect when an object just reached minHitFrames
			if (track.hitFrames == this->minHitFrames && this->minHitFrames > 1) {
				uint64_t start_f = (frameId > (uint64_t)(track.hitFrames - 1)) ? (frameId - track.hitFrames + 1) : 0;
				stateHistory->retroactively_confirm(start_f, frameId, (int)track.id);
			}
			
			state.is_confirmed = (track.hitFrames >= this->minHitFrames);
			current_frame_states[(int)track.id] = state;
		}
		stateHistory->push_frame_state(frameId, current_frame_states);
	}

	return trackedObjects;
}

// Get the current tracked objects and their tracking id
std::vector<Object> Sort::getTrackedObjects() const
{
	return trackedObjects;
}
