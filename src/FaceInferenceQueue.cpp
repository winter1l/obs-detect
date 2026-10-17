#include "FaceInferenceQueue.h"

#include <algorithm>
#include <utility>

FaceInferenceQueue::FaceInferenceQueue(size_t maxJobs) : maxJobs_(std::max<size_t>(1, maxJobs))
{
}

bool FaceInferenceQueue::tryPush(FaceInferenceJob job)
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (jobs_.size() >= maxJobs_)
		return false;

	if (job.priority) {
		auto insertionPoint = std::find_if(jobs_.begin(), jobs_.end(),
					   [](const FaceInferenceJob &queued) {
						   return !queued.priority;
					   });
		jobs_.insert(insertionPoint, std::move(job));
	} else {
		jobs_.push_back(std::move(job));
	}
	cv_.notify_one();
	return true;
}

bool FaceInferenceQueue::tryPop(FaceInferenceJob &job)
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (jobs_.empty())
		return false;
	job = std::move(jobs_.front());
	jobs_.pop_front();
	return true;
}

bool FaceInferenceQueue::waitPop(FaceInferenceJob &job,
				 const std::atomic<bool> &stopRequested)
{
	std::unique_lock<std::mutex> lock(mutex_);
	cv_.wait(lock, [&] { return stopRequested.load() || !jobs_.empty(); });
	if (jobs_.empty())
		return false;
	job = std::move(jobs_.front());
	jobs_.pop_front();
	return true;
}

void FaceInferenceQueue::clear()
{
	std::lock_guard<std::mutex> lock(mutex_);
	jobs_.clear();
}

void FaceInferenceQueue::notifyAll()
{
	cv_.notify_all();
}

size_t FaceInferenceQueue::size() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return jobs_.size();
}
