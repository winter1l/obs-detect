#include "StateHistory.h"

void StateHistoryManager::push_frame_state(uint64_t frame_id, const std::map<int, TrackedObjectState> &states)
{
	std::lock_guard<std::mutex> lock(mutex_);
	FrameStateHistory history;
	history.frame_id = frame_id;
	history.objects = states;
	
	buffer.push_back(history);
	if (buffer.size() > max_buffer_size) {
		buffer.pop_front();
	}
}

bool StateHistoryManager::get_object_state(uint64_t frame_id, int object_id, TrackedObjectState &out_state) const
{
	std::lock_guard<std::mutex> lock(mutex_);
	for (const auto &history : buffer) {
		if (history.frame_id == frame_id) {
			auto it = history.objects.find(object_id);
			if (it != history.objects.end()) {
				out_state = it->second;
				return true;
			}
		}
	}
	return false;
}

void StateHistoryManager::retroactively_confirm(uint64_t start_frame, uint64_t end_frame, int object_id)
{
	std::lock_guard<std::mutex> lock(mutex_);
	for (auto &history : buffer) {
		if (history.frame_id >= start_frame && history.frame_id <= end_frame) {
			auto it = history.objects.find(object_id);
			if (it != history.objects.end()) {
				it->second.is_confirmed = true;
			}
		}
	}
}

std::set<int> StateHistoryManager::get_all_active_ids(uint64_t start_frame, uint64_t end_frame) const
{
	std::lock_guard<std::mutex> lock(mutex_);
	std::set<int> active_ids;
	for (const auto &history : buffer) {
		if (history.frame_id >= start_frame && history.frame_id <= end_frame) {
			for (const auto &pair : history.objects) {
				active_ids.insert(pair.first);
			}
		}
	}
	return active_ids;
}

void StateHistoryManager::clear()
{
	std::lock_guard<std::mutex> lock(mutex_);
	buffer.clear();
}

std::deque<FrameStateHistory> StateHistoryManager::get_buffer_copy() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return buffer;
}
