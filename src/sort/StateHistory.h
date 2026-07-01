#ifndef STATEHISTORY_H
#define STATEHISTORY_H

#include <vector>
#include <deque>
#include <map>
#include <set>
#include <mutex>
#include <opencv2/opencv.hpp>

// 개별 객체의 상태 단위
struct TrackedObjectState {
	int id;
	cv::Rect_<float> rect;
	bool is_extrapolated;
	bool is_confirmed;
	int hit_streak;
};

// 프레임 단위의 상태 이력
struct FrameStateHistory {
	uint64_t frame_id;
	std::map<int, TrackedObjectState> objects;
};

// 상태 이력 관리 클래스 (Thread-Safe)
class StateHistoryManager {
private:
	std::deque<FrameStateHistory> buffer;
	size_t max_buffer_size = 60; // 최대 유지할 프레임 수 (충분한 룩어헤드 보장)
	mutable std::mutex mutex_;

public:
	// 상태 저장
	void push_frame_state(uint64_t frame_id, const std::map<int, TrackedObjectState> &states);

	// 특정 프레임의 특정 객체 상태 반환
	bool get_object_state(uint64_t frame_id, int object_id, TrackedObjectState &out_state) const;

	// 과거 특정 시점부터 현재까지의 tentative 상태를 confirmed로 소급 변경
	void retroactively_confirm(uint64_t start_frame, uint64_t end_frame, int object_id);

	// 특정 구간 내 및 R시점 이전 가장 가까운 과거 프레임의 모든 고유 객체 ID 수집
	std::set<int> get_all_active_ids(uint64_t start_frame, uint64_t end_frame) const;

	// 버퍼 완전 초기화 (씬 전환 등 방어)
	void clear();
	
	// 전체 버퍼 순회를 위해 락을 건 복사본 반환 (안전한 조회를 위함)
	std::deque<FrameStateHistory> get_buffer_copy() const;
};

#endif // STATEHISTORY_H
