#ifndef FACE_STATE_STORE_H
#define FACE_STATE_STORE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

enum class FaceStatus { UNKNOWN, CHECKING, IS_ME, NOT_ME };

struct FaceTrackIdentity {
	uint64_t id = 0;
	uint64_t identityEpoch = 0;
};

struct FaceTrackDecision {
	FaceStatus status = FaceStatus::UNKNOWN;
	float similarity = 0.0f;
	bool hasSimilarity = false;
	uint64_t identityEpoch = 0;
	bool carriedAcrossRecovery = false;
};

using FaceDecisionSnapshot = std::unordered_map<uint64_t, FaceTrackDecision>;

class FaceStateStore {
public:
	FaceStateStore();

	void clear();
	void setPreserveOnRecovery(bool preserve);
	bool preserveOnRecovery() const;

	void synchronizeTracks(const std::vector<FaceTrackIdentity> &tracks);
	FaceTrackDecision getDecision(uint64_t id, uint64_t identityEpoch) const;
	bool beginCheck(uint64_t id, uint64_t identityEpoch);
	bool commit(uint64_t id, uint64_t identityEpoch, FaceStatus status,
		    std::optional<float> similarity, int maxExemptPersons);
	void resetToUnknown(uint64_t id, uint64_t identityEpoch);

	std::shared_ptr<const FaceDecisionSnapshot> snapshot() const;
	static bool isExempt(const FaceDecisionSnapshot &snapshot, uint64_t id,
			     uint64_t identityEpoch, bool allowCarriedEpochMismatch = false);

private:
	void publishLocked();

	mutable std::mutex mutex_;
	FaceDecisionSnapshot decisions_;
	std::shared_ptr<const FaceDecisionSnapshot> snapshot_;
	bool preserveOnRecovery_ = false;
};

#endif // FACE_STATE_STORE_H
