#ifndef FACE_STATE_STORE_H
#define FACE_STATE_STORE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

enum class FaceStatus { UNKNOWN, IS_ME, NOT_ME };

enum class FaceCheckOutcome {
	NONE,
	MATCH,
	MISMATCH,
	NO_FACE,
	INFERENCE_ERROR,
	EXEMPT_LIMIT
};

struct FaceRetryPolicy {
	uint64_t noFaceRetryMs = 150;
	uint64_t errorRetryMs = 500;
	uint64_t mismatchRetryMs = 500;
};

struct FaceTrackIdentity {
	uint64_t id = 0;
	uint64_t identityEpoch = 0;
	float area = 0.0f;
};

struct FaceTrackDecision {
	FaceStatus status = FaceStatus::UNKNOWN;
	float similarity = 0.0f;
	bool hasSimilarity = false;
	uint64_t identityEpoch = 0;
	float area = 0.0f;
	bool carriedAcrossRecovery = false;
	// A verified IS_ME is rechecked only when another track also matched while the
	// exempt limit was full. Inconclusive results keep the exemption.
	bool revalidate = false;
	bool checkInFlight = false;
	bool priorityRecheck = false;
	uint64_t activeCheckToken = 0;
	uint64_t nextCheckAtMs = 0;
	FaceCheckOutcome lastOutcome = FaceCheckOutcome::NONE;
	std::vector<uint64_t> confirmedSelfEpochs;
};

using FaceDecisionSnapshot = std::unordered_map<uint64_t, FaceTrackDecision>;

// A verified self decision that needs no further face checks.
inline bool isSettledSelf(const FaceTrackDecision &decision)
{
	return decision.status == FaceStatus::IS_ME && !decision.carriedAcrossRecovery &&
	       !decision.revalidate;
}

class FaceStateStore {
public:
	FaceStateStore();

	void clear();
	void setPreserveOnRecovery(bool preserve);
	bool preserveOnRecovery() const;

	void synchronizeTracks(const std::vector<FaceTrackIdentity> &tracks);
	FaceTrackDecision getDecision(uint64_t id, uint64_t identityEpoch) const;
	bool beginCheck(uint64_t id, uint64_t identityEpoch, uint64_t nowMs,
			bool *priority = nullptr, uint64_t *checkToken = nullptr);
	bool completeCheck(uint64_t id, uint64_t identityEpoch, uint64_t checkToken,
			   FaceCheckOutcome outcome,
			   std::optional<float> similarity, int maxExemptPersons,
			   uint64_t nowMs, const FaceRetryPolicy &retryPolicy);
	bool cancelCheck(uint64_t id, uint64_t identityEpoch, uint64_t checkToken,
			 uint64_t nowMs,
			 uint64_t retryDelayMs);
	void resetToUnknown(uint64_t id, uint64_t identityEpoch);

	std::shared_ptr<const FaceDecisionSnapshot> snapshot() const;
	static bool isExempt(const FaceDecisionSnapshot &snapshot, uint64_t id,
			     uint64_t identityEpoch, bool allowCarriedEpochMismatch = false);

private:
	void publishLocked();
	bool resolveExemptConflictLocked(uint64_t id, FaceTrackDecision &candidate,
					 float similarity, int maxExemptPersons);

	mutable std::mutex mutex_;
	FaceDecisionSnapshot decisions_;
	std::shared_ptr<const FaceDecisionSnapshot> snapshot_;
	bool preserveOnRecovery_ = false;
	uint64_t nextCheckToken_ = 1;
};

#endif // FACE_STATE_STORE_H
