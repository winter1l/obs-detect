#include "FaceStateStore.h"

#include <atomic>
#include <algorithm>
#include <unordered_set>

namespace {
constexpr size_t maxRememberedSelfEpochs = 4;

void rememberSelfEpoch(FaceTrackDecision &decision, uint64_t epoch)
{
	if (epoch == 0 ||
	    std::find(decision.confirmedSelfEpochs.begin(), decision.confirmedSelfEpochs.end(),
		      epoch) != decision.confirmedSelfEpochs.end())
		return;
	decision.confirmedSelfEpochs.push_back(epoch);
	if (decision.confirmedSelfEpochs.size() > maxRememberedSelfEpochs)
		decision.confirmedSelfEpochs.erase(decision.confirmedSelfEpochs.begin());
}
} // namespace

FaceStateStore::FaceStateStore()
{
	publishLocked();
}

void FaceStateStore::publishLocked()
{
	auto next = std::make_shared<const FaceDecisionSnapshot>(decisions_);
	std::atomic_store_explicit(&snapshot_, std::move(next), std::memory_order_release);
}

void FaceStateStore::clear()
{
	std::lock_guard<std::mutex> lock(mutex_);
	decisions_.clear();
	publishLocked();
}

void FaceStateStore::setPreserveOnRecovery(bool preserve)
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (preserveOnRecovery_ == preserve)
		return;

	preserveOnRecovery_ = preserve;
	if (!preserve) {
		for (auto &[id, decision] : decisions_) {
			(void)id;
			if (!decision.carriedAcrossRecovery)
				continue;
			decision.status = FaceStatus::UNKNOWN;
			decision.similarity = 0.0f;
			decision.hasSimilarity = false;
			decision.carriedAcrossRecovery = false;
		}
	}
	publishLocked();
}

bool FaceStateStore::preserveOnRecovery() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return preserveOnRecovery_;
}

void FaceStateStore::synchronizeTracks(const std::vector<FaceTrackIdentity> &tracks)
{
	std::lock_guard<std::mutex> lock(mutex_);
	std::unordered_set<uint64_t> currentIds;
	currentIds.reserve(tracks.size());

	for (const FaceTrackIdentity &track : tracks) {
		currentIds.insert(track.id);
		auto it = decisions_.find(track.id);
		if (it == decisions_.end()) {
			FaceTrackDecision decision;
			decision.identityEpoch = track.identityEpoch;
			decision.area = track.area;
			decisions_.emplace(track.id, decision);
			continue;
		}

		FaceTrackDecision &decision = it->second;
		decision.area = track.area;
		if (decision.identityEpoch == track.identityEpoch)
			continue;

		const bool wasConfirmedSelf = decision.status == FaceStatus::IS_ME;
		if (wasConfirmedSelf)
			rememberSelfEpoch(decision, decision.identityEpoch);
		auto confirmedEpochs = std::move(decision.confirmedSelfEpochs);
		decision = FaceTrackDecision{};
		decision.identityEpoch = track.identityEpoch;
		decision.area = track.area;
		decision.confirmedSelfEpochs = std::move(confirmedEpochs);
		decision.priorityRecheck = wasConfirmedSelf;
		if (preserveOnRecovery_ && wasConfirmedSelf) {
			decision.status = FaceStatus::IS_ME;
			decision.carriedAcrossRecovery = true;
		}
	}

	for (auto it = decisions_.begin(); it != decisions_.end();) {
		if (currentIds.count(it->first) == 0) {
			it = decisions_.erase(it);
		} else {
			++it;
		}
	}

	publishLocked();
}

FaceTrackDecision FaceStateStore::getDecision(uint64_t id, uint64_t identityEpoch) const
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it != decisions_.end() && it->second.identityEpoch == identityEpoch)
		return it->second;

	FaceTrackDecision decision;
	decision.identityEpoch = identityEpoch;
	return decision;
}

bool FaceStateStore::beginCheck(uint64_t id, uint64_t identityEpoch, uint64_t nowMs,
				bool *priority, uint64_t *checkToken)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it == decisions_.end() || it->second.identityEpoch != identityEpoch)
		return false;
	FaceTrackDecision &decision = it->second;
	if (decision.checkInFlight || isSettledSelf(decision) || nowMs < decision.nextCheckAtMs)
		return false;

	decision.checkInFlight = true;
	decision.activeCheckToken = nextCheckToken_++;
	if (nextCheckToken_ == 0)
		nextCheckToken_ = 1;
	if (priority)
		*priority = decision.priorityRecheck || decision.revalidate;
	if (checkToken)
		*checkToken = decision.activeCheckToken;
	publishLocked();
	return true;
}

bool FaceStateStore::completeCheck(uint64_t id, uint64_t identityEpoch,
				   uint64_t checkToken,
				   FaceCheckOutcome outcome,
				   std::optional<float> similarity, int maxExemptPersons,
				   uint64_t nowMs, const FaceRetryPolicy &retryPolicy)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it == decisions_.end() || it->second.identityEpoch != identityEpoch)
		return false;
	if (!it->second.checkInFlight || it->second.activeCheckToken != checkToken)
		return false;

	FaceTrackDecision &decision = it->second;
	decision.checkInFlight = false;
	decision.activeCheckToken = 0;
	decision.lastOutcome = outcome;
	if (similarity.has_value()) {
		decision.similarity = *similarity;
		decision.hasSimilarity = true;
	}

	// An exemption is kept through inconclusive checks only when it was carried across
	// a recovery or is being revalidated after a conflict.
	const bool keepExemptWhenInconclusive =
		decision.status == FaceStatus::IS_ME &&
		(decision.carriedAcrossRecovery || decision.revalidate);
	decision.revalidate = false;

	bool exemptConflict = false;
	if (outcome == FaceCheckOutcome::MATCH && decision.status != FaceStatus::IS_ME &&
	    maxExemptPersons > 0) {
		if (!resolveExemptConflictLocked(id, decision, decision.similarity,
						 maxExemptPersons)) {
			outcome = FaceCheckOutcome::EXEMPT_LIMIT;
			decision.lastOutcome = outcome;
			exemptConflict = true;
		}
	}

	switch (outcome) {
	case FaceCheckOutcome::MATCH:
		decision.status = FaceStatus::IS_ME;
		decision.carriedAcrossRecovery = false;
		decision.priorityRecheck = false;
		decision.nextCheckAtMs = 0;
		rememberSelfEpoch(decision, identityEpoch);
		break;
	case FaceCheckOutcome::MISMATCH:
	case FaceCheckOutcome::EXEMPT_LIMIT:
		decision.status = FaceStatus::NOT_ME;
		decision.carriedAcrossRecovery = false;
		decision.priorityRecheck = false;
		// A conflicting match retries soon so it can take the slot if the current
		// holder fails its revalidation.
		decision.nextCheckAtMs = nowMs + (exemptConflict ? retryPolicy.noFaceRetryMs
								 : retryPolicy.mismatchRetryMs);
		break;
	case FaceCheckOutcome::NO_FACE:
		if (!keepExemptWhenInconclusive) {
			decision.status = FaceStatus::UNKNOWN;
			decision.hasSimilarity = false;
		}
		decision.nextCheckAtMs = nowMs + retryPolicy.noFaceRetryMs;
		break;
	case FaceCheckOutcome::INFERENCE_ERROR:
		if (!keepExemptWhenInconclusive)
			decision.status = FaceStatus::UNKNOWN;
		decision.nextCheckAtMs = nowMs + retryPolicy.errorRetryMs;
		break;
	case FaceCheckOutcome::NONE:
		decision.status = FaceStatus::UNKNOWN;
		decision.nextCheckAtMs = nowMs + retryPolicy.errorRetryMs;
		break;
	}
	publishLocked();
	return true;
}

// Called when `candidate` matched while the exempt limit may be full. Returns true if the
// candidate may become IS_ME. A clearly better match, or a much larger face (in a selfie
// stream the streamer is the closest face), takes the slot from the weakest holder, which
// is reset for an immediate recheck. Otherwise the holders are flagged for revalidation so
// a holder that was assigned to the wrong person can lose the slot.
bool FaceStateStore::resolveExemptConflictLocked(uint64_t id, FaceTrackDecision &candidate,
						 float similarity, int maxExemptPersons)
{
	constexpr float similarityMargin = 0.05f;
	constexpr float largerFaceRatio = 1.5f;

	auto holderSimilarity = [](const FaceTrackDecision &holder) {
		return holder.hasSimilarity && !holder.carriedAcrossRecovery ? holder.similarity
									      : 0.0f;
	};

	int exemptCount = 0;
	FaceTrackDecision *weakest = nullptr;
	for (auto &[otherId, other] : decisions_) {
		if (otherId == id || other.status != FaceStatus::IS_ME)
			continue;
		exemptCount++;
		if (!weakest || holderSimilarity(other) < holderSimilarity(*weakest) ||
		    (holderSimilarity(other) == holderSimilarity(*weakest) &&
		     other.area < weakest->area))
			weakest = &other;
	}
	if (exemptCount < maxExemptPersons || !weakest)
		return true;

	const bool clearlyBetterMatch = similarity >= holderSimilarity(*weakest) + similarityMargin;
	const bool clearlyLarger =
		weakest->area > 0.0f && candidate.area >= weakest->area * largerFaceRatio;
	if (clearlyBetterMatch || clearlyLarger) {
		weakest->status = FaceStatus::UNKNOWN;
		weakest->carriedAcrossRecovery = false;
		weakest->revalidate = false;
		weakest->priorityRecheck = true;
		weakest->nextCheckAtMs = 0;
		return true;
	}

	for (auto &[otherId, other] : decisions_) {
		if (otherId != id && other.status == FaceStatus::IS_ME)
			other.revalidate = true;
	}
	return false;
}

bool FaceStateStore::cancelCheck(uint64_t id, uint64_t identityEpoch,
				 uint64_t checkToken, uint64_t nowMs, uint64_t retryDelayMs)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it == decisions_.end() || it->second.identityEpoch != identityEpoch ||
	    !it->second.checkInFlight || it->second.activeCheckToken != checkToken)
		return false;
	it->second.checkInFlight = false;
	it->second.activeCheckToken = 0;
	it->second.nextCheckAtMs = nowMs + retryDelayMs;
	publishLocked();
	return true;
}

void FaceStateStore::resetToUnknown(uint64_t id, uint64_t identityEpoch)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it == decisions_.end() || it->second.identityEpoch != identityEpoch)
		return;
	it->second.status = FaceStatus::UNKNOWN;
	it->second.checkInFlight = false;
	it->second.activeCheckToken = 0;
	it->second.carriedAcrossRecovery = false;
	it->second.hasSimilarity = false;
	it->second.nextCheckAtMs = 0;
	it->second.lastOutcome = FaceCheckOutcome::INFERENCE_ERROR;
	publishLocked();
}

std::shared_ptr<const FaceDecisionSnapshot> FaceStateStore::snapshot() const
{
	return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

bool FaceStateStore::isExempt(const FaceDecisionSnapshot &snapshot, uint64_t id,
			      uint64_t identityEpoch, bool allowCarriedEpochMismatch)
{
	auto it = snapshot.find(id);
	if (it == snapshot.end())
		return false;
	const FaceTrackDecision &decision = it->second;
	if (decision.identityEpoch == identityEpoch)
		return decision.status == FaceStatus::IS_ME;
	if (std::find(decision.confirmedSelfEpochs.begin(),
		      decision.confirmedSelfEpochs.end(), identityEpoch) !=
	    decision.confirmedSelfEpochs.end())
		return true;
	return allowCarriedEpochMismatch && decision.status == FaceStatus::IS_ME &&
	       decision.carriedAcrossRecovery;
}
