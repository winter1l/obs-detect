#include "FaceStateStore.h"

#include <atomic>
#include <unordered_set>

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
			decisions_.emplace(track.id, decision);
			continue;
		}

		FaceTrackDecision &decision = it->second;
		if (decision.identityEpoch == track.identityEpoch)
			continue;

		if (preserveOnRecovery_ && decision.status == FaceStatus::IS_ME) {
			decision.identityEpoch = track.identityEpoch;
			decision.carriedAcrossRecovery = true;
		} else {
			decision = FaceTrackDecision{};
			decision.identityEpoch = track.identityEpoch;
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

bool FaceStateStore::beginCheck(uint64_t id, uint64_t identityEpoch)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it == decisions_.end() || it->second.identityEpoch != identityEpoch)
		return false;
	if (it->second.status == FaceStatus::IS_ME || it->second.status == FaceStatus::CHECKING)
		return false;

	it->second.status = FaceStatus::CHECKING;
	it->second.carriedAcrossRecovery = false;
	publishLocked();
	return true;
}

bool FaceStateStore::commit(uint64_t id, uint64_t identityEpoch, FaceStatus status,
			    std::optional<float> similarity, int maxExemptPersons)
{
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = decisions_.find(id);
	if (it == decisions_.end() || it->second.identityEpoch != identityEpoch)
		return false;

	if (status == FaceStatus::IS_ME && it->second.status != FaceStatus::IS_ME &&
	    maxExemptPersons > 0) {
		int currentExemptCount = 0;
		for (const auto &[otherId, decision] : decisions_) {
			(void)otherId;
			if (decision.status == FaceStatus::IS_ME)
				currentExemptCount++;
		}
		if (currentExemptCount >= maxExemptPersons)
			status = FaceStatus::NOT_ME;
	}

	it->second.status = status;
	it->second.carriedAcrossRecovery = false;
	if (similarity.has_value()) {
		it->second.similarity = *similarity;
		it->second.hasSimilarity = true;
	} else if (status == FaceStatus::UNKNOWN) {
		it->second.similarity = 0.0f;
		it->second.hasSimilarity = false;
	}
	publishLocked();
	return true;
}

void FaceStateStore::resetToUnknown(uint64_t id, uint64_t identityEpoch)
{
	commit(id, identityEpoch, FaceStatus::UNKNOWN, std::nullopt, 0);
}

std::shared_ptr<const FaceDecisionSnapshot> FaceStateStore::snapshot() const
{
	return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

bool FaceStateStore::isExempt(const FaceDecisionSnapshot &snapshot, uint64_t id,
			      uint64_t identityEpoch, bool allowCarriedEpochMismatch)
{
	auto it = snapshot.find(id);
	if (it == snapshot.end() || it->second.status != FaceStatus::IS_ME)
		return false;
	return it->second.identityEpoch == identityEpoch ||
	       (allowCarriedEpochMismatch && it->second.carriedAcrossRecovery);
}
