#include "FaceInferenceQueue.h"
#include "FaceReferenceSelection.h"
#include "FaceStateStore.h"
#include "FrameUtils.h"
#include "InferenceSafety.h"
#include "MaskGeometry.h"
#include "sort/StateHistory.h"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void require(bool condition, const char *message)
{
	if (!condition)
		throw std::runtime_error(message);
}

static void confirmSelf(FaceStateStore &store, uint64_t id, uint64_t epoch)
{
	store.synchronizeTracks({{id, epoch}});
	bool priority = false;
	uint64_t checkToken = 0;
	require(store.beginCheck(id, epoch, 1000, &priority, &checkToken),
		"face check should start");
	require(store.completeCheck(id, epoch, checkToken, FaceCheckOutcome::MATCH, 0.9f,
				    0, 1000, FaceRetryPolicy{}),
		"face result should commit");
}

static void testRecoveryDefaultsFailClosed()
{
	FaceStateStore store;
	confirmSelf(store, 7, 1);
	store.synchronizeTracks({{7, 2}});
	auto snapshot = store.snapshot();
	require(!FaceStateStore::isExempt(*snapshot, 7, 2),
		"recovered identity must be masked by default");
	require(FaceStateStore::isExempt(*snapshot, 7, 1),
		"a previously verified epoch must remain exempt for delayed rendering");
	require(!store.completeCheck(7, 1, 999, FaceCheckOutcome::MATCH, 0.95f, 0,
				     1010, FaceRetryPolicy{}),
		"stale epoch result must be discarded");
	bool priority = false;
	uint64_t checkToken = 0;
	require(store.beginCheck(7, 2, 1010, &priority, &checkToken),
		"recovered identity should be rechecked immediately");
	require(priority, "a recovered confirmed self track should be prioritized");
	require(store.completeCheck(7, 2, checkToken, FaceCheckOutcome::MATCH, 0.91f, 0,
				    1010, FaceRetryPolicy{}),
		"current epoch result should commit");
	require(FaceStateStore::isExempt(*store.snapshot(), 7, 2),
		"verified recovered identity should be exempt");
}

static void testRecoveryPreserveOption()
{
	FaceStateStore store;
	store.setPreserveOnRecovery(true);
	confirmSelf(store, 8, 1);

	store.synchronizeTracks({{8, 1}});
	require(FaceStateStore::isExempt(*store.snapshot(), 8, 1),
		"normal unseen interval should retain the decision");

	store.synchronizeTracks({{8, 2}});
	FaceTrackDecision carried = store.getDecision(8, 2);
	require(carried.status == FaceStatus::IS_ME && carried.carriedAcrossRecovery,
		"enabled option should carry a confirmed self decision");
	bool priority = false;
	uint64_t checkToken = 0;
	require(store.beginCheck(8, 2, 1001, &priority, &checkToken) && priority,
		"a carried decision should still be revalidated immediately");
	require(store.completeCheck(8, 2, checkToken, FaceCheckOutcome::NO_FACE,
				    std::nullopt, 0, 1001, FaceRetryPolicy{}),
		"an inconclusive carried check should complete");
	require(FaceStateStore::isExempt(*store.snapshot(), 8, 2),
		"the explicit preserve option should survive an inconclusive check");
	require(FaceStateStore::isExempt(*store.snapshot(), 8, 1, true),
		"carried decisions should cover delayed frames from the prior epoch");

	store.setPreserveOnRecovery(false);
	FaceTrackDecision invalidated = store.getDecision(8, 2);
	require(invalidated.status == FaceStatus::UNKNOWN,
		"disabling the option should invalidate carried decisions");

	store.setPreserveOnRecovery(true);
	require(store.getDecision(8, 2).status == FaceStatus::UNKNOWN,
		"enabling the option must not retroactively exempt a track");

	FaceStateStore notMeStore;
	notMeStore.setPreserveOnRecovery(true);
	notMeStore.synchronizeTracks({{9, 1}});
	uint64_t notMeToken = 0;
	require(notMeStore.beginCheck(9, 1, 1000, nullptr, &notMeToken),
		"non-self check should start");
	require(notMeStore.completeCheck(9, 1, notMeToken, FaceCheckOutcome::MISMATCH,
					 0.1f, 0, 1000, FaceRetryPolicy{}),
		"non-self result should commit");
	notMeStore.synchronizeTracks({{9, 2}});
	require(notMeStore.getDecision(9, 2).status == FaceStatus::UNKNOWN,
		"non-self decisions must not carry across recovery");
}

static void testFaceSnapshotsUnderConcurrency()
{
	FaceStateStore store;
	std::atomic<bool> done{false};
	std::atomic<bool> valid{true};

	std::thread writer([&] {
		for (uint64_t epoch = 1; epoch <= 2000; ++epoch) {
			store.synchronizeTracks({{1, epoch}});
			uint64_t checkToken = 0;
			if (store.beginCheck(1, epoch, epoch, nullptr, &checkToken))
				store.completeCheck(1, epoch, checkToken, FaceCheckOutcome::MATCH,
						    0.8f, 0, epoch, FaceRetryPolicy{});
		}
		done = true;
	});

	std::thread reader([&] {
		while (!done) {
			auto snapshot = store.snapshot();
			for (const auto &[id, decision] : *snapshot) {
				if (id == 0 || decision.identityEpoch == 0)
					valid = false;
			}
		}
	});

	writer.join();
	reader.join();
	require(valid, "published face snapshots must remain internally valid");
}

static void testNoFaceRemainsUnknownAndRetriesSoon()
{
	FaceStateStore store;
	FaceRetryPolicy policy;
	policy.noFaceRetryMs = 120;
	store.synchronizeTracks({{11, 1}});
	uint64_t checkToken = 0;
	require(store.beginCheck(11, 1, 1000, nullptr, &checkToken),
		"new track should be checked immediately");
	require(store.completeCheck(11, 1, checkToken, FaceCheckOutcome::NO_FACE,
				    std::nullopt, 1, 1000, policy),
		"no-face result should complete");
	FaceTrackDecision decision = store.getDecision(11, 1);
	require(decision.status == FaceStatus::UNKNOWN,
		"no detected face must remain unknown rather than becoming not-me");
	require(decision.lastOutcome == FaceCheckOutcome::NO_FACE && !decision.checkInFlight,
		"no-face outcome should be observable and release the in-flight check");
	require(!store.beginCheck(11, 1, 1119), "retry should respect the short backoff");
	require(store.beginCheck(11, 1, 1120), "no-face retry should become due by time");
}

static void testRecoveredSelfRetriesAfterAnOccludedFace()
{
	FaceStateStore store;
	FaceRetryPolicy policy;
	policy.noFaceRetryMs = 150;
	confirmSelf(store, 21, 1);

	store.synchronizeTracks({{21, 2}});
	bool priority = false;
	uint64_t firstToken = 0;
	require(store.beginCheck(21, 2, 2000, &priority, &firstToken) && priority,
		"a recovered self track should be checked immediately with priority");
	require(store.completeCheck(21, 2, firstToken, FaceCheckOutcome::NO_FACE,
				    std::nullopt, 1, 2000, policy),
		"an occluded recovery check should remain inconclusive");
	require(!FaceStateStore::isExempt(*store.snapshot(), 21, 2),
		"privacy-first recovery must stay masked while the face is not visible");

	uint64_t retryToken = 0;
	require(store.beginCheck(21, 2, 2150, &priority, &retryToken) && priority,
		"the recovered track should retry promptly and retain priority");
	require(store.completeCheck(21, 2, retryToken, FaceCheckOutcome::MATCH, 0.9f,
				    1, 2150, policy),
		"the first visible matching face should restore the exemption");
	require(FaceStateStore::isExempt(*store.snapshot(), 21, 2),
		"the recovered self track should be unmasked after verification");
}

static void testStaleFaceResultCannotCompleteANewerCheck()
{
	FaceStateStore store;
	store.synchronizeTracks({{12, 1}});
	uint64_t staleToken = 0;
	require(store.beginCheck(12, 1, 1000, nullptr, &staleToken),
		"initial check should start");
	store.clear();
	store.synchronizeTracks({{12, 1}});
	uint64_t currentToken = 0;
	require(store.beginCheck(12, 1, 1001, nullptr, &currentToken),
		"replacement check should start");
	require(currentToken != staleToken, "check tokens must not be reused after reset");
	require(!store.completeCheck(12, 1, staleToken, FaceCheckOutcome::MATCH, 0.9f,
				     1, 1002, FaceRetryPolicy{}),
		"a stale result must not complete the replacement check");
	require(store.getDecision(12, 1).checkInFlight,
		"the current check must remain in flight after a stale result");
	require(store.completeCheck(12, 1, currentToken, FaceCheckOutcome::MATCH, 0.9f,
				    1, 1002, FaceRetryPolicy{}),
		"the current token should complete normally");
}

static bool checkOnce(FaceStateStore &store, uint64_t id, uint64_t epoch, uint64_t nowMs,
		      FaceCheckOutcome outcome, std::optional<float> similarity, int maxExempt,
		      bool *priority = nullptr)
{
	uint64_t token = 0;
	if (!store.beginCheck(id, epoch, nowMs, priority, &token))
		return false;
	return store.completeCheck(id, epoch, token, outcome, similarity, maxExempt, nowMs,
				   FaceRetryPolicy{});
}

static void testExemptConflictPrefersLargerFace()
{
	// Track 1 holds the only slot; track 2 is a much larger face (the selfie streamer).
	FaceStateStore store;
	store.synchronizeTracks({{1, 1, 100.0f}, {2, 1, 400.0f}});
	require(checkOnce(store, 1, 1, 1000, FaceCheckOutcome::MATCH, 0.70f, 1),
		"first match should take the slot");
	require(checkOnce(store, 2, 1, 1000, FaceCheckOutcome::MATCH, 0.66f, 1),
		"second match should complete");
	auto snapshot = store.snapshot();
	require(FaceStateStore::isExempt(*snapshot, 2, 1),
		"a much larger matching face should take the exempt slot");
	require(!FaceStateStore::isExempt(*snapshot, 1, 1),
		"the displaced holder must be masked until it is verified again");
	bool priority = false;
	require(store.beginCheck(1, 1, 1000, &priority) && priority,
		"the displaced holder should be rechecked immediately");
}

static void testExemptConflictRevalidatesHolder()
{
	FaceStateStore store;
	store.synchronizeTracks({{1, 1, 300.0f}, {2, 1, 300.0f}});
	require(checkOnce(store, 1, 1, 1000, FaceCheckOutcome::MATCH, 0.70f, 1),
		"holder should match");
	require(checkOnce(store, 2, 1, 1000, FaceCheckOutcome::MATCH, 0.68f, 1),
		"similar candidate should complete");
	require(!FaceStateStore::isExempt(*store.snapshot(), 2, 1),
		"a similar candidate must not displace the holder");

	// The holder is rechecked; an inconclusive result keeps the exemption.
	bool priority = false;
	require(checkOnce(store, 1, 1, 1001, FaceCheckOutcome::NO_FACE, std::nullopt, 1,
			  &priority) &&
			priority,
		"holder should be revalidated with priority");
	require(FaceStateStore::isExempt(*store.snapshot(), 1, 1),
		"a turned-away face must not remove the verified holder's exemption");
	require(!store.beginCheck(1, 1, 1002), "an inconclusive revalidation settles the holder");

	// A new conflict triggers another revalidation; a clear mismatch frees the slot.
	require(checkOnce(store, 2, 1, 1200, FaceCheckOutcome::MATCH, 0.68f, 1),
		"candidate retries soon after a conflict");
	require(checkOnce(store, 1, 1, 1201, FaceCheckOutcome::MISMATCH, 0.10f, 1),
		"holder revalidation should complete");
	require(!FaceStateStore::isExempt(*store.snapshot(), 1, 1),
		"a holder that clearly mismatches loses the exemption");
	require(checkOnce(store, 2, 1, 1400, FaceCheckOutcome::MATCH, 0.68f, 1),
		"candidate should match again");
	require(FaceStateStore::isExempt(*store.snapshot(), 2, 1),
		"the candidate takes the freed slot");
}

static FaceInferenceJob makeFaceJob(uint64_t id, bool priority)
{
	FaceInferenceJob job;
	Object object;
	object.id = id;
	job.requests.push_back({object, id + 100});
	job.priority = priority;
	return job;
}

static void testFaceQueueDoesNotOverwritePendingWork()
{
	FaceInferenceQueue queue(2);
	require(queue.tryPush(makeFaceJob(1, false)), "first face job should queue");
	require(queue.tryPush(makeFaceJob(2, true)), "priority face job should queue");
	require(!queue.tryPush(makeFaceJob(3, false)),
		"a full queue must reject new work instead of overwriting pending checks");

	FaceInferenceJob job;
	require(queue.tryPop(job) && job.requests[0].object.id == 2,
		"priority recovery should run before ordinary checks");
	require(queue.tryPop(job) && job.requests[0].object.id == 1,
		"the original pending job must remain queued");
	require(!queue.tryPop(job), "queue should be empty after both jobs are consumed");
}

static Object makeFace(float x, float y, float width, float height, float confidence)
{
	Object face;
	face.rect = cv::Rect2f(x, y, width, height);
	face.prob = confidence;
	return face;
}

static void testReferenceFaceSelectionRejectsWeakEnrollmentImages()
{
	const cv::Size imageSize(1000, 1000);
	auto selected = selectReferenceFace(
		{makeFace(10, 10, 300, 300, 0.8f), makeFace(500, 50, 80, 80, 0.99f)},
		imageSize);
	require(selected.status == ReferenceFaceSelectionStatus::SELECTED &&
			selected.index == 0,
		"the largest usable reference face should be selected");

	auto ambiguous = selectReferenceFace(
		{makeFace(10, 10, 250, 250, 0.9f), makeFace(400, 10, 230, 230, 0.9f)},
		imageSize);
	require(ambiguous.status == ReferenceFaceSelectionStatus::AMBIGUOUS,
		"similarly sized faces should make a reference image ambiguous");

	auto tooSmall = selectReferenceFace({makeFace(10, 10, 20, 20, 0.99f)}, imageSize);
	require(tooSmall.status == ReferenceFaceSelectionStatus::TOO_SMALL,
		"tiny reference faces should be rejected");
}

static void testMappedFrameOwnership()
{
	constexpr uint32_t width = 2;
	constexpr uint32_t height = 2;
	constexpr uint32_t linesize = 12;
	std::vector<uint8_t> mapped(linesize * height, 0xEE);
	for (uint32_t y = 0; y < height; ++y) {
		for (uint32_t x = 0; x < width * 4; ++x)
			mapped[y * linesize + x] = (uint8_t)(y * 20 + x);
	}

	cv::Mat owned = copyMappedBGRA(mapped.data(), linesize, width, height);
	require(!owned.empty() && owned.cols == 2 && owned.rows == 2,
		"mapped frame should be copied");
	const uint8_t expected = owned.ptr<uint8_t>(1)[3];
	std::fill(mapped.begin(), mapped.end(), 0);
	require(owned.ptr<uint8_t>(1)[3] == expected,
		"owned frame must not reference unmapped staging memory");
}

static void testUnlimitedMaskGeometry()
{
	for (size_t count : {size_t{0}, size_t{64}, size_t{65}, size_t{256}}) {
		std::vector<MaskRect> rects(count, {10.0f, 20.0f, 30.0f, 40.0f});
		auto vertices = buildMaskVertices(rects, 5.0f);
		require(vertices.size() == count * 6,
			"mask geometry must not truncate rectangle count");
	}
	require(shouldForceFullscreenMask(false, true),
		"rectangle collection failure must force full-screen masking");
	require(shouldForceFullscreenMask(true, false),
		"mask texture failure must force full-screen masking");
	require(!shouldForceFullscreenMask(true, true),
		"a valid mask texture should use normal masking");
}

static void testShaderMaskContract()
{
	for (const char *name : {"masking.effect", "pixelate.effect"}) {
		std::ifstream input(std::string(OBS_DETECT_SOURCE_DIR) + "/data/effects/" + name);
		require(input.is_open(), "mask shader should be readable");
		std::ostringstream contents;
		contents << input.rdbuf();
		const std::string shader = contents.str();
		require(shader.find("mask_texture") != std::string::npos,
			"mask shader should sample the GPU mask texture");
		require(shader.find("mask_rects") == std::string::npos,
			"mask shader must not restore the fixed rectangle array");
	}

	// D3D11's HLSL compiler rejects these keywords as identifiers, while the OpenGL
	// backend accepts them, so CI on macOS/Linux would not notice.
	const std::regex reservedIdentifier(
		R"(\b(point|line|lineadj|triangle|triangleadj|precise|sample|centroid)\b)");
	for (const char *name :
	     {"masking.effect", "pixelate.effect", "kawase_blur.effect", "mask_shape.effect"}) {
		std::ifstream input(std::string(OBS_DETECT_SOURCE_DIR) + "/data/effects/" + name);
		require(input.is_open(), "effect file should be readable");
		std::ostringstream contents;
		contents << input.rdbuf();
		const std::string shader = contents.str();
		require(!std::regex_search(shader, reservedIdentifier),
			"effects must not use HLSL reserved keywords as identifiers");
	}
}

static void testZeroCopySelection()
{
	InferenceFrameRequirements requirements;
	require(!canUseGpuZeroCopy(true, false, true, true, requirements),
		"a model without capability must use the owned frame path");
	require(canUseGpuZeroCopy(true, true, true, true, requirements),
		"a healthy capable model may use zero-copy");
	require(!canUseGpuZeroCopy(true, true, true, false, requirements),
		"a failed zero-copy runtime must fall back");
	requirements.faceExclusionEnabled = true;
	require(!canUseGpuZeroCopy(true, true, true, true, requirements),
		"face exclusion requires an owned frame");
}

static void testStateHistorySnapshotMatchesLiveQueries()
{
	StateHistoryManager history;
	auto state = [](int id, float x) {
		TrackedObjectState s;
		s.id = id;
		s.rect = cv::Rect2f(x, 0.0f, 10.0f, 10.0f);
		s.is_confirmed = true;
		return s;
	};
	history.push_frame_state(10, {{1, state(1, 1.0f)}});
	history.push_frame_state(12, {{1, state(1, 2.0f)}, {2, state(2, 5.0f)}});
	history.push_frame_state(15, {{2, state(2, 6.0f)}});

	const auto snapshot = history.get_buffer_copy();
	for (uint64_t frame : {9, 10, 11, 12, 14, 15, 20}) {
		for (int id : {1, 2, 3}) {
			TrackedObjectState live, copied;
			const bool liveFound = history.get_object_state(frame, id, live);
			const bool copiedFound =
				StateHistoryManager::find_object_state(snapshot, frame, id, copied);
			require(liveFound == copiedFound && (!liveFound || live.rect == copied.rect),
				"snapshot lookups must match the live history");
		}
		require(history.get_all_active_ids(frame, 20) ==
				StateHistoryManager::collect_active_ids(snapshot, frame, 20),
			"snapshot active ids must match the live history");
	}
	TrackedObjectState missing;
	require(!StateHistoryManager::find_object_state(snapshot, 15, 1, missing),
		"an object absent from the closest frame has disappeared");
}

int main()
{
	try {
		testRecoveryDefaultsFailClosed();
		testRecoveryPreserveOption();
		testFaceSnapshotsUnderConcurrency();
		testNoFaceRemainsUnknownAndRetriesSoon();
		testRecoveredSelfRetriesAfterAnOccludedFace();
		testStaleFaceResultCannotCompleteANewerCheck();
		testExemptConflictPrefersLargerFace();
		testExemptConflictRevalidatesHolder();
		testFaceQueueDoesNotOverwritePendingWork();
		testReferenceFaceSelectionRejectsWeakEnrollmentImages();
		testMappedFrameOwnership();
		testUnlimitedMaskGeometry();
		testShaderMaskContract();
		testZeroCopySelection();
		testStateHistorySnapshotMatchesLiveQueries();
	} catch (const std::exception &error) {
		std::cerr << "Safety test failed: " << error.what() << '\n';
		return 1;
	}

	std::cout << "All safety tests passed\n";
	return 0;
}
