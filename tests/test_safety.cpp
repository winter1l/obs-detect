#include "FaceStateStore.h"
#include "FrameUtils.h"
#include "InferenceSafety.h"
#include "MaskGeometry.h"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
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
	require(store.beginCheck(id, epoch), "face check should start");
	require(store.commit(id, epoch, FaceStatus::IS_ME, 0.9f, 0), "face result should commit");
}

static void testRecoveryDefaultsFailClosed()
{
	FaceStateStore store;
	confirmSelf(store, 7, 1);
	store.synchronizeTracks({{7, 2}});
	auto snapshot = store.snapshot();
	require(!FaceStateStore::isExempt(*snapshot, 7, 2),
		"recovered identity must be masked by default");
	require(!store.commit(7, 1, FaceStatus::IS_ME, 0.95f, 0),
		"stale epoch result must be discarded");
	require(store.beginCheck(7, 2), "recovered identity should be rechecked");
	require(store.commit(7, 2, FaceStatus::IS_ME, 0.91f, 0),
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
	require(notMeStore.beginCheck(9, 1), "non-self check should start");
	require(notMeStore.commit(9, 1, FaceStatus::NOT_ME, 0.1f, 0),
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
			if (store.beginCheck(1, epoch))
				store.commit(1, epoch, FaceStatus::IS_ME, 0.8f, 0);
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
		auto vertices = buildMaskVertices(rects, 5.0f, 300.0f);
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
	for (const char *name : {"masking.effect", "pixelate.effect", "kawase_blur.effect"}) {
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

int main()
{
	try {
		testRecoveryDefaultsFailClosed();
		testRecoveryPreserveOption();
		testFaceSnapshotsUnderConcurrency();
		testMappedFrameOwnership();
		testUnlimitedMaskGeometry();
		testShaderMaskContract();
		testZeroCopySelection();
	} catch (const std::exception &error) {
		std::cerr << "Safety test failed: " << error.what() << '\n';
		return 1;
	}

	std::cout << "All safety tests passed\n";
	return 0;
}
