#ifndef STAGE_PROFILER_H
#define STAGE_PROFILER_H

#include <array>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

// Collects per-stage CPU timings from the graphics and inference threads and produces
// a periodic summary (count, mean, p95, max) for the OBS log.
class StageProfiler {
public:
	enum Stage : size_t {
		// Graphics thread
		RenderTotal,
		RenderCapture,
		StageReadback,
		CopyMapped,
		TickTotal,
		TickClone,
		PreviewUpload,
		MaskBuild,
		MaskDraw,
		// Inference thread
		InferenceTotal,
		InferenceCvtColor,
		ModelPreprocess,
		ModelRun,
		ModelPostprocess,
		TrackerUpdate,
		OverlayBuild,
		PreviewHistoryStore,
		FaceJobClone,
		StageCount
	};

	static const char *stageName(Stage stage);

	void add(Stage stage, double milliseconds);
	// Returns true and fills `report` once per interval; resets the collected samples.
	bool takeReport(std::chrono::steady_clock::time_point now,
			std::chrono::steady_clock::duration interval, std::string &report);
	void clear();

private:
	std::mutex mutex_;
	std::array<std::vector<float>, StageCount> samples_;
	std::chrono::steady_clock::time_point windowStart_{};
};

// Adds the elapsed time of its scope to a stage. A null profiler disables it.
class ScopedStage {
public:
	ScopedStage(StageProfiler *profiler, StageProfiler::Stage stage)
		: profiler_(profiler),
		  stage_(stage),
		  start_(profiler ? std::chrono::steady_clock::now()
				  : std::chrono::steady_clock::time_point{})
	{
	}
	~ScopedStage() { stop(); }
	void stop()
	{
		if (!profiler_)
			return;
		const auto elapsed = std::chrono::steady_clock::now() - start_;
		profiler_->add(stage_,
			       std::chrono::duration<double, std::milli>(elapsed).count());
		profiler_ = nullptr;
	}
	ScopedStage(const ScopedStage &) = delete;
	ScopedStage &operator=(const ScopedStage &) = delete;

private:
	StageProfiler *profiler_;
	StageProfiler::Stage stage_;
	std::chrono::steady_clock::time_point start_;
};

#endif // STAGE_PROFILER_H
