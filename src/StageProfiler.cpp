#include "StageProfiler.h"

#include <algorithm>
#include <cstdio>

const char *StageProfiler::stageName(Stage stage)
{
	switch (stage) {
	case RenderTotal:
		return "render.total";
	case RenderCapture:
		return "render.capture";
	case StageReadback:
		return "render.stage_map";
	case CopyMapped:
		return "render.copy_mapped";
	case TickTotal:
		return "tick.total";
	case TickClone:
		return "tick.clone";
	case PreviewUpload:
		return "render.preview_upload";
	case MaskBuild:
		return "render.mask_build";
	case MaskDraw:
		return "render.mask_draw";
	case InferenceTotal:
		return "infer.total";
	case InferenceCvtColor:
		return "infer.cvt_color";
	case ModelPreprocess:
		return "infer.preprocess";
	case ModelRun:
		return "infer.session_run";
	case ModelPostprocess:
		return "infer.postprocess";
	case TrackerUpdate:
		return "infer.tracker";
	case OverlayBuild:
		return "infer.overlay";
	case PreviewHistoryStore:
		return "infer.preview_store";
	case FaceJobClone:
		return "infer.face_job_clone";
	case StageCount:
		break;
	}
	return "unknown";
}

void StageProfiler::add(Stage stage, double milliseconds)
{
	if (stage >= StageCount)
		return;
	std::lock_guard<std::mutex> lock(mutex_);
	samples_[stage].push_back((float)milliseconds);
}

bool StageProfiler::takeReport(std::chrono::steady_clock::time_point now,
			       std::chrono::steady_clock::duration interval, std::string &report)
{
	std::array<std::vector<float>, StageCount> samples;
	double seconds = 0.0;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (windowStart_.time_since_epoch().count() == 0) {
			windowStart_ = now;
			return false;
		}
		if (now - windowStart_ < interval)
			return false;
		seconds = std::chrono::duration<double>(now - windowStart_).count();
		samples.swap(samples_);
		windowStart_ = now;
	}

	report.clear();
	char line[160];
	snprintf(line, sizeof(line), "[Profile] %.1f s window (stage: n/s, mean, p95, max ms)",
		 seconds);
	report += line;
	for (size_t stage = 0; stage < StageCount; ++stage) {
		std::vector<float> &values = samples[stage];
		if (values.empty())
			continue;
		std::sort(values.begin(), values.end());
		double sum = 0.0;
		for (float value : values)
			sum += value;
		const size_t p95Index =
			std::min(values.size() - 1, (size_t)((double)values.size() * 0.95));
		snprintf(line, sizeof(line), "\n  %-22s %6.1f/s %7.2f %7.2f %7.2f",
			 stageName((Stage)stage), (double)values.size() / seconds,
			 sum / (double)values.size(), (double)values[p95Index],
			 (double)values.back());
		report += line;
	}
	return true;
}

void StageProfiler::clear()
{
	std::lock_guard<std::mutex> lock(mutex_);
	for (auto &values : samples_)
		values.clear();
	windowStart_ = {};
}
