#include "FaceReferenceSelection.h"

#include <algorithm>
#include <utility>
#include <vector>

ReferenceFaceSelection selectReferenceFace(const std::vector<Object> &faces,
					   const cv::Size &imageSize)
{
	if (faces.empty() || imageSize.width <= 0 || imageSize.height <= 0)
		return {ReferenceFaceSelectionStatus::NO_FACE, 0};

	constexpr float minFaceSidePixels = 48.0f;
	constexpr float minFaceAreaRatio = 0.0025f;
	constexpr float ambiguousAreaRatio = 0.65f;
	const float imageArea = (float)imageSize.width * (float)imageSize.height;
	const cv::Rect2f imageRect(0.0f, 0.0f, (float)imageSize.width,
				  (float)imageSize.height);

	std::vector<std::pair<float, size_t>> usable;
	for (size_t index = 0; index < faces.size(); ++index) {
		const cv::Rect2f clipped = faces[index].rect & imageRect;
		const float area = clipped.area();
		if (clipped.width < minFaceSidePixels || clipped.height < minFaceSidePixels ||
		    area < imageArea * minFaceAreaRatio)
			continue;
		usable.emplace_back(area, index);
	}

	if (usable.empty())
		return {ReferenceFaceSelectionStatus::TOO_SMALL, 0};

	std::sort(usable.begin(), usable.end(),
		  [](const auto &left, const auto &right) { return left.first > right.first; });
	if (usable.size() > 1 && usable[1].first >= usable[0].first * ambiguousAreaRatio)
		return {ReferenceFaceSelectionStatus::AMBIGUOUS, 0};

	return {ReferenceFaceSelectionStatus::SELECTED, usable[0].second};
}
