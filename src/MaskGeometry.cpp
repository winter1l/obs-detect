#include "MaskGeometry.h"

#include <algorithm>

std::vector<MaskVertex> buildMaskVertices(const std::vector<MaskRect> &rects, float featherPixels,
					  float blurWorkPaddingPixels)
{
	std::vector<MaskVertex> vertices;
	vertices.reserve(rects.size() * 6);
	const float padding = std::max(0.0f, featherPixels) + std::max(0.0f, blurWorkPaddingPixels);

	for (const MaskRect &rect : rects) {
		if (rect.width <= 0.0f || rect.height <= 0.0f)
			continue;

		const float left = rect.x - padding;
		const float top = rect.y - padding;
		const float right = rect.x + rect.width + padding;
		const float bottom = rect.y + rect.height + padding;

		auto add = [&](float x, float y, float u, float v) {
			vertices.push_back({x, y, u, v, rect.width, rect.height});
		};

		add(left, top, 0.0f, 0.0f);
		add(right, top, 1.0f, 0.0f);
		add(left, bottom, 0.0f, 1.0f);
		add(left, bottom, 0.0f, 1.0f);
		add(right, top, 1.0f, 0.0f);
		add(right, bottom, 1.0f, 1.0f);
	}

	return vertices;
}
