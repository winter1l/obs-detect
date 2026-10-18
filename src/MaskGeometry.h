#ifndef MASK_GEOMETRY_H
#define MASK_GEOMETRY_H

#include <cmath>
#include <vector>

struct MaskRect {
	float x = 0.0f;
	float y = 0.0f;
	float width = 0.0f;
	float height = 0.0f;
};

struct MaskVertex {
	float x;
	float y;
	float u;
	float v;
	float rectWidth;
	float rectHeight;
};

std::vector<MaskVertex> buildMaskVertices(const std::vector<MaskRect> &rects, float featherPixels);

// A rectangle that can be drawn: finite, with a positive width and height. NaN passes
// "width <= 0" checks, so it is tested explicitly.
inline bool isDrawableMaskRect(const MaskRect &rect)
{
	return std::isfinite(rect.x) && std::isfinite(rect.y) && std::isfinite(rect.width) &&
	       std::isfinite(rect.height) && rect.width > 0.0f && rect.height > 0.0f;
}

inline bool shouldForceFullscreenMask(bool collectionSucceeded, bool maskTextureAvailable)
{
	return !collectionSucceeded || !maskTextureAvailable;
}

#endif // MASK_GEOMETRY_H
