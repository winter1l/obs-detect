#ifndef MASK_GEOMETRY_H
#define MASK_GEOMETRY_H

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

inline bool shouldForceFullscreenMask(bool collectionSucceeded, bool maskTextureAvailable)
{
	return !collectionSucceeded || !maskTextureAvailable;
}

#endif // MASK_GEOMETRY_H
