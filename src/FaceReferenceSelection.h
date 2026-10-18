#ifndef FACE_REFERENCE_SELECTION_H
#define FACE_REFERENCE_SELECTION_H

#include "ort-model/types.hpp"

#include <cstddef>
#include <vector>

enum class ReferenceFaceSelectionStatus { SELECTED, NO_FACE, TOO_SMALL, AMBIGUOUS };

struct ReferenceFaceSelection {
	ReferenceFaceSelectionStatus status = ReferenceFaceSelectionStatus::NO_FACE;
	size_t index = 0;
};

ReferenceFaceSelection selectReferenceFace(const std::vector<Object> &faces,
					   const cv::Size &imageSize);

#endif // FACE_REFERENCE_SELECTION_H
