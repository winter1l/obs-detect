#include "FrameUtils.h"

cv::Mat copyMappedBGRA(const uint8_t *data, uint32_t linesize, uint32_t width, uint32_t height)
{
	if (!data || width == 0 || height == 0 || linesize < width * 4)
		return {};

	const cv::Mat mapped((int)height, (int)width, CV_8UC4, const_cast<uint8_t *>(data),
			     (size_t)linesize);
	return mapped.clone();
}
