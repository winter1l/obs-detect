#ifndef SETTINGS_UNITS_H
#define SETTINGS_UNITS_H

#include <algorithm>
#include <cmath>

// The face recheck delay is edited in seconds, while the stored
// "face_inference_interval" setting keeps its original meaning (frames at 30 fps, 5-60)
// so existing scene collections and older plugin versions read the same value.
inline int faceRecheckFramesFromSeconds(double seconds)
{
	return std::clamp((int)std::lround(seconds * 30.0), 5, 60);
}

inline double faceRecheckSecondsFromFrames(int frames)
{
	return (double)std::clamp(frames, 5, 60) / 30.0;
}

#endif // SETTINGS_UNITS_H
