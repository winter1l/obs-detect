#ifndef OBS_UTILS_H
#define OBS_UTILS_H

#include "FilterData.h"

// The returned textures belong to render targets in `tf` and stay valid until the next
// call; callers must not destroy them. Both return nullptr when the effect could not be
// rendered, so callers can cover the masked regions instead of showing them unchanged.

// Blurs the whole frame; callers restrict the result to the mask when compositing.
gs_texture_t *blur_image(struct filter_data *tf, uint32_t width, uint32_t height,
			 gs_texture_t *sourceTexture);

gs_texture_t *pixelate_image(struct filter_data *tf, uint32_t width, uint32_t height,
			     gs_texture_t *alphaTexture, float pixelateRadius,
			     gs_texture_t *sourceTexture);

#endif /* OBS_UTILS_H */
