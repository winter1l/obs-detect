#include "obs-utils.h"
#include "plugin-support.h"

#include <obs-module.h>

#include <algorithm>

// Renders one full-screen pass of `technique` from `effect` into `texrender`.
static bool render_pass(gs_texrender_t *texrender, gs_effect_t *effect, const char *technique,
			gs_texture_t *input, uint32_t width, uint32_t height)
{
	gs_texrender_reset(texrender);
	if (!gs_texrender_begin(texrender, width, height))
		return false;

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), input);
	while (gs_effect_loop(effect, technique)) {
		gs_draw_sprite(input, 0, width, height);
	}
	gs_blend_state_pop();
	gs_texrender_end(texrender);
	return true;
}

static gs_texrender_t *ensure_texrender(gs_texrender_t *&texrender)
{
	if (!texrender)
		texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	return texrender;
}

gs_texture_t *blur_image(struct filter_data *tf, uint32_t width, uint32_t height,
			 gs_texture_t *sourceTexture)
{
	if (!sourceTexture)
		return nullptr;
	if (tf->kawaseBlurEffect == nullptr) {
		obs_log(LOG_ERROR, "tf->kawaseBlurEffect is null");
		return nullptr;
	}
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "yOffset");

	// Passes alternate between two render targets, so no texture is created or copied.
	// Every pass covers the whole frame; the masking composite limits the result to the
	// masked regions.
	// At least one pass, so a stored strength of 0 cannot leave masked regions sharp.
	const int passes = std::max(1, tf->maskingBlurRadius);
	gs_texture_t *current = sourceTexture;
	for (int i = 0; i < passes; i++) {
		gs_texrender_t *target = ensure_texrender(tf->blurTexrenders[i & 1]);
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);
		if (!target ||
		    !render_pass(target, tf->kawaseBlurEffect, "Draw", current, width, height))
			return nullptr;
		current = gs_texrender_get_texture(target);
	}
	return current;
}

gs_texture_t *pixelate_image(struct filter_data *tf, uint32_t width, uint32_t height,
			     gs_texture_t *alphaTexture, float pixelateRadius,
			     gs_texture_t *sourceTexture)
{
	if (!sourceTexture)
		return nullptr;
	if (tf->pixelateEffect == nullptr) {
		obs_log(LOG_ERROR, "tf->pixelateEffect is null");
		return nullptr;
	}
	gs_eparam_t *pixel_size = gs_effect_get_param_by_name(tf->pixelateEffect, "pixel_size");
	gs_eparam_t *tex_size = gs_effect_get_param_by_name(tf->pixelateEffect, "tex_size");
	gs_eparam_t *maskTexture = gs_effect_get_param_by_name(tf->pixelateEffect, "mask_texture");
	gs_eparam_t *forceFullMask =
		gs_effect_get_param_by_name(tf->pixelateEffect, "force_full_mask");
	// Always rebind, even to null, so a texture from an earlier frame is never sampled.
	if (maskTexture)
		gs_effect_set_texture(maskTexture, alphaTexture);
	if (forceFullMask)
		gs_effect_set_bool(forceFullMask, tf->maskForceFullscreen);
	gs_effect_set_float(pixel_size, pixelateRadius);
	vec2 texsize_vec;
	vec2_set(&texsize_vec, (float)width, (float)height);
	gs_effect_set_vec2(tex_size, &texsize_vec);

	gs_texrender_t *target = ensure_texrender(tf->pixelateTexrender);
	if (!target ||
	    !render_pass(target, tf->pixelateEffect, "Draw", sourceTexture, width, height))
		return nullptr;
	return gs_texrender_get_texture(target);
}
