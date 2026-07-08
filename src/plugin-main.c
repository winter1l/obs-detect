/*
obs-detect
Copyright (C) 2026 Chang jaeyong <jangjaeyong0412@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return obs_module_text("DetectFilterPlugin");
}

extern struct obs_source_info detect_filter_info;

bool obs_module_load(void)
{
	obs_register_source(&detect_filter_info);
	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

#include "obs-websocket-api.h"

obs_websocket_vendor websocket_vendor = NULL;

void obs_module_post_load(void)
{
	websocket_vendor = obs_websocket_register_vendor("obs-detect");
	if (websocket_vendor) {
		obs_log(LOG_INFO, "Successfully registered obs-websocket vendor for obs-detect");
	}
}

void emit_face_exclusion_update(int count)
{
	if (!websocket_vendor) return;

	obs_data_t *event_data = obs_data_create();
	obs_data_set_int(event_data, "excluded_count", count);

	obs_websocket_vendor_emit_event(websocket_vendor, "face_exclusion_update", event_data);

	obs_data_release(event_data);
}

void obs_module_unload(void)
{
	obs_log(LOG_INFO, "plugin unloaded");
}
