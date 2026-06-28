#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#include "sface/SFace.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _WIN32
#define NOMINMAX
#include <util/dstr.h>
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <memory>
#include <algorithm>
#include <exception>
#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "edgeyolo/edgeyolo_onnxruntime.hpp"
#include "ort-model/YOLOv8FaceONNX.hpp"
#include "yunet/YuNet.h"

#define EXTERNAL_MODEL_SIZE "!!!EXTERNAL_MODEL!!!"
#define FACE_YUNET_MODEL_SIZE "!!!FACE_YUNET!!!"
#define FACE_YOLO_N_MODEL_SIZE "!!!FACE_YOLO_N!!!"
#define FACE_YOLO_S_MODEL_SIZE "!!!FACE_YOLO_S!!!"

struct detect_filter : public filter_data {};

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop,
			    const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}



void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		// capitalize the first letter of the class name
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}

	// sort the vector based on the class names
	std::sort(indexed_classes.begin(), indexed_classes.end(),
		  [](const std::pair<size_t, std::string> &a,
		     const std::pair<size_t, std::string> &b) { return a.second < b.second; });

	// clear the object category list
	obs_property_list_clear(object_category);

	// add the sorted classes to the property list
	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	// add the sorted classes to the property list
	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings, struct detect_filter *tf_)
{
	if (model_file == nullptr || model_file[0] == '\0' || strlen(model_file) == 0) {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	// read the '.json' file near the model file to find the class names
	std::string json_file = model_file;
	json_file.replace(json_file.find(".onnx"), 5, ".json");
	std::ifstream file(json_file);
	if (!file.is_open()) {
		obs_data_set_string(settings, "error", "JSON file not found");
		obs_log(LOG_ERROR, "JSON file not found: %s", json_file.c_str());
	} else {
		obs_data_set_string(settings, "error", "");
		// parse the JSON file
		nlohmann::json j;
		file >> j;
		if (j.contains("names")) {
			std::vector<std::string> labels = j["names"];
			set_class_names_on_object_category(
				obs_properties_get(props_, "object_category"), labels);
			set_class_names_on_object_category(
				obs_properties_get(props_, "face_category"), labels);
			set_class_names_on_object_category(
				obs_properties_get(props_, "person_category"), labels);
			tf_->classNames = labels;
		} else {
			obs_data_set_string(settings, "error",
					    "JSON file does not contain 'names' field");
			obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
		}
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));
	obs_properties_add_bool(props, "sync_mode", obs_module_text("SynchronousMode"));

	// add dropdown selection for object category selection: "All", or COCO classes
	obs_property_t *object_category =
		obs_properties_add_list(props, "object_category", obs_module_text("ObjectCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(object_category, edgeyolo_cpp::COCO_CLASSES);

	tf->classNames = edgeyolo_cpp::COCO_CLASSES;

	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, [](obs_properties_t *props_,
								  obs_property_t *,
								  obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "masking_group");
		obs_property_t *prop = obs_properties_get(props_, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "dilation_iterations");

		obs_property_set_visible(prop, enabled);
		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		obs_property_set_visible(masking_dilation, enabled);
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, enabled);
		}
		return true;
	});

	// add masking options drop down selection: "None", "Solid color", "Blur", "Transparent"
	obs_property_t *masking_type = obs_properties_add_list(masking_group, "masking_type",
							       obs_module_text("MaskingType"),
							       OBS_COMBO_TYPE_LIST,
							       OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(masking_type, obs_module_text("None"), "none");
	obs_property_list_add_string(masking_type, obs_module_text("SolidColor"), "solid_color");
	obs_property_list_add_string(masking_type, obs_module_text("OutputMask"), "output_mask");
	obs_property_list_add_string(masking_type, obs_module_text("Blur"), "blur");
	obs_property_list_add_string(masking_type, obs_module_text("Pixelate"), "pixelate");
	obs_property_list_add_string(masking_type, obs_module_text("Transparent"), "transparent");

	// add color picker for solid color masking
	obs_properties_add_color(masking_group, "masking_color", obs_module_text("MaskingColor"));

	// add slider for blur radius
	obs_properties_add_int_slider(masking_group, "masking_blur_radius",
				      obs_module_text("MaskingBlurRadius"), 1, 30, 1);

	// add callback to show/hide blur radius and color picker
	auto update_masking_visibility = [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius =
			obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_dilation =
			obs_properties_get(props_, "masking_dilate_iterations");
		obs_property_t *masking_dynamic_expansion =
			obs_properties_get(props_, "masking_dynamic_expansion");
		obs_property_t *masking_dynamic_base =
			obs_properties_get(props_, "masking_dynamic_expansion_base");
		obs_property_t *masking_dynamic_ratio =
			obs_properties_get(props_, "masking_dynamic_expansion_ratio");

		obs_property_set_visible(masking_color, false);
		obs_property_set_visible(masking_blur_radius, false);
		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
		const bool dynamic_enabled = obs_data_get_bool(settings, "masking_dynamic_expansion");

		obs_property_set_visible(masking_dynamic_expansion, masking_enabled);
		obs_property_set_visible(masking_dilation, masking_enabled && !dynamic_enabled);
		if (masking_dynamic_base) obs_property_set_visible(masking_dynamic_base, masking_enabled && dynamic_enabled);
		if (masking_dynamic_ratio) obs_property_set_visible(masking_dynamic_ratio, masking_enabled && dynamic_enabled);

		if (masking_type_value == "solid_color") {
			obs_property_set_visible(masking_color, masking_enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			obs_property_set_visible(masking_blur_radius, masking_enabled);
		}
		return true;
	};

	obs_property_set_modified_callback(masking_type, update_masking_visibility);

	// add slider for dilation iterations
	obs_properties_add_int_slider(masking_group, "masking_dilate_iterations",
				      obs_module_text("DilationIterations"), 0, 100, 1);
	obs_property_t *masking_dynamic_expansion = obs_properties_add_bool(
		masking_group, "masking_dynamic_expansion",
		obs_module_text("MaskingDynamicExpansion"));
	obs_property_set_modified_callback(masking_dynamic_expansion, update_masking_visibility);

	obs_properties_add_float_slider(masking_group, "masking_dynamic_expansion_base",
					obs_module_text("MaskingDynamicExpansionBase"), 0.0, 100.0, 1.0);
	obs_properties_add_float_slider(masking_group, "masking_dynamic_expansion_ratio",
					obs_module_text("MaskingDynamicExpansionRatio"), 0.0, 2.0, 0.05);

	// add options group for tracking and zoom-follow options
	obs_properties_t *tracking_group_props = obs_properties_create();
	obs_property_t *tracking_group = obs_properties_add_group(
		props, "tracking_group", obs_module_text("TrackingZoomFollowGroup"),
		OBS_GROUP_CHECKABLE, tracking_group_props);

	// add callback to show/hide tracking options
	obs_property_set_modified_callback(tracking_group, [](obs_properties_t *props_,
							      obs_property_t *,
							      obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "tracking_group");
		for (auto prop_name : {"zoom_factor", "zoom_object", "zoom_speed_factor"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add zoom factor slider
	obs_properties_add_float_slider(tracking_group_props, "zoom_factor",
					obs_module_text("ZoomFactor"), 0.0, 1.0, 0.05);

	obs_properties_add_float_slider(tracking_group_props, "zoom_speed_factor",
					obs_module_text("ZoomSpeed"), 0.0, 0.1, 0.01);

	// add object selection for zoom drop down: "Single", "All"
	obs_property_t *zoom_object = obs_properties_add_list(tracking_group_props, "zoom_object",
							      obs_module_text("ZoomObject"),
							      OBS_COMBO_TYPE_LIST,
							      OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(zoom_object, obs_module_text("SingleFirst"), "single");
	obs_property_list_add_string(zoom_object, obs_module_text("Biggest"), "biggest");
	obs_property_list_add_string(zoom_object, obs_module_text("Oldest"), "oldest");
	obs_property_list_add_string(zoom_object, obs_module_text("All"), "all");



	// add a checkable group for crop region settings
	obs_properties_t *crop_group_props = obs_properties_create();
	obs_property_t *crop_group =
		obs_properties_add_group(props, "crop_group", obs_module_text("CropGroup"),
					 OBS_GROUP_CHECKABLE, crop_group_props);

	// add callback to show/hide crop region options
	obs_property_set_modified_callback(crop_group, [](obs_properties_t *props_,
							  obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "crop_group");
		for (auto prop_name : {"crop_left", "crop_right", "crop_top", "crop_bottom"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// add crop region settings
	obs_properties_add_int_slider(crop_group_props, "crop_left", obs_module_text("CropLeft"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_right", obs_module_text("CropRight"),
				      0, 1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_top", obs_module_text("CropTop"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_bottom",
				      obs_module_text("CropBottom"), 0, 1000, 1);

	// add a text input for the currently detected object
	obs_property_t *detected_obj_prop = obs_properties_add_text(
		props, "detected_object", obs_module_text("DetectedObject"), OBS_TEXT_DEFAULT);
	// disable the text input by default
	obs_property_set_enabled(detected_obj_prop, false);

	// add threshold slider
	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.10,
					1.0, 0.025);

	obs_properties_add_int_slider(props, "min_size_threshold",
				      obs_module_text("MinSizeThreshold"), 0, 10000, 1);
	// Face Exclusion Group
	obs_properties_t *face_ex_group_props = obs_properties_create();
	obs_property_t *enable_face_exclusion =
		obs_properties_add_group(props, "enable_face_exclusion", obs_module_text("EnableFaceExclusion"),
					 OBS_GROUP_CHECKABLE, face_ex_group_props);

	obs_property_t *person_category =
		obs_properties_add_list(face_ex_group_props, "person_category", obs_module_text("PersonCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(person_category, edgeyolo_cpp::COCO_CLASSES);

	obs_properties_add_path(face_ex_group_props, "reference_face_path", obs_module_text("ReferenceFaceImage"), OBS_PATH_DIRECTORY, "", nullptr);
	obs_properties_add_float_slider(face_ex_group_props, "face_match_threshold", obs_module_text("FaceMatchThreshold"), 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(face_ex_group_props, "min_face_area_ratio", obs_module_text("MinFaceAreaRatio"), 0.0, 100.0, 0.1);
	obs_properties_add_int_slider(face_ex_group_props, "face_inference_interval", obs_module_text("FaceInferenceInterval"), 1, 120, 1);
	obs_properties_add_int_slider(face_ex_group_props, "max_exempt_persons", obs_module_text("MaxExemptPersons"), 1, 10, 1);

	// Hide subproperties completely when unchecked
	obs_property_set_modified_callback(enable_face_exclusion, [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "enable_face_exclusion");
		for (auto prop_name : {"reference_face_path", "face_match_threshold", "min_face_area_ratio", "face_inference_interval", "max_exempt_persons", "person_category"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			if (prop) obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	// SORT Tracking Group
	obs_properties_t *sort_group_props = obs_properties_create();
	obs_property_t *sort_tracking =
		obs_properties_add_group(props, "sort_tracking", obs_module_text("SORTTracking"),
					 OBS_GROUP_CHECKABLE, sort_group_props);

	obs_properties_add_int(sort_group_props, "min_hit_frames", obs_module_text("MinHitFrames"), 1, 30, 1);
	obs_properties_add_float_slider(sort_group_props, "iou_threshold", obs_module_text("IouThreshold"), 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(sort_group_props, "instant_track_area_ratio", obs_module_text("InstantTrackAreaRatio"), 0.0, 100.0, 0.1);
	obs_properties_add_int(sort_group_props, "max_unseen_frames", obs_module_text("MaxUnseenFrames"), 1, 30, 1);
	obs_properties_add_bool(sort_group_props, "show_unseen_objects", obs_module_text("ShowUnseenObjects"));
	obs_properties_add_float_slider(sort_group_props, "ghost_recovery_multiplier", obs_module_text("GhostRecoveryMultiplier"), 0.5, 5.0, 0.1);
	obs_properties_add_int(sort_group_props, "ghost_recovery_max_unseen", obs_module_text("GhostRecoveryMaxUnseen"), 1, 15, 1);

	// Hide subproperties completely when unchecked
	obs_property_set_modified_callback(sort_tracking, [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "sort_tracking");
		for (auto prop_name : {"min_hit_frames", "iou_threshold", "instant_track_area_ratio", "max_unseen_frames", "show_unseen_objects", "ghost_recovery_multiplier", "ghost_recovery_max_unseen"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			if (prop) obs_property_set_visible(prop, enabled);
		}
		if (!enabled) {
			obs_data_set_bool(settings, "show_unseen_objects", true);
		}
		return true;
	});

	/* GPU, CPU and performance Props */
	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUTensorRT"), USEGPU_TENSORRT);
#endif
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUDirectML"), USEGPU_DML);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 32, 1);

	// add drop down option for model size: Small, Medium, Large
	obs_property_t *model_size =
		obs_properties_add_list(props, "model_size", obs_module_text("ModelSize"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(model_size, obs_module_text("SmallFast"), "small");
	obs_property_list_add_string(model_size, obs_module_text("Medium"), "medium");
	obs_property_list_add_string(model_size, obs_module_text("LargeSlow"), "large");
	obs_property_list_add_string(model_size, obs_module_text("FaceTrackingModel.YuNet"),
				     FACE_YUNET_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("FaceTrackingModel.YOLOv8n"),
				     FACE_YOLO_N_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("FaceTrackingModel.YOLOv8s"),
				     FACE_YOLO_S_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("ExternalModel"),
				     EXTERNAL_MODEL_SIZE);

	// add external model file path
	obs_properties_add_path(props, "external_model_file", obs_module_text("ModelPath"),
				OBS_PATH_FILE, "EdgeYOLO onnx files (*.onnx);;all files (*.*)",
				nullptr);

	// add callback to show/hide the external model file path
	obs_property_set_modified_callback2(
		model_size,
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			std::string model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = model_size_value == EXTERNAL_MODEL_SIZE;
			bool is_face_detect = (model_size_value == FACE_YUNET_MODEL_SIZE ||
					       model_size_value == FACE_YOLO_N_MODEL_SIZE ||
					       model_size_value == FACE_YOLO_S_MODEL_SIZE);
			
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);

			if (!is_external) {
				if (is_face_detect) {
					// set the class names to COCO classes for face detection model
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						yunet::FACE_CLASSES);
					set_class_names_on_object_category(
						obs_properties_get(props_, "face_category"),
						yunet::FACE_CLASSES);
					set_class_names_on_object_category(
						obs_properties_get(props_, "person_category"),
						yunet::FACE_CLASSES);
					tf_->classNames = yunet::FACE_CLASSES;
				} else {
					// reset the class names to COCO classes for default models
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						edgeyolo_cpp::COCO_CLASSES);
					set_class_names_on_object_category(
						obs_properties_get(props_, "face_category"),
						edgeyolo_cpp::COCO_CLASSES);
					set_class_names_on_object_category(
						obs_properties_get(props_, "person_category"),
						edgeyolo_cpp::COCO_CLASSES);
					tf_->classNames = edgeyolo_cpp::COCO_CLASSES;
				}
			} else {
				// if the model path is already set - update the class names
				const char *model_file =
					obs_data_get_string(settings, "external_model_file");
				read_model_config_json_and_set_class_names(model_file, props_,
									   settings, tf_);
			}
			return true;
		},
		tf);

	// add callback on the model file path to check if the file exists
	obs_property_set_modified_callback2(
		obs_properties_get(props, "external_model_file"),
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			const char *model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = strcmp(model_size_value, EXTERNAL_MODEL_SIZE) == 0;
			if (!is_external) {
				return true;
			}
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			const char *model_file =
				obs_data_get_string(settings, "external_model_file");
			read_model_config_json_and_set_class_names(model_file, props_, settings,
								   tf_);
			return true;
		},
		tf);

	// Debug & Statistics Group (At the very bottom)
	obs_properties_t *debug_group_props = obs_properties_create();
	obs_properties_add_group(props, "debug_group", obs_module_text("DebugGroup"), OBS_GROUP_NORMAL, debug_group_props);

	obs_properties_add_bool(debug_group_props, "debug_mode", obs_module_text("DebugMode"));
	obs_properties_add_bool(debug_group_props, "enable_face_stats", obs_module_text("ShowFaceSimilarityStats"));
	
	obs_properties_add_bool(debug_group_props, "show_yunet_detections", obs_module_text("ShowYuNetDetections"));
	
	obs_property_t *enable_face_stats_log = obs_properties_add_bool(debug_group_props, "enable_face_stats_log", obs_module_text("SaveFaceStatsToCSV"));
	obs_properties_add_path(debug_group_props, "face_stats_log_path", obs_module_text("SaveFaceStatsCSVPath"), OBS_PATH_FILE_SAVE, "CSV file (*.csv);;All files (*.*)", nullptr);
	obs_properties_add_path(debug_group_props, "save_detections_path", obs_module_text("SaveDetectionsPath"), OBS_PATH_FILE_SAVE, "JSON file (*.json);;All files (*.*)", nullptr);

	obs_property_set_modified_callback(enable_face_stats_log, [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "enable_face_stats_log");
		obs_property_t *prop = obs_properties_get(props_, "face_stats_log_path");
		if (prop) obs_property_set_visible(prop, enabled);
		return true;
	});

	// Add a informative text about the plugin
	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_bool(settings, "sort_tracking", true);
	obs_data_set_default_bool(settings, "show_unseen_objects", false);
	obs_data_set_default_bool(settings, "preview", false);
	obs_data_set_default_bool(settings, "sync_mode", false);
	obs_data_set_default_bool(settings, "debug_mode", false);
	obs_data_set_default_double(settings, "face_match_threshold", 0.36);
	obs_data_set_default_double(settings, "min_face_area_ratio", 2.0);
	obs_data_set_default_int(settings, "face_inference_interval", 30);
	obs_data_set_default_int(settings, "max_exempt_persons", 1);
	obs_data_set_default_bool(settings, "enable_face_stats", false);
	obs_data_set_default_bool(settings, "enable_face_stats_log", false);
	obs_data_set_default_bool(settings, "show_yunet_detections", false);
	obs_data_set_default_string(settings, "face_stats_log_path", "");
	obs_data_set_default_int(settings, "person_category", 0);
	obs_data_set_default_int(settings, "max_unseen_frames", 10);
	obs_data_set_default_bool(settings, "show_unseen_objects", true);
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "model_size", "small");
	obs_data_set_default_int(settings, "object_category", -1);
	obs_data_set_default_bool(settings, "enable_face_exclusion", false);
	obs_data_set_default_string(settings, "reference_face_path", "");
	obs_data_set_default_double(settings, "face_match_threshold", 0.36);
	obs_data_set_default_int(settings, "person_category", -1);
	obs_data_set_default_double(settings, "min_face_area_ratio", 30.0);
	obs_data_set_default_int(settings, "face_inference_interval", 30);
	obs_data_set_default_bool(settings, "masking_group", false);
	obs_data_set_default_string(settings, "masking_type", "none");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_dilate_iterations", 0);
	obs_data_set_default_bool(settings, "masking_dynamic_expansion", false);
	obs_data_set_default_double(settings, "masking_dynamic_expansion_base", 10.0);
	obs_data_set_default_double(settings, "masking_dynamic_expansion_ratio", 0.2);
	obs_data_set_default_int(settings, "dilation_iterations", 0);
	obs_data_set_default_bool(settings, "tracking_group", false);
	obs_data_set_default_double(settings, "zoom_factor", 0.0);
	obs_data_set_default_double(settings, "zoom_speed_factor", 0.05);
	obs_data_set_default_string(settings, "zoom_object", "single");
	obs_data_set_default_string(settings, "save_detections_path", "");
	obs_data_set_default_bool(settings, "crop_group", false);
	obs_data_set_default_int(settings, "crop_left", 0);
	obs_data_set_default_int(settings, "crop_right", 0);
	obs_data_set_default_int(settings, "crop_top", 0);
	obs_data_set_default_int(settings, "crop_bottom", 0);
	obs_data_set_default_double(settings, "ghost_recovery_multiplier", 2.0);
	obs_data_set_default_int(settings, "ghost_recovery_max_unseen", 3);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->syncMode = obs_data_get_bool(settings, "sync_mode");
	tf->debugMode = obs_data_get_bool(settings, "debug_mode");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->enableFaceExclusion = obs_data_get_bool(settings, "enable_face_exclusion");
	const char *ref_path = obs_data_get_string(settings, "reference_face_path");
	std::string newRefPath = (ref_path && strlen(ref_path) > 0) ? ref_path : "";
	bool refPathChanged = (tf->referenceFacePath != newRefPath);
	tf->referenceFacePath = newRefPath;
	tf->faceMatchThreshold = (float)obs_data_get_double(settings, "face_match_threshold");
	tf->personCategory = (int)obs_data_get_int(settings, "person_category");
	tf->minFaceAreaRatio = (float)obs_data_get_double(settings, "min_face_area_ratio");
	tf->faceInferenceInterval = (int)obs_data_get_int(settings, "face_inference_interval");
	tf->maskingEnabled = obs_data_get_bool(settings, "masking_group");
	tf->maskingType = obs_data_get_string(settings, "masking_type");
	tf->maskingColor = (int)obs_data_get_int(settings, "masking_color");
	tf->maskingBlurRadius = (int)obs_data_get_int(settings, "masking_blur_radius");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "masking_dilate_iterations");
	tf->maskingDynamicExpansion = obs_data_get_bool(settings, "masking_dynamic_expansion");
	tf->maskingDynamicExpansionBase = (float)obs_data_get_double(settings, "masking_dynamic_expansion_base");
	tf->maskingDynamicExpansionRatio = (float)obs_data_get_double(settings, "masking_dynamic_expansion_ratio");
	bool newTrackingEnabled = obs_data_get_bool(settings, "tracking_group");
	tf->zoomFactor = (float)obs_data_get_double(settings, "zoom_factor");
	tf->zoomSpeedFactor = (float)obs_data_get_double(settings, "zoom_speed_factor");
	tf->zoomObject = obs_data_get_string(settings, "zoom_object");
	tf->sortTracking = obs_data_get_bool(settings, "sort_tracking");
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	tf->showUnseenObjects = obs_data_get_bool(settings, "show_unseen_objects");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");
	tf->minAreaThreshold = (int)obs_data_get_int(settings, "min_size_threshold");
	tf->maxExemptPersons = (int)obs_data_get_int(settings, "max_exempt_persons");
	tf->minHitFrames = (int)obs_data_get_int(settings, "min_hit_frames");
	tf->enableFaceStats = obs_data_get_bool(settings, "enable_face_stats");
	tf->enableFaceStatsLog = obs_data_get_bool(settings, "enable_face_stats_log");
	tf->faceStatsLogPath = obs_data_get_string(settings, "face_stats_log_path");
	tf->showYuNetDetections = obs_data_get_bool(settings, "show_yunet_detections");

	if (tf->tracker.getMinHitFrames() != tf->minHitFrames) {
		tf->tracker.setMinHitFrames(tf->minHitFrames);
	}
	float iou_threshold = (float)obs_data_get_double(settings, "iou_threshold");
	if (tf->tracker.getIouThreshold() != iou_threshold) {
		tf->tracker.setIouThreshold(iou_threshold);
	}
	float instant_track_area_ratio = (float)obs_data_get_double(settings, "instant_track_area_ratio");
	if (tf->tracker.getInstantTrackAreaRatio() != instant_track_area_ratio) {
		tf->tracker.setInstantTrackAreaRatio(instant_track_area_ratio);
	}
	float ghost_recovery_multiplier = (float)obs_data_get_double(settings, "ghost_recovery_multiplier");
	if (tf->tracker.getGhostRecoveryMultiplier() != ghost_recovery_multiplier) {
		tf->tracker.setGhostRecoveryMultiplier(ghost_recovery_multiplier);
	}
	int ghost_recovery_max_unseen = (int)obs_data_get_int(settings, "ghost_recovery_max_unseen");
	if (tf->tracker.getGhostRecoveryMaxUnseen() != ghost_recovery_max_unseen) {
		tf->tracker.setGhostRecoveryMaxUnseen(ghost_recovery_max_unseen);
	}

	// check if tracking state has changed
	if (tf->trackingEnabled != newTrackingEnabled) {
		tf->trackingEnabled = newTrackingEnabled;
		obs_source_t *parent = obs_filter_get_parent(tf->source);
		if (!parent) {
			obs_log(LOG_ERROR, "Parent source not found");
			return;
		}
		if (tf->trackingEnabled) {
			obs_log(LOG_DEBUG, "Tracking enabled");
			// get the parent of the source
			// check if it has a crop/pad filter
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (!crop_pad_filter) {
				// create a crop-pad filter
				crop_pad_filter = obs_source_create(
					"crop_filter", "Detect Tracking", nullptr, nullptr);
				// add a crop/pad filter to the source
				// set the parent of the crop/pad filter to the parent of the source
				obs_source_filter_add(parent, crop_pad_filter);
			}
			tf->trackingFilter = crop_pad_filter;
		} else {
			obs_log(LOG_DEBUG, "Tracking disabled");
			// remove the crop/pad filter
			obs_source_t *crop_pad_filter =
				obs_source_get_filter_by_name(parent, "Detect Tracking");
			if (crop_pad_filter) {
				obs_source_filter_remove(parent, crop_pad_filter);
			}
			tf->trackingFilter = nullptr;
		}
	}

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		obs_log(LOG_INFO, "Reinitializing model");
		reinitialize = true;

		// lock modelMutex
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;
		if (newModelSize == "small") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_256x416.onnx");
		} else if (newModelSize == "medium") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_480x800.onnx");
		} else if (newModelSize == "large") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_736x1280.onnx");
		} else if (newModelSize == FACE_YUNET_MODEL_SIZE) {
			modelFilepath_rawPtr = obs_module_file("models/face_detection_yunet_2023mar.onnx");
		} else if (newModelSize == FACE_YOLO_N_MODEL_SIZE) {
			modelFilepath_rawPtr = obs_module_file("models/yolov8n-face.onnx");
		} else if (newModelSize == FACE_YOLO_S_MODEL_SIZE) {
			modelFilepath_rawPtr = obs_module_file("models/yolov8s-face.onnx");
		} else if (newModelSize == EXTERNAL_MODEL_SIZE) {
			const char *external_model_file =
				obs_data_get_string(settings, "external_model_file");
			if (external_model_file == nullptr || external_model_file[0] == '\0' ||
			    strlen(external_model_file) == 0) {
				obs_log(LOG_ERROR, "External model file path is empty");
				tf->isDisabled = true;
				return;
			}
			modelFilepath_rawPtr = bstrdup(external_model_file);
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
						    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		// Re-initialize model if it's not already the selected one or switching inference device
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		// parameters
		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;
		float nms_th_ = 0.45f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;

		// If this is an external model - look for the config JSON file
		if (tf->modelSize == EXTERNAL_MODEL_SIZE) {
#ifdef _WIN32
			std::wstring labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(L".onnx"), 5, L".json");
#else
			std::string labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(".onnx"), 5, ".json");
#endif
			std::ifstream labelsFile(labelsFilepath);
			if (labelsFile.is_open()) {
				// Parse the JSON file
				nlohmann::json j;
				labelsFile >> j;
				if (j.contains("names")) {
					std::vector<std::string> labels = j["names"];
					num_classes_ = (int)labels.size();
					tf->classNames = labels;
				} else {
					obs_log(LOG_ERROR,
						"JSON file does not contain 'labels' field");
					tf->isDisabled = true;
					tf->onnxruntimemodel.reset();
					return;
				}
			} else {
				obs_log(LOG_ERROR, "Failed to open JSON file: %s",
					labelsFilepath.c_str());
				tf->isDisabled = true;
				tf->onnxruntimemodel.reset();
				return;
			}
		} else if (tf->modelSize == FACE_YUNET_MODEL_SIZE ||
			   tf->modelSize == FACE_YOLO_N_MODEL_SIZE ||
			   tf->modelSize == FACE_YOLO_S_MODEL_SIZE) {
			num_classes_ = 1;
			tf->classNames = yunet::FACE_CLASSES;
		}

		// Load model
		try {
			if (tf->onnxruntimemodel) {
				tf->onnxruntimemodel.reset();
			}
			if (tf->modelSize == FACE_YUNET_MODEL_SIZE) {
				tf->onnxruntimemodel = std::make_unique<yunet::YuNetONNX>(
					tf->modelFilepath, tf->numThreads, 50, tf->numThreads,
					tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
			} else if (tf->modelSize == FACE_YOLO_N_MODEL_SIZE ||
				   tf->modelSize == FACE_YOLO_S_MODEL_SIZE) {
				tf->onnxruntimemodel = std::make_unique<YOLOv8FaceONNX>(
					tf->modelFilepath, tf->numThreads, tf->numThreads,
					tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
			} else {
				tf->onnxruntimemodel =
					std::make_unique<edgeyolo_cpp::EdgeYOLOONNXRuntime>(
						tf->modelFilepath, tf->numThreads, num_classes_,
						tf->numThreads, tf->useGPU, onnxruntime_device_id_,
						onnxruntime_use_parallel_, nms_th_,
						tf->conf_threshold);
			}
			// clear error message
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());
			// disable filter
			tf->isDisabled = true;
			tf->onnxruntimemodel.reset();
			return;
		}
	}

	// Load face recognition models if enabled
	if (tf->enableFaceExclusion) {
		try {
			char *yunet_raw = obs_module_file("models/face_detection_yunet_2023mar.onnx");
			char *sface_raw = obs_module_file("models/face_recognition_sface_2021dec.onnx");
			
			if (yunet_raw && sface_raw) {
#ifdef _WIN32
				int y_len = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, yunet_raw, -1, nullptr, 0);
				std::wstring y_wstr(y_len, L'\0');
				MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, yunet_raw, -1, y_wstr.data(), y_len);
				
				int s_len = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, sface_raw, -1, nullptr, 0);
				std::wstring s_wstr(s_len, L'\0');
				MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, sface_raw, -1, s_wstr.data(), s_len);
				
				std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
				tf->yunetModel = std::make_shared<yunet::YuNetONNX>(y_wstr, 1, 50, 1, "cpu", 0, false, 0.45f, 0.5f);
				tf->sfaceModel = std::make_shared<sface::SFaceONNX>(s_wstr, 1, 1, "cpu", 0, false);
#else
				std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
				tf->yunetModel = std::make_shared<yunet::YuNetONNX>(std::string(yunet_raw), 1, 50, 1, "cpu", 0, false, 0.45f, 0.5f);
				tf->sfaceModel = std::make_shared<sface::SFaceONNX>(std::string(sface_raw), 1, 1, "cpu", 0, false);
#endif
			}
			if (yunet_raw) bfree(yunet_raw);
			if (sface_raw) bfree(sface_raw);

			// Pre-calculate reference face features
			if (!tf->referenceFacePath.empty() && tf->yunetModel && tf->sfaceModel) {
				if (reinitialize || refPathChanged || tf->referenceFaceFeatures.empty()) {
					tf->referenceFaceFeatures.clear();
#ifdef _WIN32
				int p_len = MultiByteToWideChar(CP_UTF8, 0, tf->referenceFacePath.c_str(), -1, nullptr, 0);
				std::wstring p_wstr(p_len, L'\0');
				MultiByteToWideChar(CP_UTF8, 0, tf->referenceFacePath.c_str(), -1, p_wstr.data(), p_len);
				if (p_len > 0 && p_wstr.back() == L'\0') p_wstr.pop_back();
				std::filesystem::path refPath(p_wstr);
#else
				std::filesystem::path refPath(tf->referenceFacePath);
#endif
				std::vector<std::filesystem::path> filesToProcess;
				try {
					if (std::filesystem::is_directory(refPath)) {
						for (const auto& entry : std::filesystem::directory_iterator(refPath)) {
							if (entry.is_regular_file()) {
								filesToProcess.push_back(entry.path());
							}
						}
					} else if (std::filesystem::is_regular_file(refPath)) {
						filesToProcess.push_back(refPath);
					}
				} catch (const std::exception& e) {
					obs_log(LOG_ERROR, "Failed to read reference face path: %s", e.what());
				}
				
				int successCount = 0;
				for (const auto& filePath : filesToProcess) {
#ifdef _WIN32
					FILE* f = _wfopen(filePath.wstring().c_str(), L"rb");
#else
					FILE* f = fopen(filePath.string().c_str(), "rb");
#endif
					cv::Mat refImg;
					if (f) {
						int width, height, channels;
						unsigned char *data = stbi_load_from_file(f, &width, &height, &channels, 3);
						fclose(f);
						if (data) {
							cv::Mat img(height, width, CV_8UC3, data);
							cv::cvtColor(img, refImg, cv::COLOR_RGB2BGR);
							stbi_image_free(data);
						}
					}
					
					if (!refImg.empty()) {
						std::vector<Object> faces = tf->yunetModel->inference(refImg);
						if (!faces.empty()) {
							std::vector<float> feat = tf->sfaceModel->inference(refImg, faces[0].landmarks);
							tf->referenceFaceFeatures.push_back(feat);
							successCount++;
						}
					}
				}
				
				if (successCount > 0) {
					obs_log(LOG_INFO, "Reference faces loaded successfully: %d images", successCount);
				} else {
					obs_log(LOG_WARNING, "No valid faces detected in the reference face path: %s", tf->referenceFacePath.c_str());
				}
				}
			} else {
				tf->referenceFaceFeatures.clear();
			}
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load face recognition models: %s", e.what());
			std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
			tf->yunetModel.reset();
			tf->sfaceModel.reset();
			tf->referenceFaceFeatures.clear();
		}
	} else {
		std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
		tf->yunetModel.reset();
		tf->sfaceModel.reset();
		tf->referenceFaceFeatures.clear();
		tf->faceStatusCache.clear();
		tf->faceSimilarityCache.clear();
		tf->faceExemptIds.clear();
		{
			std::lock_guard<std::mutex> d_lock(tf->debugFaceMutex);
			tf->debugFaceBoxes.clear();
		}
	}

	// update threshold on edgeyolo
	if (tf->onnxruntimemodel) {
		tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		// Log the currently selected options
		obs_log(LOG_INFO, "Detect Filter Options:");
		// name of the source that the filter is attached to
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
		obs_log(LOG_INFO, "  Object Category: %s",
			obs_data_get_string(settings, "object_category"));
		obs_log(LOG_INFO, "  Masking Enabled: %s",
			obs_data_get_bool(settings, "masking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Masking Type: %s",
			obs_data_get_string(settings, "masking_type"));
		obs_log(LOG_INFO, "  Masking Color: %s",
			obs_data_get_string(settings, "masking_color"));
		obs_log(LOG_INFO, "  Masking Blur Radius: %d",
			obs_data_get_int(settings, "masking_blur_radius"));
		obs_log(LOG_INFO, "  Tracking Enabled: %s",
			obs_data_get_bool(settings, "tracking_group") ? "true" : "false");
		obs_log(LOG_INFO, "  Zoom Factor: %.2f",
			obs_data_get_double(settings, "zoom_factor"));
		obs_log(LOG_INFO, "  Zoom Object: %s",
			obs_data_get_string(settings, "zoom_object"));
		obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	// enable
	tf->isDisabled = false;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER CORE                     */

static void run_model_inference(struct detect_filter *tf, cv::Mat imageBGRA)
{
	static auto last_loop_time = std::chrono::high_resolution_clock::now();
	if (imageBGRA.empty() || !tf->onnxruntimemodel) {
		tf->isInferencing = false;
		return;
	}
	tf->frameCount++;
		auto start_time = std::chrono::high_resolution_clock::now();

		cv::Mat inferenceFrame;

		cv::Rect cropRect(0, 0, imageBGRA.cols, imageBGRA.rows);
		if (tf->crop_enabled) {
			cropRect = cv::Rect(tf->crop_left, tf->crop_top,
					    imageBGRA.cols - tf->crop_left - tf->crop_right,
					    imageBGRA.rows - tf->crop_top - tf->crop_bottom);
			cv::cvtColor(imageBGRA(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
		} else {
			cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
		}

	std::vector<Object> objects;

	try {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		objects = tf->onnxruntimemodel->inference(inferenceFrame);
	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}

	if (tf->crop_enabled) {
		// translate the detected objects to the original frame
		for (Object &obj : objects) {
			obj.rect.x += (float)cropRect.x;
			obj.rect.y += (float)cropRect.y;
			for (int n = 0; n < 5; n++) {
				obj.landmarks[n].x += (float)cropRect.x;
				obj.landmarks[n].y += (float)cropRect.y;
			}
		}
	}

	// update the detected object text input
	if (objects.size() > 0) {
		if (tf->lastDetectedObjectId != objects[0].label) {
			tf->lastDetectedObjectId = objects[0].label;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object",
					    tf->classNames[objects[0].label].c_str());
			// release the source settings
			obs_data_release(source_settings);
		}
	} else {
		if (tf->lastDetectedObjectId != -1) {
			tf->lastDetectedObjectId = -1;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object", "");
			// release the source settings
			obs_data_release(source_settings);
		}
	}

	if (tf->minAreaThreshold > 0) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.rect.area() > (float)tf->minAreaThreshold) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}



	if (tf->objectCategory != -1) {
		std::vector<Object> filtered_objects;
		for (const Object &obj : objects) {
			if (obj.label == tf->objectCategory) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	if (tf->sortTracking) {
		float screenArea = (float)(imageBGRA.cols * imageBGRA.rows);
		tf->tracker.setScreenArea(screenArea);
		objects = tf->tracker.update(objects);
	}

	std::vector<Object> all_objects; // for preview

	if (tf->enableFaceExclusion && tf->sortTracking && tf->yunetModel && tf->sfaceModel && !tf->referenceFaceFeatures.empty()) {
		std::unordered_set<uint64_t> current_ids;
		float screenArea = (float)(imageBGRA.cols * imageBGRA.rows);
		float minPersonAreaPixels = screenArea * (tf->minFaceAreaRatio / 100.0f); // use as min person size to save perf

		int current_is_me_count = 0;
		for (Object &obj : objects) {
			current_ids.insert(obj.id);
			if (tf->faceStatusCache.count(obj.id) && tf->faceStatusCache[obj.id] == filter_data::FaceStatus::IS_ME) {
				current_is_me_count++;
			}
		}

		std::vector<Object> to_check;

		for (Object &obj : objects) {
			if (tf->personCategory != -1 && obj.label != tf->personCategory) continue;

			// Check Cache
			if (tf->faceStatusCache.count(obj.id) && tf->faceStatusCache[obj.id] == filter_data::FaceStatus::IS_ME) {
				tf->faceExemptIds.insert(obj.id);
				continue; // Skip YuNet if already resolved as IS_ME
			}
			if (tf->faceStatusCache.count(obj.id) && tf->faceStatusCache[obj.id] == filter_data::FaceStatus::CHECKING) {
				continue;
			}

			if (current_is_me_count >= tf->maxExemptPersons) {
				continue; // Max exempt persons reached, skip inference
			}

			// Throttle Check
			if (tf->frameCount % tf->faceInferenceInterval == 0) {
				if (obj.rect.area() < minPersonAreaPixels) {
					// obs_log(LOG_INFO, "Face check skipped for obj %llu: area %.0f < min %.0f", (unsigned long long)obj.id, obj.rect.area(), minPersonAreaPixels);
				} else {
					tf->faceStatusCache[obj.id] = filter_data::FaceStatus::CHECKING;
					to_check.push_back(obj);
				}
			} else if (!tf->faceStatusCache.count(obj.id)) {
				tf->faceStatusCache[obj.id] = filter_data::FaceStatus::UNKNOWN;
			}
		}

		if (!to_check.empty()) {
			std::unique_lock<std::mutex> lock(tf->faceInferenceMutex);
			tf->faceInferenceFrame = imageBGRA.clone();
			tf->faceInferenceQueue = to_check;
			tf->faceInferenceCV.notify_one();
		}
		
		// Cleanup lost tracks from cache and faceExemptIds
		for (auto it = tf->faceExemptIds.begin(); it != tf->faceExemptIds.end();) {
			if (!current_ids.count(*it)) {
				it = tf->faceExemptIds.erase(it);
			} else {
				++it;
			}
		}
		for (auto it = tf->faceStatusCache.begin(); it != tf->faceStatusCache.end();) {
			if (!current_ids.count(it->first)) {
				it = tf->faceStatusCache.erase(it);
			} else {
				++it;
			}
		}
	}

	if (tf->enableFaceExclusion) {
		std::vector<Object> filtered_objects;
		
		for (Object &obj : objects) {
			if (tf->faceExemptIds.count(obj.id)) {
				obj.isExempt = true;
			}

			if (tf->sortTracking && obj.hitFrames < tf->minHitFrames) {
				obj.isUnconfirmed = true;
			}
			all_objects.push_back(obj);
			if (!obj.isUnconfirmed && !obj.isExempt) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	} else {
		std::vector<Object> filtered_objects;
		for (Object &obj : objects) {
			if (tf->sortTracking && obj.hitFrames < tf->minHitFrames) {
				obj.isUnconfirmed = true;
			}
			all_objects.push_back(obj);
			if (!obj.isUnconfirmed) {
				filtered_objects.push_back(obj);
			}
		}
		objects = filtered_objects;
	}

	if (!tf->showUnseenObjects) {
		objects.erase(
			std::remove_if(objects.begin(), objects.end(),
				       [](const Object &obj) { return obj.unseenFrames > 0; }),
			objects.end());
		all_objects.erase(
			std::remove_if(all_objects.begin(), all_objects.end(),
				       [](const Object &obj) { return obj.unseenFrames > 0; }),
			all_objects.end());
	}

	if (!tf->saveDetectionsPath.empty()) {
		std::string savePath = tf->saveDetectionsPath;
		nlohmann::json j;
		for (const Object &obj : objects) {
			nlohmann::json obj_json;
			obj_json["label"] = obj.label;
			obj_json["confidence"] = obj.prob;
			obj_json["rect"] = {{"x", obj.rect.x},
					    {"y", obj.rect.y},
					    {"width", obj.rect.width},
					    {"height", obj.rect.height}};
			obj_json["id"] = obj.id;
			j.push_back(obj_json);
		}
		std::string jsonStr = j.dump(4);
		
		std::thread([savePath, jsonStr]() {
			std::ofstream detectionsFile(savePath);
			if (detectionsFile.is_open()) {
				detectionsFile << jsonStr;
				detectionsFile.close();
			} else {
				obs_log(LOG_ERROR, "Failed to open file for writing detections: %s",
					savePath.c_str());
			}
		}).detach();
	}

	if (tf->preview || tf->maskingEnabled || tf->debugMode) {
		cv::Mat frame;
		cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);

		if (tf->preview && tf->crop_enabled) {
			// draw the crop rectangle on the frame in a dashed line
			drawDashedRectangle(frame, cropRect, cv::Scalar(0, 255, 0), 5, 8, 15);
		}
		if (tf->preview && all_objects.size() > 0) {
			if (tf->enableFaceExclusion && tf->minFaceAreaRatio > 0.0f) {
				float screenArea = (float)(imageBGRA.cols * imageBGRA.rows);
				float minPersonAreaPixels = screenArea * (tf->minFaceAreaRatio / 100.0f);
				for (Object &obj : all_objects) {
					if (tf->personCategory != -1 && obj.label != tf->personCategory) continue;
					
					if (obj.rect.area() < minPersonAreaPixels) {
						obj.customText = "Too Small for Face Check";
					} else {
						float sim = 0.0f;
						bool has_sim = false;
						{
							std::lock_guard<std::mutex> lock(tf->outputLock);
							if (tf->faceSimilarityCache.count(obj.id)) {
								sim = tf->faceSimilarityCache[obj.id];
								has_sim = true;
							}
						}
						if (has_sim) {
							char buf[64];
							snprintf(buf, sizeof(buf), "Sim: %.2f", sim);
							if (obj.isExempt) {
								obj.customText = std::string("Exempted / ") + buf;
							} else if (sim < tf->faceMatchThreshold) {
								obj.customText = std::string("Not Me / ") + buf;
							} else {
								obj.customText = std::string("Checking / ") + buf;
							}
						} else {
							obj.customText = "Checking Face...";
						}
					}
				}
			}
		}
		
		cv::Mat overlay = cv::Mat::zeros(frame.size(), CV_8UC4);

		if (tf->preview) {
			draw_objects(overlay, all_objects, tf->classNames);
		}
		if (tf->maskingEnabled) {
			cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
			for (const Object &obj : objects) {
				cv::Rect2f rect = obj.rect;
				float expansion = 0.0f;
				if (tf->maskingDynamicExpansion) {
					expansion = tf->maskingDynamicExpansionBase + (rect.height * tf->maskingDynamicExpansionRatio);
				} else if (tf->maskingDilateIterations > 0) {
					expansion = (float)tf->maskingDilateIterations;
				}

				if (expansion > 0.0f) {
					rect.x -= expansion;
					rect.y -= expansion;
					rect.width += expansion * 2.0f;
					rect.height += expansion * 2.0f;
				}
				cv::rectangle(mask, rect, cv::Scalar(255), -1);
			}
			std::lock_guard<std::mutex> lock(tf->outputLock);
			mask.copyTo(tf->outputMask);
		}

		if (tf->debugMode) {
			char debugText[128];
			snprintf(debugText, sizeof(debugText), "FPS: %.1f | Latency: %.1fms", tf->currentInferenceFPS, tf->currentInferenceTimeMs);
			cv::putText(overlay, debugText, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255, 255), 3, cv::LINE_AA);
		}

		if (tf->showYuNetDetections) {
			auto now = std::chrono::steady_clock::now();
			std::lock_guard<std::mutex> d_lock(tf->debugFaceMutex);
			auto it = tf->debugFaceBoxes.begin();
			while (it != tf->debugFaceBoxes.end()) {
				double elapsed = std::chrono::duration<double>(now - it->timestamp).count();
				if (elapsed > 1.0) {
					it = tf->debugFaceBoxes.erase(it);
				} else {
					int alpha = (int)((1.0 - elapsed) * 200.0);
					if (alpha < 0) alpha = 0;
					cv::rectangle(overlay, it->rect, cv::Scalar(255, 0, 0, alpha), 1, cv::LINE_AA);
					
					char lbl[64];
					snprintf(lbl, sizeof(lbl), "YuNet (%.1fs)", 1.0 - elapsed);
					cv::putText(overlay, lbl, cv::Point((int)it->rect.x, (int)it->rect.y - 5), 
					            cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 0, alpha), 1, cv::LINE_AA);
					++it;
				}
			}
		}

		if (tf->enableFaceStats) {
			int tChecks, tIsMe, tNotMe;
			float tSumSim;
			int hist[10];
			{
				std::lock_guard<std::mutex> statLock(tf->statsMutex);
				tChecks = tf->statTotalChecks;
				tIsMe = tf->statIsMe;
				tNotMe = tf->statNotMe;
				tSumSim = tf->statSumSimilarity;
				for (int i=0; i<10; i++) hist[i] = tf->similarityHistogram[i];
			}
			
			int panelWidth = 300;
			int panelHeight = 200;
			int padding = 20;
			int startX = frame.cols - panelWidth - padding;
			int startY = padding;
			
			cv::Mat stats_overlay = cv::Mat::zeros(frame.size(), CV_8UC4);
			cv::rectangle(stats_overlay, cv::Rect(startX, startY, panelWidth, panelHeight), cv::Scalar(0, 0, 0, 180), -1);
			cv::addWeighted(overlay, 1.0, stats_overlay, 1.0, 0, overlay);
			
			char txt[128];
			snprintf(txt, sizeof(txt), "Face Stats (Total: %d)", tChecks);
			cv::putText(overlay, txt, cv::Point(startX + 10, startY + 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,255), 1);
			
			snprintf(txt, sizeof(txt), "IS_ME: %d | NOT_ME: %d", tIsMe, tNotMe);
			cv::putText(overlay, txt, cv::Point(startX + 10, startY + 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0,255), 1);
			
			float avgSim = (tChecks > 0) ? (tSumSim / tChecks) : 0.0f;
			snprintf(txt, sizeof(txt), "Avg Similarity: %.3f", avgSim);
			cv::putText(overlay, txt, cv::Point(startX + 10, startY + 65), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255,255), 1);
			
			int histX = startX + 10;
			int histY = startY + 180;
			int histW = panelWidth - 20;
			int histH = 100;
			int barW = histW / 10;
			
			int maxCount = 1;
			for (int i=0; i<10; i++) {
				if (hist[i] > maxCount) maxCount = hist[i];
			}
			
			for (int i=0; i<10; i++) {
				int barHeight = (int)(((float)hist[i] / maxCount) * histH);
				cv::Rect barRect(histX + i*barW, histY - barHeight, barW - 2, barHeight);
				cv::Scalar barColor = (i >= (int)(tf->faceMatchThreshold * 10.0f)) ? cv::Scalar(0,200,0,255) : cv::Scalar(0,0,200,255);
				cv::rectangle(overlay, barRect, barColor, -1);
				
				snprintf(txt, sizeof(txt), ".%d", i);
				cv::putText(overlay, txt, cv::Point(histX + i*barW + 2, histY + 12), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200,200,200,255), 1);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		tf->outputPreviewBGRA = overlay.clone();
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	tf->currentInferenceTimeMs = std::chrono::duration<float, std::milli>(end_time - start_time).count();
	
	float loopTimeMs = std::chrono::duration<float, std::milli>(end_time - last_loop_time).count();
	last_loop_time = end_time;
	
	if (loopTimeMs > 0) {
		tf->currentInferenceFPS = 1000.0f / loopTimeMs;
	} else {
		tf->currentInferenceFPS = 0.0f;
	}

	{
		std::lock_guard<std::mutex> lock(tf->outputLock);
		tf->latestObjects = objects;
	}
	tf->isInferencing = false;
}
static void inference_thread_loop(struct detect_filter *tf);
static void face_inference_thread_loop(struct detect_filter *tf);

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;
	tf->frameCount = 0;

	std::vector<std::tuple<const char *, gs_effect_t **>> effects = {
		{KAWASE_BLUR_EFFECT_PATH, &tf->kawaseBlurEffect},
		{MASKING_EFFECT_PATH, &tf->maskingEffect},
		{PIXELATE_EFFECT_PATH, &tf->pixelateEffect},
	};

	for (auto [effectPath, effect] : effects) {
		char *effectPathPtr = obs_module_file(effectPath);
		if (!effectPathPtr) {
			obs_log(LOG_ERROR, "Failed to get effect path: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_enter_graphics();
		*effect = gs_effect_create_from_file(effectPathPtr, nullptr);
		bfree(effectPathPtr);
		if (!*effect) {
			obs_log(LOG_ERROR, "Failed to load effect: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
		obs_leave_graphics();
	}

	detect_filter_update(tf, settings);

	tf->stopInferenceThread = false;
	tf->isInferencing = false;
	tf->inferenceFrameReady = false;
	tf->stopFaceInferenceThread = false;

	tf->inferenceThread = std::thread(inference_thread_loop, tf);
	tf->faceInferenceThread = std::thread(face_inference_thread_loop, tf);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		tf->stopInferenceThread = true;
		tf->inferenceCV.notify_all();
		if (tf->inferenceThread.joinable()) {
			tf->inferenceThread.join();
		}
		
		tf->stopFaceInferenceThread = true;
		tf->faceInferenceCV.notify_all();
		if (tf->faceInferenceThread.joinable()) {
			tf->faceInferenceThread.join();
		}

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		if (tf->previewTexture) {
			gs_texture_destroy(tf->previewTexture);
		}
		if (tf->renderMaskTexture) {
			gs_texture_destroy(tf->renderMaskTexture);
		}
		gs_effect_destroy(tf->kawaseBlurEffect);
		gs_effect_destroy(tf->maskingEffect);
		obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

static void inference_thread_loop(struct detect_filter *tf)
{
	obs_log(LOG_INFO, "Inference thread started");
	while (!tf->stopInferenceThread) {
		cv::Mat imageBGRA;
		{
			std::unique_lock<std::mutex> lock(tf->inferenceMutex);
			tf->inferenceCV.wait(lock, [tf] {
				return tf->inferenceFrameReady || tf->stopInferenceThread;
			});

			if (tf->stopInferenceThread) {
				break;
			}

			imageBGRA = tf->inferenceInputFrame.clone();
			tf->inferenceFrameReady = false;
			tf->isInferencing = true;
		}

		run_model_inference(tf, imageBGRA);
	}
	obs_log(LOG_INFO, "Inference thread stopped");
}

static void face_inference_thread_loop(struct detect_filter *tf)
{
	obs_log(LOG_INFO, "Face Inference thread started");
	while (!tf->stopFaceInferenceThread) {
		std::vector<Object> to_check;
		cv::Mat imageBGRA;
		std::shared_ptr<yunet::YuNetONNX> local_yunet;
		std::shared_ptr<sface::SFaceONNX> local_sface;
		
		{
			std::unique_lock<std::mutex> lock(tf->faceInferenceMutex);
			tf->faceInferenceCV.wait(lock, [tf] {
				return !tf->faceInferenceQueue.empty() || tf->stopFaceInferenceThread;
			});

			if (tf->stopFaceInferenceThread) {
				break;
			}
			
			to_check = std::move(tf->faceInferenceQueue);
			tf->faceInferenceQueue.clear();
			imageBGRA = tf->faceInferenceFrame.clone();
			local_yunet = tf->yunetModel;
			local_sface = tf->sfaceModel;
		}

		if (imageBGRA.empty() || !local_yunet || !local_sface) {
			continue;
		}

		for (const Object& obj : to_check) {
			if (tf->stopFaceInferenceThread) break;

			bool is_already_face = (obj.label < tf->classNames.size() && 
			                        (tf->classNames[obj.label] == "face" || tf->classNames[obj.label] == "head"));
			bool has_valid_landmarks = false;
			if (is_already_face) {
				if (std::abs(obj.landmarks[0].x - obj.landmarks[1].x) > 1e-3 || 
				    std::abs(obj.landmarks[0].y - obj.landmarks[1].y) > 1e-3) {
					has_valid_landmarks = true;
				}
			}
			
			cv::Rect_<float> crop_box;
			if (is_already_face && has_valid_landmarks) {
				// SFace can align the face directly from the full BGR frame using absolute landmarks
				crop_box = cv::Rect_<float>(0, 0, (float)imageBGRA.cols, (float)imageBGRA.rows);
			} else {
				// Use the full person box or expand the face box slightly to allow YuNet to detect it
				crop_box = obj.rect;
				if (is_already_face && !has_valid_landmarks) {
					float margin_w = crop_box.width * 0.2f;
					float margin_h = crop_box.height * 0.2f;
					crop_box.x -= margin_w;
					crop_box.y -= margin_h;
					crop_box.width += margin_w * 2.0f;
					crop_box.height += margin_h * 2.0f;
				}
			}
			cv::Rect crop_rect = crop_box & cv::Rect_<float>(0, 0, (float)imageBGRA.cols, (float)imageBGRA.rows);
			
			if (crop_rect.width > 0 && crop_rect.height > 0) {
				cv::Mat cropped = imageBGRA(crop_rect);
				cv::Mat croppedBGR;
				cv::cvtColor(cropped, croppedBGR, cv::COLOR_BGRA2BGR);
				
				std::vector<Object> faces;
				if (is_already_face && has_valid_landmarks) {
					// We already have the landmarks from the main thread!
					// They are relative to the full frame. We need to shift them to match croppedBGR.
					Object shifted_face = obj;
					for (int l = 0; l < 5; ++l) {
						shifted_face.landmarks[l].x -= crop_rect.x;
						shifted_face.landmarks[l].y -= crop_rect.y;
					}
					shifted_face.rect.x -= crop_rect.x;
					shifted_face.rect.y -= crop_rect.y;
					faces.push_back(shifted_face);
				} else {
					faces = local_yunet->inference(croppedBGR);
				}
				
				bool is_me = false;
				if (!faces.empty()) {
					// Only store debug boxes and sort if we actually ran YuNet in the background
					if (!(is_already_face && has_valid_landmarks)) {
						// Store all detected faces in absolute coordinates for debug overlay
						std::vector<DebugFaceBox> new_debug_boxes;
						for (const auto& face : faces) {
							DebugFaceBox dbg;
							dbg.rect = face.rect;
							dbg.rect.x += crop_rect.x;
							dbg.rect.y += crop_rect.y;
							dbg.timestamp = std::chrono::steady_clock::now();
							new_debug_boxes.push_back(dbg);
						}
						{
							std::lock_guard<std::mutex> d_lock(tf->debugFaceMutex);
							tf->debugFaceBoxes.insert(tf->debugFaceBoxes.end(), new_debug_boxes.begin(), new_debug_boxes.end());
						}

						// Sort faces to select the one closest to the top-center of the person box
						if (faces.size() > 1) {
							float p_cx = (float)crop_rect.width / 2.0f;
							float p_ty = (float)crop_rect.height * 0.15f; // Target face center (15% from the top)
							std::sort(faces.begin(), faces.end(), [p_cx, p_ty](const Object& a, const Object& b) {
								float a_cx = a.rect.x + a.rect.width / 2.0f;
								float a_cy = a.rect.y + a.rect.height / 2.0f;
								float b_cx = b.rect.x + b.rect.width / 2.0f;
								float b_cy = b.rect.y + b.rect.height / 2.0f;
								
								float a_dx = a_cx - p_cx;
								float a_dy = a_cy - p_ty;
								float a_score = (a_dx * a_dx * 2.0f) + (a_dy * a_dy); // Weight centerline closeness higher
								
								float b_dx = b_cx - p_cx;
								float b_dy = b_cy - p_ty;
								float b_score = (b_dx * b_dx * 2.0f) + (b_dy * b_dy);
								
								return a_score < b_score;
							});
						}
					}
					std::vector<float> feat = local_sface->inference(croppedBGR, faces[0].landmarks);
					float max_sim = -1.0f;
					for (const auto& refFeat : tf->referenceFaceFeatures) {
						float sim = sface::SFaceONNX::match(feat, refFeat);
						if (sim > max_sim) max_sim = sim;
					}
					obs_log(LOG_INFO, "Face matched for obj %llu: similarity %.3f (threshold %.3f)", (unsigned long long)obj.id, max_sim, tf->faceMatchThreshold);
					
					{
						std::lock_guard<std::mutex> lock(tf->outputLock);
						tf->faceSimilarityCache[obj.id] = max_sim;
					}

					if (max_sim >= tf->faceMatchThreshold) {
						is_me = true;
					}

					if (tf->enableFaceStats) {
						std::lock_guard<std::mutex> statLock(tf->statsMutex);
						tf->statTotalChecks++;
						tf->statSumSimilarity += max_sim;
						if (is_me) tf->statIsMe++; else tf->statNotMe++;
						int bin = std::max(0, std::min(9, (int)(max_sim * 10.0f)));
						tf->similarityHistogram[bin]++;
					}

					if (!tf->faceStatsLogPath.empty() && tf->enableFaceStatsLog) {
						std::string logPath = tf->faceStatsLogPath;
						float sim_val = max_sim;
						uint64_t oid = obj.id;
						bool is_me_val = is_me;
						std::thread([logPath, sim_val, oid, is_me_val]() {
							std::ofstream f(logPath, std::ios::app);
							if (f.is_open()) {
								auto now = std::chrono::system_clock::now();
								auto in_time_t = std::chrono::system_clock::to_time_t(now);
								char time_buf[100];
								std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&in_time_t));
								f << time_buf << "," << oid << "," << sim_val << "," << (is_me_val ? "IS_ME" : "NOT_ME") << "\n";
							}
						}).detach();
					}
				} else {
					// obs_log(LOG_INFO, "No face detected in cropped region for obj %llu", (unsigned long long)obj.id);
				}
				
				if (is_me) {
					tf->faceStatusCache[obj.id] = filter_data::FaceStatus::IS_ME;
					tf->faceExemptIds.insert(obj.id);
				} else {
					tf->faceStatusCache[obj.id] = filter_data::FaceStatus::NOT_ME;
				}
			} else {
				tf->faceStatusCache[obj.id] = filter_data::FaceStatus::UNKNOWN;
			}
		}
	}
	obs_log(LOG_INFO, "Face Inference thread stopped");
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel || !obs_source_enabled(tf->source)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock() || tf->inputBGRA.empty()) {
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	if (tf->syncMode) {
		run_model_inference(tf, imageBGRA);
	} else {
		std::unique_lock<std::mutex> lock(tf->inferenceMutex, std::try_to_lock);
		if (lock.owns_lock() && !tf->isInferencing) {
			tf->inferenceInputFrame = imageBGRA;
			tf->inferenceFrameReady = true;
			tf->inferenceCV.notify_one();
		}
	}

	if (tf->trackingEnabled && tf->trackingFilter) {
		const int width = imageBGRA.cols;
		const int height = imageBGRA.rows;

		std::vector<Object> objects;
		{
			std::lock_guard<std::mutex> lock(tf->outputLock);
			objects = tf->latestObjects;
		}

		cv::Rect2f boundingBox = cv::Rect2f(0, 0, (float)width, (float)height);
		// get location of the objects
		if (tf->zoomObject == "single") {
			if (objects.size() > 0) {
				// find first visible object
				for (const Object &obj : objects) {
					if (obj.unseenFrames == 0) {
						boundingBox = obj.rect;
						break;
					}
				}
			}
		} else if (tf->zoomObject == "biggest") {
			// get the bounding box of the biggest object
			if (objects.size() > 0) {
				float maxArea = 0;
				for (const Object &obj : objects) {
					const float area = obj.rect.width * obj.rect.height;
					if (area > maxArea) {
						maxArea = area;
						boundingBox = obj.rect;
					}
				}
			}
		} else if (tf->zoomObject == "oldest") {
			// get the object with the oldest id that's visible currently
			if (objects.size() > 0) {
				uint64_t oldestId = UINT64_MAX;
				for (const Object &obj : objects) {
					if (obj.unseenFrames == 0 && obj.id < oldestId) {
						oldestId = obj.id;
						boundingBox = obj.rect;
					}
				}
			}
		} else {
			// get the bounding box of all objects
			if (objects.size() > 0) {
				boundingBox = objects[0].rect;
				for (const Object &obj : objects) {
					if (obj.unseenFrames > 0) {
						continue;
					}
					boundingBox |= obj.rect;
				}
			}
		}
		bool lostTracking = objects.size() == 0;
		// the zooming box should maintain the aspect ratio of the image
		// with the tf->zoomFactor controlling the effective buffer around the bounding box
		// the bounding box is the center of the zooming box
		float frameAspectRatio = (float)width / (float)height;
		// calculate an aspect ratio box around the object using its height
		float boxHeight = boundingBox.height;
		// calculate the zooming box size
		float dh = (float)height - boxHeight;
		float buffer = dh * (1.0f - tf->zoomFactor);
		float zh = boxHeight + buffer;
		float zw = zh * frameAspectRatio;
		// calculate the top left corner of the zooming box
		float zx = boundingBox.x - (zw - boundingBox.width) / 2.0f;
		float zy = boundingBox.y - (zh - boundingBox.height) / 2.0f;

		if (tf->trackingRect.width == 0) {
			// initialize the trackingRect
			tf->trackingRect = cv::Rect2f(zx, zy, zw, zh);
		} else {
			// interpolate the zooming box to tf->trackingRect
			float factor = tf->zoomSpeedFactor * (lostTracking ? 0.2f : 1.0f);
			tf->trackingRect.x =
				tf->trackingRect.x + factor * (zx - tf->trackingRect.x);
			tf->trackingRect.y =
				tf->trackingRect.y + factor * (zy - tf->trackingRect.y);
			tf->trackingRect.width =
				tf->trackingRect.width + factor * (zw - tf->trackingRect.width);
			tf->trackingRect.height =
				tf->trackingRect.height + factor * (zh - tf->trackingRect.height);
		}

		// get the settings of the crop/pad filter
		obs_data_t *crop_pad_settings = obs_source_get_settings(tf->trackingFilter);
		obs_data_set_int(crop_pad_settings, "left", (int)tf->trackingRect.x);
		obs_data_set_int(crop_pad_settings, "top", (int)tf->trackingRect.y);
		// right = image width - (zx + zw)
		obs_data_set_int(
			crop_pad_settings, "right",
			(int)((float)width - (tf->trackingRect.x + tf->trackingRect.width)));
		// bottom = image height - (zy + zh)
		obs_data_set_int(
			crop_pad_settings, "bottom",
			(int)((float)height - (tf->trackingRect.y + tf->trackingRect.height)));
		// apply the settings
		obs_source_update(tf->trackingFilter, crop_pad_settings);
		obs_data_release(crop_pad_settings);
	}
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled || !tf->onnxruntimemodel) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	// if preview, masking, debug stats, or debug mode is enabled, render the image
	if (tf->preview || tf->maskingEnabled || tf->debugMode || tf->enableFaceStats) {
		cv::Mat outputBGRA, outputMask;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "Preview image is empty");
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
			outputBGRA = tf->outputPreviewBGRA.clone();
			outputMask = tf->outputMask.clone();
		}

		gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
		bool destroy_tex = false;
		
		std::string technique_name = "Draw";
		gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
		gs_eparam_t *maskParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "focalmask");
		gs_eparam_t *maskColorParam =
			gs_effect_get_param_by_name(tf->maskingEffect, "color");

		if (tf->maskingEnabled) {
			if (!tf->renderMaskTexture || tf->lastTexWidth != width || tf->lastTexHeight != height) {
				if (tf->renderMaskTexture) gs_texture_destroy(tf->renderMaskTexture);
				tf->renderMaskTexture = gs_texture_create(width, height, GS_R8, 1,
								(const uint8_t **)&outputMask.data, GS_DYNAMIC);
			} else {
				gs_texture_set_image(tf->renderMaskTexture, outputMask.data, width, false);
			}
			gs_effect_set_texture(maskParam, tf->renderMaskTexture);
			if (tf->maskingType == "output_mask") {
				technique_name = "DrawMask";
			} else if (tf->maskingType == "blur") {
				tex = blur_image(tf, width, height, tf->renderMaskTexture);
				destroy_tex = true; // blur_image creates a new texture
			} else if (tf->maskingType == "pixelate") {
				tex = pixelate_image(tf, width, height, tf->renderMaskTexture,
						     (float)tf->maskingBlurRadius);
				destroy_tex = true;
			} else if (tf->maskingType == "transparent") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, 0);
			} else if (tf->maskingType == "solid_color") {
				technique_name = "DrawSolidColor";
				gs_effect_set_color(maskColorParam, tf->maskingColor);
			}
		}

		gs_effect_set_texture(imageParam, tex);

		while (gs_effect_loop(tf->maskingEffect, technique_name.c_str())) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		if (destroy_tex) {
			gs_texture_destroy(tex);
		}
		
		if (tf->preview || tf->debugMode || tf->enableFaceStats) {
			if (!tf->previewTexture || tf->lastTexWidth != width || tf->lastTexHeight != height) {
				if (tf->previewTexture) gs_texture_destroy(tf->previewTexture);
				tf->previewTexture = gs_texture_create(width, height, GS_BGRA, 1,
							      (const uint8_t **)&outputBGRA.data, GS_DYNAMIC);
			} else {
				gs_texture_set_image(tf->previewTexture, outputBGRA.data, width * 4, false);
			}
			
			gs_blend_state_push();
			gs_blend_function(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA);
			gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
			gs_eparam_t *defaultImageParam = gs_effect_get_param_by_name(default_effect, "image");
			gs_effect_set_texture(defaultImageParam, tf->previewTexture);
			while (gs_effect_loop(default_effect, "Draw")) {
				gs_draw_sprite(tf->previewTexture, 0, 0, 0);
			}
			gs_blend_state_pop();
		}
		
		tf->lastTexWidth = width;
		tf->lastTexHeight = height;
	} else {
		// Just render the original video
		gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
		gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
