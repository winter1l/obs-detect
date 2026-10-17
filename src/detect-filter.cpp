#include "detect-filter.h"

#include <obs-frontend-api.h>
#include <QMainWindow>
#include <QMetaObject>

#include <onnxruntime_cxx_api.h>

#include "sface/SFace.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _WIN32
#define NOMINMAX
#include <util/dstr.h>
#include <wchar.h>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
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
#include <limits>
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
#include "FrameUtils.h"
#include "FaceReferenceSelection.h"
#include "InferenceSafety.h"
#include "MaskGeometry.h"
#include "edgeyolo/edgeyolo_onnxruntime.hpp"
#include "ort-model/YOLOv8FaceONNX.hpp"
#include "yunet/YuNet.h"

extern "C" void emit_face_exclusion_update(int count);

#define EXTERNAL_MODEL_SIZE "EXTERNAL_MODEL"
#define FACE_YUNET_MODEL_SIZE "FACE_YUNET"
#define FACE_YOLO_N_MODEL_SIZE "FACE_YOLO_N"
#define FACE_YOLO_S_MODEL_SIZE "FACE_YOLO_S"

struct detect_filter : public filter_data {};

static uint64_t face_now_ms()
{
	return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
		       std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

// Tracks can outlive a model swap, so labels are not guaranteed to be valid indices.
static const char *class_name_or_unknown(const std::vector<std::string> &names, int label)
{
	if (label < 0 || (size_t)label >= names.size())
		return "unknown";
	return names[(size_t)label].c_str();
}

static FaceRetryPolicy face_retry_policy(int intervalAt30Fps)
{
	FaceRetryPolicy policy;
	policy.noFaceRetryMs = 150;
	policy.errorRetryMs = 500;
	policy.mismatchRetryMs =
		(uint64_t)std::max(100, intervalAt30Fps * 1000 / 30);
	return policy;
}

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
		if (!class_name_cap.empty())
			class_name_cap[0] = (char)std::toupper((unsigned char)class_name_cap[0]);
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

// Reads the class names from the '.json' file next to an external model. Never throws:
// it runs in property callbacks and on the IO thread, where an escaping exception
// terminates OBS.
static bool load_class_names_for_model(const std::filesystem::path &modelPath,
				       std::vector<std::string> &names, std::string &error)
{
	try {
		std::filesystem::path jsonPath = modelPath;
		jsonPath.replace_extension(".json");
		std::ifstream file(jsonPath);
		if (!file.is_open()) {
			error = "JSON file not found";
			return false;
		}
		const nlohmann::json j = nlohmann::json::parse(file, nullptr, false);
		if (j.is_discarded() || !j.is_object()) {
			error = "JSON file is not valid";
			return false;
		}
		const auto namesIt = j.find("names");
		if (namesIt == j.end() || !namesIt->is_array() || namesIt->empty()) {
			error = "JSON file does not contain a non-empty 'names' array";
			return false;
		}
		std::vector<std::string> parsed;
		for (const auto &name : *namesIt) {
			if (!name.is_string()) {
				error = "'names' must contain only strings";
				return false;
			}
			parsed.push_back(name.get<std::string>());
		}
		names = std::move(parsed);
		return true;
	} catch (const std::exception &e) {
		error = e.what();
		return false;
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings)
{
	if (model_file == nullptr || model_file[0] == '\0') {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	std::vector<std::string> labels;
	std::string error;
	if (!load_class_names_for_model(std::filesystem::u8path(model_file), labels, error)) {
		obs_data_set_string(settings, "error", error.c_str());
		obs_log(LOG_ERROR, "Failed to read class names for %s: %s", model_file,
			error.c_str());
		return;
	}
	obs_data_set_string(settings, "error", "");
	// Only the dropdowns are updated here; tf->classNames is replaced together with the
	// model by the load task so labels always match the running model.
	set_class_names_on_object_category(obs_properties_get(props_, "object_category"), labels);
	set_class_names_on_object_category(obs_properties_get(props_, "person_category"), labels);
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

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
			UNUSED_PARAMETER(data_);
			std::string model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = model_size_value == EXTERNAL_MODEL_SIZE;
			bool is_face_detect = (model_size_value == FACE_YUNET_MODEL_SIZE ||
					       model_size_value == FACE_YOLO_N_MODEL_SIZE ||
					       model_size_value == FACE_YOLO_S_MODEL_SIZE);
			
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);

			if (!is_external) {
				const std::vector<std::string> &names =
					is_face_detect ? yunet::FACE_CLASSES : edgeyolo_cpp::COCO_CLASSES;
				set_class_names_on_object_category(
					obs_properties_get(props_, "object_category"), names);
				set_class_names_on_object_category(
					obs_properties_get(props_, "person_category"), names);
			} else {
				// if the model path is already set - update the class names
				const char *model_file =
					obs_data_get_string(settings, "external_model_file");
				read_model_config_json_and_set_class_names(model_file, props_,
									   settings);
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
			UNUSED_PARAMETER(data_);
			const char *model_file =
				obs_data_get_string(settings, "external_model_file");
			read_model_config_json_and_set_class_names(model_file, props_, settings);
			return true;
		},
		tf);

	// add dropdown selection for object category selection: "All", or COCO classes
	obs_property_t *object_category =
		obs_properties_add_list(props, "object_category", obs_module_text("ObjectCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(object_category, edgeyolo_cpp::COCO_CLASSES);

	// add threshold slider
	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.10,
					1.0, 0.025);
	obs_properties_add_int_slider(props, "min_size_threshold",
				      obs_module_text("MinSizeThreshold"), 0, 10000, 1);

	// options group for masking
	obs_properties_t *masking_group = obs_properties_create();
	obs_property_t *masking_group_prop =
		obs_properties_add_group(props, "masking_group", obs_module_text("MaskingGroup"),
					 OBS_GROUP_CHECKABLE, masking_group);

	auto update_masking_visibility = [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		std::string masking_type_value = obs_data_get_string(settings, "masking_type");
		obs_property_t *masking_color = obs_properties_get(props_, "masking_color");
		obs_property_t *masking_blur_radius = obs_properties_get(props_, "masking_blur_radius");
		obs_property_t *masking_feather = obs_properties_get(props_, "masking_feather");
		obs_property_t *masking_dilation = obs_properties_get(props_, "masking_dilate_iterations");
		obs_property_t *masking_dynamic_expansion = obs_properties_get(props_, "masking_dynamic_expansion");
		obs_property_t *masking_dynamic_base = obs_properties_get(props_, "masking_dynamic_expansion_base");
		obs_property_t *masking_dynamic_ratio = obs_properties_get(props_, "masking_dynamic_expansion_ratio");

		const bool masking_enabled = obs_data_get_bool(settings, "masking_group");
		const bool dynamic_enabled = obs_data_get_bool(settings, "masking_dynamic_expansion");

		if (masking_color) obs_property_set_visible(masking_color, false);
		if (masking_blur_radius) obs_property_set_visible(masking_blur_radius, false);
		if (masking_feather) obs_property_set_visible(masking_feather, false);

		if (masking_dynamic_expansion) obs_property_set_visible(masking_dynamic_expansion, masking_enabled);
		if (masking_dilation) obs_property_set_visible(masking_dilation, masking_enabled && !dynamic_enabled);
		if (masking_dynamic_base) obs_property_set_visible(masking_dynamic_base, masking_enabled && dynamic_enabled);
		if (masking_dynamic_ratio) obs_property_set_visible(masking_dynamic_ratio, masking_enabled && dynamic_enabled);

		if (masking_type_value == "solid_color") {
			if (masking_color) obs_property_set_visible(masking_color, masking_enabled);
		} else if (masking_type_value == "blur" || masking_type_value == "pixelate") {
			if (masking_blur_radius) obs_property_set_visible(masking_blur_radius, masking_enabled);
		}
		if (masking_feather) obs_property_set_visible(masking_feather, masking_enabled);
		return true;
	};

	// add callback to show/hide masking options
	obs_property_set_modified_callback(masking_group_prop, update_masking_visibility);

	// add masking options drop down selection: "Solid color", "Blur", "Transparent"
	obs_property_t *masking_type = obs_properties_add_list(masking_group, "masking_type",
							       obs_module_text("MaskingType"),
							       OBS_COMBO_TYPE_LIST,
							       OBS_COMBO_FORMAT_STRING);
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
					  
	// add slider for mask feather
	obs_properties_add_int_slider(masking_group, "masking_feather",
				      obs_module_text("MaskingFeather"), 0, 200, 1);



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
					obs_module_text("MaskingDynamicExpansionRatio"), 0.0, 1.0, 0.01);

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

	// SORT Tracking Group (always enabled)
	obs_properties_t *sort_group_props = obs_properties_create();
	obs_properties_add_group(props, "sort_tracking", obs_module_text("SORTTracking"),
				 OBS_GROUP_NORMAL, sort_group_props);

	obs_properties_add_int(sort_group_props, "min_hit_frames", obs_module_text("MinHitFrames"), 1, 10, 1);
	obs_properties_add_float_slider(sort_group_props, "instant_track_area_ratio", obs_module_text("InstantTrackAreaRatio"), 0.0, 100.0, 0.1);
	obs_properties_add_int(sort_group_props, "max_unseen_frames", obs_module_text("MaxUnseenFrames"), 1, 30, 1);
	obs_properties_add_float_slider(sort_group_props, "iou_threshold", obs_module_text("IouThreshold"), 0.01, 1.0, 0.01);
	obs_properties_add_float_slider(sort_group_props, "ghost_recovery_multiplier", obs_module_text("GhostRecoveryMultiplier"), 1.0, 5.0, 0.1);
	obs_properties_add_float_slider(sort_group_props, "ghost_recovery_size_ratio_limit", obs_module_text("GhostRecoverySizeRatioLimit"), 1.1, 3.0, 0.1);

	// Face Exclusion Group
	obs_properties_t *face_ex_group_props = obs_properties_create();
	obs_property_t *enable_face_exclusion =
		obs_properties_add_group(props, "enable_face_exclusion", obs_module_text("EnableFaceExclusion"),
					 OBS_GROUP_CHECKABLE, face_ex_group_props);

	obs_property_t *person_category =
		obs_properties_add_list(face_ex_group_props, "person_category", obs_module_text("PersonCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(person_category, edgeyolo_cpp::COCO_CLASSES);

	obs_property_t *reference_face_path = obs_properties_add_path(
		face_ex_group_props, "reference_face_path", obs_module_text("ReferenceFaceImage"),
		OBS_PATH_DIRECTORY, "", nullptr);
	obs_property_set_long_description(reference_face_path,
					  obs_module_text("ReferenceFaceImage.Description"));
	obs_properties_add_float_slider(face_ex_group_props, "face_match_threshold", obs_module_text("FaceMatchThreshold"), 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(face_ex_group_props, "min_face_area_ratio", obs_module_text("MinFaceAreaRatio"), 0.0, 100.0, 0.1);
	obs_property_t *face_inference_interval = obs_properties_add_int_slider(
		face_ex_group_props, "face_inference_interval",
		obs_module_text("FaceInferenceInterval"), 5, 60, 1);
	obs_property_set_long_description(
		face_inference_interval, obs_module_text("FaceInferenceInterval.Description"));
	obs_properties_add_int_slider(face_ex_group_props, "max_exempt_persons", obs_module_text("MaxExemptPersons"), 0, 10, 1);
	obs_property_t *preserve_face_exclusion =
		obs_properties_add_bool(face_ex_group_props, "preserve_face_exclusion_on_recovery",
					obs_module_text("PreserveFaceExclusionOnRecovery"));
	obs_property_set_long_description(
		preserve_face_exclusion,
		obs_module_text("PreserveFaceExclusionOnRecovery.Description"));
	obs_property_t *track_switch_detection = obs_properties_add_list(
		face_ex_group_props, "track_switch_detection",
		obs_module_text("TrackSwitchDetection"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(track_switch_detection,
				  obs_module_text("TrackSwitchDetection.Off"), 0);
	obs_property_list_add_int(track_switch_detection,
				  obs_module_text("TrackSwitchDetection.Log"), 1);
	obs_property_set_long_description(track_switch_detection,
					  obs_module_text("TrackSwitchDetection.Description"));

	// Hide subproperties completely when unchecked
	obs_property_set_modified_callback(enable_face_exclusion, [](obs_properties_t *props_, obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "enable_face_exclusion");
		for (auto prop_name :
		     {"reference_face_path", "face_match_threshold", "min_face_area_ratio",
		      "face_inference_interval", "max_exempt_persons", "person_category",
		      "preserve_face_exclusion_on_recovery", "track_switch_detection"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			if (prop) obs_property_set_visible(prop, enabled);
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

	// Debug & Statistics Group (At the very bottom)
	obs_properties_t *debug_group_props = obs_properties_create();
	obs_properties_add_group(props, "debug_group", obs_module_text("DebugGroup"), OBS_GROUP_NORMAL, debug_group_props);
	obs_properties_add_int_slider(debug_group_props, "video_delay_frames", obs_module_text("VideoDelayFrames"), 0, 10, 1);
	obs_properties_add_bool(debug_group_props, "debug_mode", obs_module_text("DebugMode")); // FPS/Latency 로 바꿔
	obs_properties_add_bool(debug_group_props, "preview", obs_module_text("Preview"));
	obs_properties_add_bool(debug_group_props, "show_yunet_detections", obs_module_text("ShowYuNetDetections"));
	obs_properties_add_bool(debug_group_props, "enable_face_stats", obs_module_text("ShowFaceSimilarityStats"));
	obs_properties_add_bool(debug_group_props, "enable_similarity_log", obs_module_text("EnableSimilarityLog"));
	
	obs_properties_add_button(debug_group_props, "open_obs_logs", "OBS 로그 보기", [](obs_properties_t *, obs_property_t *, void *) {
		QMainWindow *main_window = static_cast<QMainWindow *>(obs_frontend_get_main_window());
		if (main_window) {
			QMetaObject::invokeMethod(main_window, "on_actionViewCurrentLog_triggered", Qt::QueuedConnection);
		}
		return false;
	});

	obs_properties_add_path(debug_group_props, "save_detections_path", obs_module_text("SaveDetectionsPath"), OBS_PATH_FILE, "*.json", NULL);

	obs_property_t *detected_obj_prop = obs_properties_add_text(
		debug_group_props, "detected_object", obs_module_text("DetectedObject"), OBS_TEXT_DEFAULT);
	obs_property_set_enabled(detected_obj_prop, false);

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
	obs_data_set_default_int(settings, "video_delay_frames", 0);

	obs_data_set_default_bool(settings, "debug_mode", false);
	obs_data_set_default_double(settings, "face_match_threshold", 0.6);
	obs_data_set_default_double(settings, "min_face_area_ratio", 15.0);
	obs_data_set_default_int(settings, "max_exempt_persons", 1);
	obs_data_set_default_bool(settings, "enable_face_stats", false);
	obs_data_set_default_bool(settings, "enable_face_stats_log", false);
	obs_data_set_default_bool(settings, "enable_similarity_log", false);
	obs_data_set_default_bool(settings, "show_yunet_detections", false);
	obs_data_set_default_string(settings, "face_stats_log_path", "");
	
	obs_data_set_default_int(settings, "max_unseen_frames", 15);
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.45);
	obs_data_set_default_string(settings, "model_size", "small");
	obs_data_set_default_int(settings, "object_category", 0);
	obs_data_set_default_bool(settings, "enable_face_exclusion", false);
	obs_data_set_default_bool(settings, "preserve_face_exclusion_on_recovery", false);
	obs_data_set_default_int(settings, "track_switch_detection", 1);
	obs_data_set_default_string(settings, "reference_face_path", "");
	obs_data_set_default_int(settings, "person_category", -1);
	obs_data_set_default_int(settings, "face_inference_interval", 15);
	obs_data_set_default_bool(settings, "masking_group", false);
	obs_data_set_default_string(settings, "masking_type", "blur");
	obs_data_set_default_string(settings, "masking_color", "#000000");
	obs_data_set_default_int(settings, "masking_blur_radius", 10);
	obs_data_set_default_int(settings, "masking_feather", 30);
	obs_data_set_default_int(settings, "masking_dilate_iterations", 0);
	obs_data_set_default_bool(settings, "masking_dynamic_expansion", true);
	obs_data_set_default_double(settings, "masking_dynamic_expansion_base", 1.0);
	obs_data_set_default_double(settings, "masking_dynamic_expansion_ratio", 0.05);
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
	obs_data_set_default_int(settings, "min_size_threshold", 3000);
	obs_data_set_default_double(settings, "ghost_recovery_multiplier", 1.8);
	obs_data_set_default_double(settings, "ghost_recovery_size_ratio_limit", 1.7);
	obs_data_set_default_int(settings, "min_hit_frames", 5);
	obs_data_set_default_double(settings, "iou_threshold", 0.20);
	obs_data_set_default_double(settings, "instant_track_area_ratio", 4.0);
}

static void request_tracker_reset_locked(struct detect_filter *tf);

// Attaches or removes the "Detect Tracking" crop filter once the filter has a parent.
// tf->trackingFilter owns one reference to the crop filter while tracking is enabled.
static void sync_tracking_filter(struct detect_filter *tf)
{
	if (tf->trackingEnabled == tf->trackingRequested)
		return;
	constexpr uint64_t attachGraceTicks = 180; // about 3 seconds at 60 fps
	if (!tf->loadCompleted && tf->ticksSinceCreate < attachGraceTicks)
		return;
	obs_source_t *parent = obs_filter_get_parent(tf->source);
	if (!parent)
		return;

	if (tf->trackingRequested) {
		obs_log(LOG_DEBUG, "Tracking enabled");
		obs_source_t *crop_pad_filter =
			obs_source_get_filter_by_name(parent, "Detect Tracking");
		if (!crop_pad_filter) {
			crop_pad_filter =
				obs_source_create("crop_filter", "Detect Tracking", nullptr, nullptr);
			if (!crop_pad_filter)
				return;
			obs_source_filter_add(parent, crop_pad_filter);
		}
		if (tf->trackingFilter)
			obs_source_release(tf->trackingFilter);
		tf->trackingFilter = crop_pad_filter;
	} else {
		obs_log(LOG_DEBUG, "Tracking disabled");
		obs_source_t *crop_pad_filter =
			obs_source_get_filter_by_name(parent, "Detect Tracking");
		if (crop_pad_filter) {
			obs_source_filter_remove(parent, crop_pad_filter);
			obs_source_release(crop_pad_filter);
		}
		if (tf->trackingFilter) {
			obs_source_release(tf->trackingFilter);
			tf->trackingFilter = nullptr;
		}
	}
	tf->trackingEnabled = tf->trackingRequested;
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	int targetVersion = tf->settingsVersion; // Will be incremented later if queueing loadTask

	tf->preview = obs_data_get_bool(settings, "preview");

	int newVideoDelayFrames = (int)obs_data_get_int(settings, "video_delay_frames");
	if (tf->videoDelayFrames != newVideoDelayFrames) {
		tf->videoDelayFrames = newVideoDelayFrames;
		std::lock_guard<std::mutex> lock(tf->audioMutex);
		tf->resetAudio = true;
	}
	tf->debugMode = obs_data_get_bool(settings, "debug_mode");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	bool newEnableFaceExclusion = obs_data_get_bool(settings, "enable_face_exclusion");
	tf->enableFaceExclusion = newEnableFaceExclusion;
	bool preserveFaceExclusionOnRecovery =
		obs_data_get_bool(settings, "preserve_face_exclusion_on_recovery");
	tf->preserveFaceExclusionOnRecovery = preserveFaceExclusionOnRecovery;
	tf->faceState.setPreserveOnRecovery(preserveFaceExclusionOnRecovery);
	tf->trackSwitchDetection = (int)obs_data_get_int(settings, "track_switch_detection");
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
	tf->maskingFeather = (int)obs_data_get_int(settings, "masking_feather");
	tf->maskingDilateIterations = (int)obs_data_get_int(settings, "masking_dilate_iterations");
	tf->maskingDynamicExpansion = obs_data_get_bool(settings, "masking_dynamic_expansion");
	tf->maskingDynamicExpansionBase = (float)obs_data_get_double(settings, "masking_dynamic_expansion_base");
	tf->maskingDynamicExpansionRatio = (float)obs_data_get_double(settings, "masking_dynamic_expansion_ratio");
	bool newTrackingEnabled = obs_data_get_bool(settings, "tracking_group");
	tf->zoomFactor = (float)obs_data_get_double(settings, "zoom_factor");
	tf->zoomSpeedFactor = (float)obs_data_get_double(settings, "zoom_speed_factor");
	tf->zoomObject = obs_data_get_string(settings, "zoom_object");
	tf->sortTracking = true;
	size_t maxUnseenFrames = (size_t)obs_data_get_int(settings, "max_unseen_frames");
	if (tf->tracker.getMaxUnseenFrames() != maxUnseenFrames) {
		tf->tracker.setMaxUnseenFrames(maxUnseenFrames);
	}
	
	tf->tracker.setKalmanMinNoise(0.35f);
	tf->tracker.setKalmanAreaThreshold(0.0f);
	tf->showUnseenObjects = true;
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
	tf->enableSimilarityLog = obs_data_get_bool(settings, "enable_similarity_log");
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
	float ghost_recovery_size_ratio_limit = (float)obs_data_get_double(settings, "ghost_recovery_size_ratio_limit");
	if (tf->tracker.getGhostRecoverySizeRatioLimit() != ghost_recovery_size_ratio_limit) {
		tf->tracker.setGhostRecoverySizeRatioLimit(ghost_recovery_size_ratio_limit);
	}
	tf->tracker.setGhostRecoveryMaxUnseen(6);

	// During create() the filter has no parent yet; video_tick retries the attach.
	tf->trackingRequested = newTrackingEnabled;
	sync_tracking_filter(tf);

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");
	const std::string externalModelFile = obs_data_get_string(settings, "external_model_file") ? obs_data_get_string(settings, "external_model_file") : "";

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		reinitialize = true;
	}

	bool needsFaceLoader = false;
	if (newEnableFaceExclusion) {
		if (reinitialize || refPathChanged || (!tf->referenceFacesAttempted && tf->referenceFaceFeatures.empty())) {
			needsFaceLoader = true;
		}
	}

	if (reinitialize || needsFaceLoader || (!newEnableFaceExclusion && !tf->referenceFaceFeatures.empty())) {
		targetVersion = ++tf->settingsVersion;
		tf->isModelLoading = true;

		std::string modelFilepath_source = "";
		char *modelFilepath_rawPtr = nullptr;
		bool invalidModel = false;
		if (reinitialize) {
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
				if (!externalModelFile.empty()) {
					modelFilepath_rawPtr = bstrdup(externalModelFile.c_str());
				} else {
					obs_log(LOG_ERROR, "External model file path is empty");
					invalidModel = true;
				}
			} else {
				obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
				invalidModel = true;
			}

			if (modelFilepath_rawPtr) {
				modelFilepath_source = std::string(modelFilepath_rawPtr);
				bfree(modelFilepath_rawPtr);
			} else {
				if (!invalidModel) {
					obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
				}
				invalidModel = true;
			}
		}

		if (invalidModel) {
			// Keep the current model, if any. Without a model, video_render masks the
			// whole frame instead of passing it through.
			tf->isModelLoading = false;
			return;
		}

		auto loadTask = [tf, targetVersion, reinitialize, needsFaceLoader, newUseGpu, newNumThreads, newModelSize, modelFilepath_source, newEnableFaceExclusion, newRefPath, refPathChanged]() {
			// Ensure isModelLoading is safely reset if this is the last queued task
			struct ScopedReset {
				detect_filter* t;
				int tv;
				ScopedReset(detect_filter* tf, int v) : t(tf), tv(v) {}
				~ScopedReset() {
					if (t->settingsVersion == tv) {
						t->isModelLoading = false;
					}
				}
			} reset_guard(tf, targetVersion);

			if (tf->settingsVersion != targetVersion) {
				return;
			}

			std::unique_ptr<ONNXRuntimeModel> tempModel;
			std::vector<std::string> tempClassNames;
#if _WIN32
			std::wstring tempModelFilepath;
#else
			std::string tempModelFilepath;
#endif

			if (reinitialize) {
#if _WIN32
				int outLength = MultiByteToWideChar(CP_UTF8, 0, modelFilepath_source.c_str(),
									-1, nullptr, 0);
				tempModelFilepath = std::wstring(outLength, L'\0');
				MultiByteToWideChar(CP_UTF8, 0, modelFilepath_source.c_str(), -1,
									tempModelFilepath.data(), outLength);
				if (outLength > 0 && !tempModelFilepath.empty() && tempModelFilepath.back() == L'\0') {
					tempModelFilepath.pop_back();
				}
#else
				tempModelFilepath = modelFilepath_source;
#endif

				int onnxruntime_device_id_ = 0;
				bool onnxruntime_use_parallel_ = true;
				float nms_th_ = 0.45f;
				int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
				tempClassNames = edgeyolo_cpp::COCO_CLASSES;

				if (newModelSize == EXTERNAL_MODEL_SIZE) {
					std::vector<std::string> labels;
					std::string error;
					if (!load_class_names_for_model(std::filesystem::path(tempModelFilepath),
									labels, error)) {
						obs_log(LOG_ERROR, "Failed to read class names for %s: %s",
							modelFilepath_source.c_str(), error.c_str());
						return;
					}
					num_classes_ = (int)labels.size();
					tempClassNames = std::move(labels);
				} else if (newModelSize == FACE_YUNET_MODEL_SIZE ||
						   newModelSize == FACE_YOLO_N_MODEL_SIZE ||
						   newModelSize == FACE_YOLO_S_MODEL_SIZE) {
					num_classes_ = 1;
					tempClassNames = yunet::FACE_CLASSES;
				}

				if (tf->settingsVersion != targetVersion) return;

				try {
					if (newModelSize == FACE_YUNET_MODEL_SIZE) {
						tempModel = std::make_unique<yunet::YuNetONNX>(
							tempModelFilepath, newNumThreads, 50, newNumThreads,
							newUseGpu, onnxruntime_device_id_,
							onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
					} else if (newModelSize == FACE_YOLO_N_MODEL_SIZE ||
							   newModelSize == FACE_YOLO_S_MODEL_SIZE) {
						tempModel = std::make_unique<YOLOv8FaceONNX>(
							tempModelFilepath, newNumThreads, newNumThreads,
							newUseGpu, onnxruntime_device_id_,
							onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
					} else {
						tempModel = std::make_unique<edgeyolo_cpp::EdgeYOLOONNXRuntime>(
							tempModelFilepath, newNumThreads, num_classes_,
							newNumThreads, newUseGpu, onnxruntime_device_id_,
							onnxruntime_use_parallel_, nms_th_,
							tf->conf_threshold);
					}
				} catch (const std::exception &e) {
					obs_log(LOG_ERROR, "Failed to load model: %s", e.what());
					return;
				}
			}

			if (tf->settingsVersion != targetVersion) return;

			std::shared_ptr<yunet::YuNetONNX> tempYunet;
			std::shared_ptr<sface::SFaceONNX> tempSface;
			std::vector<std::vector<float>> tempFeatures;

			if (newEnableFaceExclusion) {
				try {
					{
						std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
						tempYunet = tf->yunetModel;
						tempSface = tf->sfaceModel;
						tempFeatures = tf->referenceFaceFeatures;
					}

					if (!tempYunet || !tempSface || needsFaceLoader) {
						char *yunet_raw = obs_module_file("models/face_detection_yunet_2023mar.onnx");
						char *sface_raw = obs_module_file("models/face_recognition_sface_2021dec.onnx");
						
						if (yunet_raw && sface_raw) {
#ifdef _WIN32
							int y_len = MultiByteToWideChar(CP_UTF8, 0, yunet_raw, -1, nullptr, 0);
							std::wstring y_wstr(y_len, L'\0');
							MultiByteToWideChar(CP_UTF8, 0, yunet_raw, -1, y_wstr.data(), y_len);
							if (y_len > 0 && y_wstr.back() == L'\0') y_wstr.pop_back();
							
							int s_len = MultiByteToWideChar(CP_UTF8, 0, sface_raw, -1, nullptr, 0);
							std::wstring s_wstr(s_len, L'\0');
							MultiByteToWideChar(CP_UTF8, 0, sface_raw, -1, s_wstr.data(), s_len);
							if (s_len > 0 && s_wstr.back() == L'\0') s_wstr.pop_back();

							tempYunet = std::make_shared<yunet::YuNetONNX>(y_wstr, 1, 50, 1, "cpu", 0, false, 0.45f, 0.5f);
							tempSface = std::make_shared<sface::SFaceONNX>(s_wstr, 1, 1, "cpu", 0, false);
#else
							tempYunet = std::make_shared<yunet::YuNetONNX>(std::string(yunet_raw), 1, 50, 1, "cpu", 0, false, 0.45f, 0.5f);
							tempSface = std::make_shared<sface::SFaceONNX>(std::string(sface_raw), 1, 1, "cpu", 0, false);
#endif
						}
						if (yunet_raw) bfree(yunet_raw);
						if (sface_raw) bfree(sface_raw);
					}

					if (tf->settingsVersion != targetVersion) return;

					if (needsFaceLoader && !newRefPath.empty() && tempYunet && tempSface) {
						tempFeatures.clear();
#ifdef _WIN32
						int p_len = MultiByteToWideChar(CP_UTF8, 0, newRefPath.c_str(), -1, nullptr, 0);
						std::wstring p_wstr(p_len, L'\0');
						MultiByteToWideChar(CP_UTF8, 0, newRefPath.c_str(), -1, p_wstr.data(), p_len);
						if (p_len > 0 && p_wstr.back() == L'\0') p_wstr.pop_back();
						std::filesystem::path refPath(p_wstr);
#else
						std::filesystem::path refPath(newRefPath);
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
						std::sort(filesToProcess.begin(), filesToProcess.end());

						int successCount = 0;
						int noFaceCount = 0;
						int tooSmallCount = 0;
						int ambiguousCount = 0;
						int unreadableCount = 0;
						int processingErrorCount = 0;
						int embeddingFailureCount = 0;
						for (const auto& filePath : filesToProcess) {
							if (tf->settingsVersion != targetVersion) return;

							try {
#ifdef _WIN32
								FILE* f = _wfopen(filePath.wstring().c_str(), L"rb");
#else
								FILE* f = fopen(filePath.string().c_str(), "rb");
#endif
								cv::Mat refImg;
								if (f) {
									int width, height, channels;
									unsigned char *data = stbi_load_from_file(
										f, &width, &height, &channels, 3);
									fclose(f);
									if (data) {
										cv::Mat img(height, width, CV_8UC3, data);
										cv::cvtColor(img, refImg, cv::COLOR_RGB2BGR);
										stbi_image_free(data);
									}
								}

								if (refImg.empty()) {
									unreadableCount++;
									continue;
								}

								std::vector<Object> faces = tempYunet->inference(refImg);
								const ReferenceFaceSelection selection =
									selectReferenceFace(faces, refImg.size());
								if (selection.status ==
								    ReferenceFaceSelectionStatus::NO_FACE) {
									noFaceCount++;
									continue;
								}
								if (selection.status ==
								    ReferenceFaceSelectionStatus::TOO_SMALL) {
									tooSmallCount++;
									continue;
								}
								if (selection.status ==
								    ReferenceFaceSelectionStatus::AMBIGUOUS) {
									ambiguousCount++;
									continue;
								}

								std::vector<float> feat = tempSface->inference(
									refImg, faces[selection.index].landmarks);
								if (feat.size() == 128) {
									tempFeatures.push_back(std::move(feat));
									successCount++;
								} else {
									embeddingFailureCount++;
								}
							} catch (const std::exception &) {
								processingErrorCount++;
							}
						}

						if (successCount > 0) {
							obs_log(LOG_INFO, "Reference faces loaded successfully: %d images", successCount);
							if (successCount < 3) {
								obs_log(LOG_WARNING,
									"Only %d reference face image(s) were usable. Add clear front, left, and right pose images for stable self recognition.",
									successCount);
							}
						} else {
							obs_log(LOG_WARNING, "No valid faces detected in the reference face path: %s", newRefPath.c_str());
						}
						if (noFaceCount > 0 || tooSmallCount > 0 || ambiguousCount > 0 ||
						    unreadableCount > 0 || processingErrorCount > 0 ||
						    embeddingFailureCount > 0) {
							obs_log(LOG_WARNING,
								"Reference images skipped: unreadable=%d, no face=%d, face too small=%d, multiple similarly sized faces=%d, processing errors=%d, invalid embeddings=%d",
								unreadableCount, noFaceCount, tooSmallCount,
								ambiguousCount, processingErrorCount,
								embeddingFailureCount);
						}
					} else if (newRefPath.empty()) {
						tempFeatures.clear();
					}
				} catch (const std::exception &e) {
					obs_log(LOG_ERROR, "Failed to load face recognition models: %s", e.what());
				}
			}

			if (tf->settingsVersion != targetVersion) return;

			{
				std::lock_guard<std::mutex> lock(tf->modelMutex);
				if (reinitialize) {
					tf->onnxruntimemodel = std::move(tempModel);
					tf->zeroCopyRuntimeHealthy = true;
					tf->zeroCopyPipelineReady = false;
					tf->classNames = std::move(tempClassNames);
					tf->modelFilepath = std::move(tempModelFilepath);
					tf->useGPU = newUseGpu;
					tf->numThreads = newNumThreads;
					tf->modelSize = newModelSize;
					tf->useDirectML = newUseGpu == USEGPU_DML;
#ifdef _WIN32
					tf->modelSupportsZeroCopy = tf->onnxruntimemodel &&
								    tf->onnxruntimemodel->supports_zero_copy();
#endif
				}
				if (tf->onnxruntimemodel) {
					tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
				}
			}

			if (reinitialize) {
				// Tracks and history from the previous model use its labels and box scale.
				{
					std::lock_guard<std::mutex> lock(tf->outputLock);
					tf->latestObjects.clear();
					request_tracker_reset_locked(tf);
				}
				if (!tf->modelLoaded.exchange(true)) {
					// Audio was passed through while no model existed; start delaying it
					// together with the video from now on.
					std::lock_guard<std::mutex> lock(tf->audioMutex);
					tf->resetAudio = true;
				}
			}

			{
				std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
				if (newEnableFaceExclusion) {
					tf->yunetModel = tempYunet;
					tf->sfaceModel = tempSface;
					tf->referenceFaceFeatures = std::move(tempFeatures);
					tf->referenceFacesAttempted = true;
					
					if (refPathChanged) {
						tf->faceState.clear();
						{
							std::lock_guard<std::mutex> d_lock(tf->debugFaceMutex);
							tf->debugFaceBoxes.clear();
						}
					}
				} else {
					tf->yunetModel.reset();
					tf->sfaceModel.reset();
					tf->referenceFaceFeatures.clear();
					tf->referenceFacesAttempted = false;

					tf->faceState.clear();
					{
						std::lock_guard<std::mutex> d_lock(tf->debugFaceMutex);
						tf->debugFaceBoxes.clear();
					}
				}
			}

			if (reinitialize) {
				obs_log(LOG_INFO, "Detect Filter Options updated asynchronously:");
				obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
				obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
				obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
				obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
			}
		};

		{
			std::lock_guard<std::mutex> lock(tf->ioMutex);
			tf->ioQueue.push_back(loadTask);
			tf->ioCV.notify_all();
		}
	} else {
		{
			std::lock_guard<std::mutex> lock(tf->modelMutex);
			if (tf->onnxruntimemodel)
				tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
		}

		if (!newEnableFaceExclusion) {
			std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
			tf->yunetModel.reset();
			tf->sfaceModel.reset();
			tf->referenceFaceFeatures.clear();
			tf->referenceFacesAttempted = false;

			tf->faceState.clear();
			{
				std::lock_guard<std::mutex> d_lock(tf->debugFaceMutex);
				tf->debugFaceBoxes.clear();
			}
		}
	}
}

// Called after a scene collection load has created the parent and all of its filters.
void detect_filter_load(void *data, obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->loadCompleted = true;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	if (tf->effectsLoaded)
		tf->isDisabled = false;
}

// Caller must hold outputLock. The inference thread owns the tracker, so the reset is
// only requested here and applied before the next tracker update.
static void request_tracker_reset_locked(struct detect_filter *tf)
{
	tf->trackerResetGeneration++;
	tf->coverageStartFrameId = 0;
	tf->lastCoveredFrameId = 0;
}

static void clear_all_buffers(struct detect_filter *tf, bool in_graphics_ctx)
{
	bool cleared_any = false;
	auto faceSnapshot = tf->faceState.snapshot();
	if (faceSnapshot && !faceSnapshot->empty()) {
		tf->faceState.clear();
		cleared_any = true;
	}
	if (tf->faceInferenceJobs.size() > 0) {
		tf->faceInferenceJobs.clear();
		cleared_any = true;
	}
	if (!tf->delayedTextures.empty() || !tf->texturePool.empty() || !tf->delayedFrameIds.empty()) {
		if (!in_graphics_ctx) obs_enter_graphics();
		for (auto tex : tf->delayedTextures) gs_texture_destroy(tex);
		tf->delayedTextures.clear();
		for (auto tex : tf->texturePool) gs_texture_destroy(tex);
		tf->texturePool.clear();
		tf->delayedFrameIds.clear();
		if (!in_graphics_ctx) obs_leave_graphics();
		cleared_any = true;
	}

	{
		std::lock_guard<std::mutex> lock(tf->outputLock);
		if (!tf->latestObjects.empty()) {
			tf->latestObjects.clear();
			cleared_any = true;
		}
		if (cleared_any || tf->coverageStartFrameId != 0 || tf->lastCoveredFrameId != 0) {
			request_tracker_reset_locked(tf);
			cleared_any = true;
		}
	}

	if (cleared_any) {
		if (tf->stateHistory) tf->stateHistory->clear();
		std::lock_guard<std::mutex> lock(tf->audioMutex);
		for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
			circlebuf_free(&tf->audioBuffers[i]);
			circlebuf_init(&tf->audioBuffers[i]);
		}
		tf->resetAudio = true;
	}
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
	clear_all_buffers(tf, false);
	tf->last_excluded_count = -1;
	emit_face_exclusion_update(0);
}

/**                   FILTER CORE                     */

static InferenceFrameRequirements get_inference_frame_requirements(const detect_filter *tf)
{
	InferenceFrameRequirements requirements;
	requirements.cropEnabled = tf->crop_enabled;
	requirements.faceExclusionEnabled = tf->enableFaceExclusion;
	requirements.previewEnabled = tf->preview;
	requirements.debugEnabled = tf->debugMode;
	requirements.faceStatsEnabled = tf->enableFaceStats;
	requirements.faceDebugEnabled = tf->showYuNetDetections;
	return requirements;
}

// Log-only detection of moments where an exempt (self) track may have been handed to
// another person: a sudden size or position jump, re-acquisition after being unseen,
// or heavy overlap with another detection. Masking is not changed; the events are
// logged so the rule can be evaluated on real streams before it is enforced.
static void detect_exempt_track_switches(struct detect_filter *tf,
					 const std::vector<Object> &objects, uint64_t frameId)
{
	if (tf->trackSwitchDetection.load() == 0) {
		tf->exemptTrackLastRects.clear();
		tf->trackSwitchSuspects.clear();
		return;
	}

	const auto now = std::chrono::steady_clock::now();
	std::unordered_map<uint64_t, filter_data::ExemptTrackSample> current;
	for (const Object &obj : objects) {
		if (!obj.isExempt)
			continue;
		current[obj.id] = {obj.rect, obj.unseenFrames};

		auto previous = tf->exemptTrackLastRects.find(obj.id);
		if (previous == tf->exemptTrackLastRects.end())
			continue;
		const cv::Rect2f &before = previous->second.rect;
		const float beforeArea = before.area();
		const float areaRatio = beforeArea > 0.0f ? obj.rect.area() / beforeArea : 1.0f;
		const cv::Point2f beforeCenter(before.x + before.width / 2.0f,
					       before.y + before.height / 2.0f);
		const cv::Point2f center(obj.rect.x + obj.rect.width / 2.0f,
					 obj.rect.y + obj.rect.height / 2.0f);
		const float jump = (float)cv::norm(center - beforeCenter);

		const char *reason = nullptr;
		if (areaRatio < 0.5f || areaRatio > 2.0f)
			reason = "size jump";
		else if (jump > 0.5f * std::max(before.width, before.height))
			reason = "position jump";
		else if (previous->second.unseenFrames > 0 && obj.unseenFrames == 0)
			reason = "re-acquired";
		else {
			for (const Object &other : objects) {
				if (other.id == obj.id || other.unseenFrames > 0)
					continue;
				const float inter = (obj.rect & other.rect).area();
				const float uni = obj.rect.area() + other.rect.area() - inter;
				if (uni > 0.0f && inter / uni > 0.3f) {
					reason = "overlap";
					break;
				}
			}
		}
		if (!reason)
			continue;

		tf->trackSwitchSuspects[obj.id] = now;
		if (tf->lastTrackSwitchLog.time_since_epoch().count() != 0 &&
		    now - tf->lastTrackSwitchLog < std::chrono::seconds(1)) {
			tf->suppressedTrackSwitchLogs++;
			continue;
		}
		obs_log(LOG_INFO,
			"[TrackSwitch] exempt track %llu epoch %llu: %s at frame %llu (area x%.2f, moved %.0f px, %d earlier events not logged)",
			(unsigned long long)obj.id, (unsigned long long)obj.identityEpoch, reason,
			(unsigned long long)frameId, areaRatio, jump, tf->suppressedTrackSwitchLogs);
		tf->lastTrackSwitchLog = now;
		tf->suppressedTrackSwitchLogs = 0;
	}
	tf->exemptTrackLastRects = std::move(current);

	for (auto it = tf->trackSwitchSuspects.begin(); it != tf->trackSwitchSuspects.end();) {
		if (now - it->second > std::chrono::seconds(1))
			it = tf->trackSwitchSuspects.erase(it);
		else
			++it;
	}
}

static void run_model_inference(struct detect_filter *tf, InferenceJob job)
{
	struct InferenceCompletionGuard {
		std::atomic<bool> &flag;
		~InferenceCompletionGuard() { flag = false; }
	} completionGuard{tf->isInferencing};

	static auto last_loop_time = std::chrono::high_resolution_clock::now();

	// Apply tracker resets requested by other threads (deactivate, model swap).
	const uint64_t resetGeneration = tf->trackerResetGeneration.load();
	if (resetGeneration != tf->appliedTrackerResetGeneration) {
		tf->tracker.reset();
		if (tf->stateHistory)
			tf->stateHistory->clear();
		tf->coverageUpdates = 0;
		tf->exemptTrackLastRects.clear();
		tf->appliedTrackerResetGeneration = resetGeneration;
	}

	std::unique_lock<std::mutex> lock(tf->modelMutex);
	if (!tf->onnxruntimemodel)
		return;

	cv::Mat imageBGRA = std::move(job.ownedBGRA);
	const uint64_t inferenceFrameId = job.frameId;
	const uint32_t sourceWidth = job.sourceWidth > 0 ? job.sourceWidth
							 : (uint32_t)std::max(0, imageBGRA.cols);
	const uint32_t sourceHeight = job.sourceHeight > 0 ? job.sourceHeight
							   : (uint32_t)std::max(0, imageBGRA.rows);
	if (sourceWidth == 0 || sourceHeight == 0)
		return;

	auto start_time = std::chrono::high_resolution_clock::now();

	std::vector<Object> objects;
	bool gpu_run_success = false;
	cv::Rect cropRect(0, 0, 1, 1);

#ifdef _WIN32
	if (job.mode == InferenceMode::GPUZeroCopy && tf->onnxruntimemodel->supports_zero_copy() &&
	    tf->onnxruntimemodel->is_gpu_pipeline_ready() && tf->zeroCopyRuntimeHealthy) {
		try {
			objects = tf->onnxruntimemodel->inference_gpu();
			gpu_run_success = true;
		} catch (const std::exception &e) {
			tf->zeroCopyRuntimeHealthy = false;
			tf->zeroCopyPipelineReady = false;
			obs_log(LOG_ERROR,
				"GPU Zero-Copy Inference failed: %s. Keeping prior tracks and using the owned DirectML path on the next frame.",
				e.what());
			return;
		}
	}
#endif

	if (!gpu_run_success) {
		if (imageBGRA.empty())
			return;
		cv::Mat inferenceFrame;
		cropRect = cv::Rect(0, 0, imageBGRA.cols, imageBGRA.rows);
		if (tf->crop_enabled) {
			const int cropWidth = imageBGRA.cols - tf->crop_left - tf->crop_right;
			const int cropHeight = imageBGRA.rows - tf->crop_top - tf->crop_bottom;
			if (tf->crop_left < 0 || tf->crop_top < 0 || tf->crop_right < 0 ||
			    tf->crop_bottom < 0 || cropWidth <= 0 || cropHeight <= 0) {
				obs_log(LOG_ERROR,
					"Inference crop is outside the source frame. Keeping prior tracks.");
				return;
			}
			cropRect = cv::Rect(tf->crop_left, tf->crop_top, cropWidth, cropHeight);
		}
		try {
			if (tf->crop_enabled)
				cv::cvtColor(imageBGRA(cropRect), inferenceFrame,
					     cv::COLOR_BGRA2BGR);
			else
				cv::cvtColor(imageBGRA, inferenceFrame, cv::COLOR_BGRA2BGR);
		} catch (const cv::Exception &e) {
			obs_log(LOG_ERROR, "Inference frame preprocessing failed: %s", e.what());
			return;
		}

		try {
			objects = tf->onnxruntimemodel->inference(inferenceFrame);
		} catch (const Ort::Exception &e) {
			obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
			return;
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "%s", e.what());
			return;
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
	}

	// Labels in `objects` belong to the model that just ran, so copy its class names
	// before another thread can swap the model.
	const std::vector<std::string> classNames = tf->classNames;
	lock.unlock();

	// update the detected object text input
	if (objects.size() > 0) {
		if (tf->lastDetectedObjectId != objects[0].label) {
			tf->lastDetectedObjectId = objects[0].label;
			// get source settings
			obs_data_t *source_settings = obs_source_get_settings(tf->source);
			obs_data_set_string(source_settings, "detected_object",
					    class_name_or_unknown(classNames, objects[0].label));
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
		float screenArea = (float)(sourceWidth * sourceHeight);
		tf->tracker.setScreenArea(screenArea);
		tf->tracker.setScreenDimensions((float)sourceWidth, (float)sourceHeight);
		objects = tf->tracker.update(inferenceFrameId, objects);
	} else {
		// When SORT tracking is off, track objects by assigning temporary IDs and record states to history
		if (tf->stateHistory) {
			std::map<int, TrackedObjectState> current_frame_states;
			int temp_id = 1;
			for (auto &obj : objects) {
				obj.id = temp_id++;
				
				TrackedObjectState state;
				state.id = (int)obj.id;
				state.identityEpoch = obj.identityEpoch;
				state.rect = obj.rect;
				state.is_confirmed = true;
				state.is_extrapolated = false;
				state.hit_streak = 1;
				
				current_frame_states[(int)obj.id] = state;
			}
			tf->stateHistory->push_frame_state(inferenceFrameId, current_frame_states);
		}
	}

	std::vector<Object> all_objects; // for preview

	if (tf->enableFaceExclusion && tf->sortTracking) {
		std::vector<FaceTrackIdentity> currentTracks;
		currentTracks.reserve(objects.size());
		for (const Object &obj : objects)
			currentTracks.push_back({obj.id, obj.identityEpoch, obj.rect.area()});
		tf->faceState.synchronizeTracks(currentTracks);

		bool faceModelsReady = false;
		{
			std::lock_guard<std::mutex> faceLock(tf->faceInferenceMutex);
			faceModelsReady = tf->yunetModel && tf->sfaceModel &&
					  !tf->referenceFaceFeatures.empty();
		}

		if (faceModelsReady && !imageBGRA.empty()) {
			const float screenArea = (float)(sourceWidth * sourceHeight);
			const float minPersonAreaPixels =
				screenArea * (tf->minFaceAreaRatio / 100.0f);
			const uint64_t nowMs = face_now_ms();
			const FaceRetryPolicy retryPolicy =
				face_retry_policy(tf->faceInferenceInterval);
			std::vector<FaceInferenceRequest> requests;
			bool priorityJob = false;

			for (const Object &obj : objects) {
				if (tf->personCategory != -1 && obj.label != tf->personCategory)
					continue;

				FaceTrackDecision decision =
					tf->faceState.getDecision(obj.id, obj.identityEpoch);
				if (decision.checkInFlight || isSettledSelf(decision))
					continue;
				if (obj.rect.area() < minPersonAreaPixels)
					continue;
				bool priority = false;
				uint64_t checkToken = 0;
				if (tf->faceState.beginCheck(obj.id, obj.identityEpoch, nowMs,
							     &priority, &checkToken)) {
					requests.push_back({obj, checkToken});
					priorityJob = priorityJob || priority;
				}
			}

			if (!requests.empty()) {
				FaceInferenceJob faceJob;
				faceJob.imageBGRA = imageBGRA.clone();
				faceJob.requests = requests;
				faceJob.frameId = inferenceFrameId;
				faceJob.priority = priorityJob;
				faceJob.matchThreshold = tf->faceMatchThreshold;
				faceJob.maxExemptPersons = tf->maxExemptPersons;
				faceJob.settingsVersion = tf->settingsVersion;
				faceJob.retryPolicy = retryPolicy;
				if (!tf->faceInferenceJobs.tryPush(std::move(faceJob))) {
					for (const FaceInferenceRequest &request : requests) {
						tf->faceState.cancelCheck(
							request.object.id, request.object.identityEpoch,
							request.checkToken, nowMs,
							retryPolicy.noFaceRetryMs);
					}
					const auto now = std::chrono::steady_clock::now();
					if (tf->lastFaceQueueFullLog.time_since_epoch().count() == 0 ||
					    now - tf->lastFaceQueueFullLog >= std::chrono::seconds(5)) {
						obs_log(LOG_WARNING,
							"Face inference queue is full. Deferred checks will retry without losing their state.");
						tf->lastFaceQueueFullLog = now;
					}
				}
			}
		}
	}

	if (tf->enableFaceExclusion) {
		std::vector<Object> filtered_objects;
		auto faceSnapshot = tf->faceState.snapshot();

		for (Object &obj : objects) {
			if (faceSnapshot &&
			    FaceStateStore::isExempt(*faceSnapshot, obj.id, obj.identityEpoch)) {
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
		detect_exempt_track_switches(tf, all_objects, inferenceFrameId);
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
		for (Object &obj : all_objects) {
			nlohmann::json obj_json;
			obj_json["label"] = class_name_or_unknown(classNames, obj.label);
			obj_json["confidence"] = obj.prob;
			obj_json["rect"] = {{"x", obj.rect.x},
					    {"y", obj.rect.y},
					    {"width", obj.rect.width},
					    {"height", obj.rect.height}};
			obj_json["id"] = obj.id;
			j.push_back(obj_json);
		}
		
		std::string jsonStr = j.dump(4);
		
		{
			std::lock_guard<std::mutex> lock(tf->ioMutex);
			tf->ioQueue.push_back([savePath, jsonStr]() {
#if defined(_WIN32) && (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
				std::ofstream detectionsFile(std::filesystem::u8path(savePath));
#else
				std::ofstream detectionsFile(savePath);
#endif
				if (detectionsFile.is_open()) {
					detectionsFile << jsonStr;
					detectionsFile.close();
				} else {
					obs_log(LOG_ERROR, "Failed to open file for writing detections: %s",
						savePath.c_str());
				}
			});
		}
		tf->ioCV.notify_one();
	}

	if (tf->preview || tf->debugMode || tf->enableFaceStats || tf->showYuNetDetections) {
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
						FaceTrackDecision faceDecision =
							tf->faceState.getDecision(
								obj.id, obj.identityEpoch);
						const float sim = faceDecision.similarity;
						if (faceDecision.checkInFlight) {
							obj.customText = obj.isExempt
									 ? "Exempted / Rechecking Face..."
									 : "Checking Face...";
						} else if (faceDecision.hasSimilarity) {
							char buf[64];
							snprintf(buf, sizeof(buf), "Sim: %.2f", sim);
							if (obj.isExempt) {
								obj.customText = std::string("Exempted / ") + buf;
							} else if (faceDecision.lastOutcome ==
								   FaceCheckOutcome::EXEMPT_LIMIT) {
								obj.customText = std::string("Exempt Limit / ") + buf;
							} else {
								obj.customText = std::string("Not Me / ") + buf;
							}
						} else if (faceDecision.lastOutcome ==
							   FaceCheckOutcome::NO_FACE) {
							obj.customText = "Waiting for a Visible Face";
						} else if (faceDecision.lastOutcome ==
							   FaceCheckOutcome::INFERENCE_ERROR) {
							obj.customText = "Face Check Retry Pending";
						} else {
							obj.customText = "Face Check Pending";
						}
					}
					if (tf->trackSwitchSuspects.count(obj.id))
						obj.customText = "Switch? / " + obj.customText;
				}
			}
		}
		
		cv::Mat overlay = cv::Mat::zeros(frame.size(), CV_8UC4);

		if (tf->preview) {
			draw_objects(overlay, all_objects, classNames);
		}
		if (tf->maskingEnabled) {
			// OpenCV Mask generation has been moved to GPU shaders
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
				
				if (hist[i] > 0) {
					snprintf(txt, sizeof(txt), "%d", hist[i]);
					cv::putText(overlay, txt, cv::Point(histX + i*barW + 2, histY - barHeight - 2), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255,255), 1);
				}
				
				snprintf(txt, sizeof(txt), ".%d", i);
				cv::putText(overlay, txt, cv::Point(histX + i*barW + 2, histY + 12), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(200,200,200,255), 1);
			}
		}

		std::lock_guard<std::mutex> lock(tf->outputLock);
		tf->previewHistory[inferenceFrameId] = overlay.clone();
		// Clean up old preview frames
		while (!tf->previewHistory.empty() &&
		       tf->previewHistory.begin()->first <
			       (inferenceFrameId > 120 ? inferenceFrameId - 120 : 0)) {
			tf->previewHistory.erase(tf->previewHistory.begin());
		}
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
		if (tf->trackerResetGeneration.load() != resetGeneration) {
			// A reset was requested while this frame was processed. Drop the results;
			// the next job resets the tracker before it runs.
			return;
		}
		tf->latestObjects = objects;
		// Masking trusts the history only once tracks present at the reset had enough
		// updates to be confirmed.
		if (tf->coverageUpdates < std::numeric_limits<int>::max())
			tf->coverageUpdates++;
		if (tf->coverageStartFrameId == 0 &&
		    tf->coverageUpdates >= std::max(1, tf->minHitFrames))
			tf->coverageStartFrameId = inferenceFrameId;
		tf->lastCoveredFrameId = inferenceFrameId;
	}
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
	tf->maskTexrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	tf->lastDetectedObjectId = -1;
	tf->currentFrameId = 0;
	tf->stateHistory = std::make_shared<StateHistoryManager>();
	tf->tracker.setStateHistoryManager(tf->stateHistory);

	for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
		circlebuf_init(&tf->audioBuffers[i]);
		tf->lastSamples[i][0] = 0.0f;
		tf->lastSamples[i][1] = 0.0f;
	}

	std::vector<std::tuple<const char *, gs_effect_t **>> effects = {
		{KAWASE_BLUR_EFFECT_PATH, &tf->kawaseBlurEffect},
		{MASKING_EFFECT_PATH, &tf->maskingEffect},
		{PIXELATE_EFFECT_PATH, &tf->pixelateEffect},
		{MASK_SHAPE_EFFECT_PATH, &tf->maskShapeEffect},
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
		obs_leave_graphics();
		bfree(effectPathPtr);
		if (!*effect) {
			obs_log(LOG_ERROR, "Failed to load effect: %s", effectPath);
			tf->isDisabled = true;
			return tf;
		}
	}
	tf->effectsLoaded = true;

	detect_filter_update(tf, settings);

	tf->stopInferenceThread = false;
	tf->isInferencing = false;
	tf->inferenceFrameReady = false;
	tf->stopFaceInferenceThread = false;

	tf->stopIOThread = false;
	tf->ioThread = std::thread([tf]() {
		while (true) {
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lock(tf->ioMutex);
				tf->ioCV.wait(lock, [tf] {
					return !tf->ioQueue.empty() || tf->stopIOThread;
				});
				if (tf->stopIOThread && tf->ioQueue.empty()) {
					break;
				}
				if (!tf->ioQueue.empty()) {
					task = std::move(tf->ioQueue.front());
					tf->ioQueue.erase(tf->ioQueue.begin());
				}
			}
			if (task) {
				// An exception escaping this thread would terminate OBS.
				try {
					task();
				} catch (const std::exception &e) {
					obs_log(LOG_ERROR, "Background task failed: %s", e.what());
				} catch (...) {
					obs_log(LOG_ERROR, "Background task failed with an unknown error");
				}
			}
		}
	});

	tf->inferenceThread = std::thread(inference_thread_loop, tf);
	tf->faceInferenceThread = std::thread(face_inference_thread_loop, tf);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		emit_face_exclusion_update(0);
		tf->isDisabled = true;

		tf->stopInferenceThread = true;
		tf->inferenceCV.notify_all();
		if (tf->inferenceThread.joinable()) {
			tf->inferenceThread.join();
		}
		
		tf->stopFaceInferenceThread = true;
		tf->faceInferenceJobs.notifyAll();
		if (tf->faceInferenceThread.joinable()) {
			tf->faceInferenceThread.join();
		}

		tf->stopIOThread = true;
		tf->ioCV.notify_all();
		if (tf->ioThread.joinable()) {
			tf->ioThread.join();
		}

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->maskTexrender) {
			gs_texrender_destroy(tf->maskTexrender);
		}
		for (auto tex : tf->delayedTextures) gs_texture_destroy(tex);
		tf->delayedTextures.clear();
		for (auto tex : tf->texturePool) gs_texture_destroy(tex);
		tf->texturePool.clear();
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		if (tf->previewTexture) {
			gs_texture_destroy(tf->previewTexture);
		}
		if (tf->baseTexture) {
			gs_texture_destroy(tf->baseTexture);
		}
		gs_effect_destroy(tf->kawaseBlurEffect);
		gs_effect_destroy(tf->maskingEffect);
		if (tf->pixelateEffect) {
			gs_effect_destroy(tf->pixelateEffect);
		}
		if (tf->maskShapeEffect) {
			gs_effect_destroy(tf->maskShapeEffect);
		}
		obs_leave_graphics();

		for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
			circlebuf_free(&tf->audioBuffers[i]);
		}

		if (tf->trackingFilter) {
			obs_source_release(tf->trackingFilter);
			tf->trackingFilter = nullptr;
		}

		tf->~detect_filter();
		bfree(tf);
	}
}

static void inference_thread_loop(struct detect_filter *tf)
{
	obs_log(LOG_INFO, "Inference thread started");
	while (!tf->stopInferenceThread) {
		InferenceJob job;
		{
			std::unique_lock<std::mutex> lock(tf->inferenceMutex);
			tf->inferenceCV.wait(lock, [tf] {
				return tf->inferenceFrameReady || tf->stopInferenceThread;
			});

			if (tf->stopInferenceThread) {
				break;
			}

			job = std::move(tf->inferenceJob);
			tf->inferenceFrameReady = false;
		}

		run_model_inference(tf, std::move(job));
	}
	obs_log(LOG_INFO, "Inference thread stopped");
}

static void face_inference_thread_loop(struct detect_filter *tf)
{
	obs_log(LOG_INFO, "Face Inference thread started");
	while (!tf->stopFaceInferenceThread) {
		FaceInferenceJob job;
		if (!tf->faceInferenceJobs.waitPop(job, tf->stopFaceInferenceThread)) {
			if (tf->stopFaceInferenceThread)
				break;
			continue;
		}
		if (tf->stopFaceInferenceThread)
			break;

		if (job.settingsVersion != tf->settingsVersion) {
			const uint64_t nowMs = face_now_ms();
			for (const FaceInferenceRequest &request : job.requests) {
				tf->faceState.cancelCheck(
					request.object.id, request.object.identityEpoch,
					request.checkToken, nowMs, job.retryPolicy.errorRetryMs);
			}
			continue;
		}

		std::shared_ptr<yunet::YuNetONNX> local_yunet;
		std::shared_ptr<sface::SFaceONNX> local_sface;
		std::vector<std::vector<float>> localReferenceFeatures;
		{
			std::lock_guard<std::mutex> lock(tf->faceInferenceMutex);
			local_yunet = tf->yunetModel;
			local_sface = tf->sfaceModel;
			localReferenceFeatures = tf->referenceFaceFeatures;
		}

		if (job.imageBGRA.empty() || !local_yunet || !local_sface ||
		    localReferenceFeatures.empty()) {
			const uint64_t nowMs = face_now_ms();
			for (const FaceInferenceRequest &request : job.requests) {
				tf->faceState.completeCheck(
					request.object.id, request.object.identityEpoch,
					request.checkToken, FaceCheckOutcome::INFERENCE_ERROR,
					std::nullopt, job.maxExemptPersons, nowMs,
					job.retryPolicy);
			}
			continue;
		}

		std::vector<std::string> localClassNames;
		{
			std::lock_guard<std::mutex> modelLock(tf->modelMutex);
			localClassNames = tf->classNames;
		}

		for (const FaceInferenceRequest &request : job.requests) {
			if (tf->stopFaceInferenceThread)
				break;
			const Object &obj = request.object;
			if (job.settingsVersion != tf->settingsVersion) {
				tf->faceState.cancelCheck(
					obj.id, obj.identityEpoch, request.checkToken, face_now_ms(),
					job.retryPolicy.errorRetryMs);
				continue;
			}

			FaceCheckOutcome outcome = FaceCheckOutcome::INFERENCE_ERROR;
			std::optional<float> similarity;
			try {
				const bool isAlreadyFace =
					obj.label >= 0 && (size_t)obj.label < localClassNames.size() &&
					(localClassNames[obj.label] == "face" ||
					 localClassNames[obj.label] == "head");
				const bool hasValidLandmarks =
					isAlreadyFace &&
					(std::abs(obj.landmarks[0].x - obj.landmarks[1].x) > 1e-3 ||
					 std::abs(obj.landmarks[0].y - obj.landmarks[1].y) > 1e-3);

				cv::Rect2f cropBox;
				if (isAlreadyFace && hasValidLandmarks) {
					cropBox = cv::Rect2f(0, 0, (float)job.imageBGRA.cols,
							 (float)job.imageBGRA.rows);
				} else {
					cropBox = obj.rect;
					if (isAlreadyFace) {
						const float marginW = cropBox.width * 0.2f;
						const float marginH = cropBox.height * 0.2f;
						cropBox.x -= marginW;
						cropBox.y -= marginH;
						cropBox.width += marginW * 2.0f;
						cropBox.height += marginH * 2.0f;
					}
				}

				const cv::Rect cropRect =
					cropBox & cv::Rect2f(0, 0, (float)job.imageBGRA.cols,
							    (float)job.imageBGRA.rows);
				if (cropRect.width <= 0 || cropRect.height <= 0) {
					outcome = FaceCheckOutcome::INFERENCE_ERROR;
				} else {
					cv::Mat croppedBGR;
					cv::cvtColor(job.imageBGRA(cropRect), croppedBGR,
						     cv::COLOR_BGRA2BGR);

					std::vector<Object> faces;
					if (isAlreadyFace && hasValidLandmarks) {
						Object shiftedFace = obj;
						for (int landmark = 0; landmark < 5; ++landmark) {
							shiftedFace.landmarks[landmark].x -= cropRect.x;
							shiftedFace.landmarks[landmark].y -= cropRect.y;
						}
						shiftedFace.rect.x -= cropRect.x;
						shiftedFace.rect.y -= cropRect.y;
						faces.push_back(shiftedFace);
					} else {
						faces = local_yunet->inference(croppedBGR);
					}

					if (faces.empty()) {
						outcome = FaceCheckOutcome::NO_FACE;
					} else {
						if (!(isAlreadyFace && hasValidLandmarks)) {
							std::vector<DebugFaceBox> debugBoxes;
							for (const Object &face : faces) {
								DebugFaceBox debugBox;
								debugBox.rect = face.rect;
								debugBox.rect.x += cropRect.x;
								debugBox.rect.y += cropRect.y;
								debugBox.timestamp =
									std::chrono::steady_clock::now();
								debugBoxes.push_back(debugBox);
							}
							{
								std::lock_guard<std::mutex> debugLock(
									tf->debugFaceMutex);
								tf->debugFaceBoxes.insert(
									tf->debugFaceBoxes.end(), debugBoxes.begin(),
									debugBoxes.end());
							}

							if (faces.size() > 1) {
								const float personCenterX =
									(float)cropRect.width / 2.0f;
								const float personTopY =
									(float)cropRect.height * 0.15f;
								std::sort(
									faces.begin(), faces.end(),
									[personCenterX, personTopY](
										const Object &left,
										const Object &right) {
										auto score = [&](const Object &face) {
											const float dx =
												face.rect.x +
													face.rect.width / 2.0f -
												personCenterX;
											const float dy =
												face.rect.y +
													face.rect.height / 2.0f -
												personTopY;
											return dx * dx * 2.0f + dy * dy;
										};
										return score(left) < score(right);
									});
							}
						}

						const std::vector<float> feature =
							local_sface->inference(croppedBGR,
									       faces[0].landmarks);
						if (feature.size() != 128) {
							outcome = FaceCheckOutcome::INFERENCE_ERROR;
						} else {
							float maxSimilarity = -1.0f;
							for (const auto &reference : localReferenceFeatures) {
								maxSimilarity = std::max(
									maxSimilarity,
									sface::SFaceONNX::match(feature, reference));
							}
							similarity = maxSimilarity;
							outcome = maxSimilarity >= job.matchThreshold
									  ? FaceCheckOutcome::MATCH
									  : FaceCheckOutcome::MISMATCH;

							if (tf->enableSimilarityLog) {
								obs_log(LOG_INFO,
									"Face check obj %llu epoch %llu: similarity %.3f (threshold %.3f)",
									(unsigned long long)obj.id,
									(unsigned long long)obj.identityEpoch,
									maxSimilarity, job.matchThreshold);
							}

							if (tf->enableFaceStats) {
								std::lock_guard<std::mutex> statLock(
									tf->statsMutex);
								tf->statTotalChecks++;
								tf->statSumSimilarity += maxSimilarity;
								if (outcome == FaceCheckOutcome::MATCH)
									tf->statIsMe++;
								else
									tf->statNotMe++;
								const int bin = std::max(
									0, std::min(9, (int)(maxSimilarity * 10.0f)));
								tf->similarityHistogram[bin]++;
							}
						}
					}
				}
			} catch (const std::exception &error) {
				obs_log(LOG_ERROR, "Face check failed for obj %llu epoch %llu: %s",
					(unsigned long long)obj.id,
					(unsigned long long)obj.identityEpoch, error.what());
				outcome = FaceCheckOutcome::INFERENCE_ERROR;
			}

			bool committed = false;
			if (job.settingsVersion != tf->settingsVersion) {
				tf->faceState.cancelCheck(
					obj.id, obj.identityEpoch, request.checkToken, face_now_ms(),
					job.retryPolicy.errorRetryMs);
			} else {
				committed = tf->faceState.completeCheck(
					obj.id, obj.identityEpoch, request.checkToken, outcome,
					similarity, job.maxExemptPersons, face_now_ms(),
					job.retryPolicy);
			}
			if (committed && tf->enableSimilarityLog &&
			    outcome == FaceCheckOutcome::NO_FACE) {
				obs_log(LOG_INFO,
					"Face check obj %llu epoch %llu: no visible face; retrying in %llu ms",
					(unsigned long long)obj.id,
					(unsigned long long)obj.identityEpoch,
					(unsigned long long)job.retryPolicy.noFaceRetryMs);
			}
		}
	}
	obs_log(LOG_INFO, "Face Inference thread stopped");
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->tickRendered = false;
	tf->ticksSinceCreate++;
	sync_tracking_filter(tf);

	if (tf->isDisabled || !tf->modelLoaded || !obs_source_enabled(tf->source)) {
		clear_all_buffers(tf, false);
		return;
	}

	bool use_gpu_zero_copy = false;
#ifdef _WIN32
	use_gpu_zero_copy = canUseGpuZeroCopy(tf->useDirectML, tf->modelSupportsZeroCopy,
					      tf->zeroCopyPipelineReady,
					      tf->zeroCopyRuntimeHealthy,
					      get_inference_frame_requirements(tf));
#endif

	cv::Mat imageBGRA;
	uint64_t ownedFrameId = 0;
	int frameWidth = 0;
	int frameHeight = 0;
	if (!use_gpu_zero_copy) {
		std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
		if (!lock.owns_lock() || tf->inputBGRA.empty()) {
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
		ownedFrameId = tf->inputBGRAFrameId;
		frameWidth = imageBGRA.cols;
		frameHeight = imageBGRA.rows;
	}

	if (!use_gpu_zero_copy) {
		std::unique_lock<std::mutex> lock(tf->inferenceMutex, std::try_to_lock);
		if (lock.owns_lock() && !tf->isInferencing && !tf->inferenceFrameReady &&
		    ownedFrameId > tf->lastQueuedInferenceFrameId) {
			tf->inferenceJob.mode = InferenceMode::OwnedBGRA;
			tf->inferenceJob.ownedBGRA = std::move(imageBGRA);
			tf->inferenceJob.sourceWidth = (uint32_t)tf->inferenceJob.ownedBGRA.cols;
			tf->inferenceJob.sourceHeight = (uint32_t)tf->inferenceJob.ownedBGRA.rows;
			tf->inferenceJob.frameId = ownedFrameId;
			tf->lastQueuedInferenceFrameId = ownedFrameId;
			tf->isInferencing = true;
			tf->inferenceFrameReady = true;
			tf->inferenceCV.notify_one();
		}
	}

	if (tf->trackingEnabled && tf->trackingFilter) {
		int width = 0;
		int height = 0;
		if (frameWidth > 0 && frameHeight > 0) {
			width = frameWidth;
			height = frameHeight;
		} else {
			width = tf->lastTexWidth;
			height = tf->lastTexHeight;
		}
		if (width <= 0 || height <= 0) return;

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

// Bounding box scaling (Padding)
static cv::Rect2f scaleBoundingBox(const cv::Rect2f& rect, float scale) {
	float dw = rect.width * (scale - 1.0f);
	float dh = rect.height * (scale - 1.0f);
	return cv::Rect2f(rect.x - dw / 2.0f, rect.y - dh / 2.0f, rect.width + dw, rect.height + dh);
}

// Final render box for objects (confirmed/extrapolated/missing)
static cv::Rect2f getFinalRenderBox(int object_id, uint64_t R,
				    const StateHistoryManager *historyManager,
				    uint64_t current_realtime_T, int preemptiveMaskingFrames,
				    std::string &out_mode, uint64_t *out_identity_epoch = nullptr)
{
	if (out_identity_epoch)
		*out_identity_epoch = 0;
	if (!historyManager) {
		out_mode = "None";
		return cv::Rect2f(0, 0, 0, 0);
	}
	
	TrackedObjectState current_state;
	bool exists_now = historyManager->get_object_state(R, object_id, current_state);

	// 1. Object exists in current frame and is confirmed (not extrapolated)
	if (exists_now && current_state.is_confirmed && !current_state.is_extrapolated) {
		out_mode = "Confirmed";
		if (out_identity_epoch)
			*out_identity_epoch = current_state.identityEpoch;
		return current_state.rect;
	}

	// 2. Search for future state in lookup buffer (R+1 to current_realtime_T)
	bool will_be_confirmed = false;
	bool will_be_recovered = false;
	uint64_t first_active_future_frame = 0;

	const auto buffer = historyManager->get_buffer_copy();
	for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
		if (it->frame_id > R && it->frame_id <= current_realtime_T) {
			auto obj_it = it->objects.find(object_id);
			if (obj_it != it->objects.end()) {
				const auto& future_state = obj_it->second;
				if (future_state.is_confirmed && !future_state.is_extrapolated) {
					will_be_confirmed = true;
					will_be_recovered = true;
					first_active_future_frame = it->frame_id;
					// Since we iterate backwards (newest to oldest), the LAST match we find 
					// will be the EARLIEST future frame! So we don't break immediately, we let it finish.
				}
			}
		}
	}

	// 3. Apply reverse masking for the earliest future frame (confirmed, within preemptive masking range, not confirmed in current frame)
	if (will_be_confirmed && (first_active_future_frame > R) && (first_active_future_frame - R <= (uint64_t)preemptiveMaskingFrames) && (!exists_now || !current_state.is_confirmed)) {
		TrackedObjectState future_active_state;
		if (historyManager->get_object_state(first_active_future_frame, object_id, future_active_state)) {
			out_mode = "Reverse Masking";
			if (out_identity_epoch)
				*out_identity_epoch = future_active_state.identityEpoch;

			// Get next future frame for scaling
			TrackedObjectState next_future_state;
			bool has_next = false;
			uint64_t next_frame = first_active_future_frame;
			
			// Find the furthest confirmed frame within a window of 10 frames to calculate a stable velocity
			for (auto it = buffer.begin(); it != buffer.end(); ++it) {
				if (it->frame_id > first_active_future_frame && it->frame_id <= first_active_future_frame + 10) {
					auto obj_it = it->objects.find(object_id);
					if (obj_it != it->objects.end() && obj_it->second.is_confirmed && !obj_it->second.is_extrapolated) {
						next_future_state = obj_it->second;
						next_frame = it->frame_id;
						has_next = true;
					}
				} else if (it->frame_id > first_active_future_frame + 10) {
					break;
				}
			}

			if (has_next && next_frame > first_active_future_frame) {
				// Calculate velocity vector (Distance / Frame)
				float dt = static_cast<float>(next_frame - first_active_future_frame);
				float dx = (next_future_state.rect.x - future_active_state.rect.x) / dt;
				float dy = (next_future_state.rect.y - future_active_state.rect.y) / dt;
				float dw = (next_future_state.rect.width - future_active_state.rect.width) / dt;
				float dh = (next_future_state.rect.height - future_active_state.rect.height) / dt;
				
				// Limit velocity to prevent overshooting
				float max_dx = future_active_state.rect.width * 0.5f;
				float max_dy = future_active_state.rect.height * 0.5f;
				dx = std::max(-max_dx, std::min(max_dx, dx));
				dy = std::max(-max_dy, std::min(max_dy, dy));
				dw = std::max(-max_dx, std::min(max_dx, dw));
				dh = std::max(-max_dy, std::min(max_dy, dh));
				
				// Extrapolate using velocity vector (Time = F - R)
				float time_diff = static_cast<float>(first_active_future_frame - R);
				cv::Rect2f extrapolated_rect;
				extrapolated_rect.x = future_active_state.rect.x - (dx * time_diff);
				extrapolated_rect.y = future_active_state.rect.y - (dy * time_diff);
				extrapolated_rect.width = future_active_state.rect.width - (dw * time_diff);
				extrapolated_rect.height = future_active_state.rect.height - (dh * time_diff);
				
				return scaleBoundingBox(extrapolated_rect, 1.2f);
			} else {
				// If no next future frame is found, use the current confirmed state
				return scaleBoundingBox(future_active_state.rect, 1.2f);
			}
		}
	}

	// 4. Handle missing frames during reverse masking period (confirmed but became extrapolated)
	if (will_be_recovered && exists_now && current_state.is_confirmed && current_state.is_extrapolated) {
		// Search for confirmed state just before masking started
		uint64_t last_active_frame = 0;
		TrackedObjectState left_state;
		bool found_left = false;
		
		for (auto it = buffer.begin(); it != buffer.end(); ++it) {
			if (it->frame_id < R) { // Look for frames strictly before R
				auto obj_it = it->objects.find(object_id);
				if (obj_it != it->objects.end()) {
					if (obj_it->second.is_confirmed && !obj_it->second.is_extrapolated) {
						found_left = true;
						left_state = obj_it->second;
						last_active_frame = it->frame_id;
					}
				}
			} else {
				break; // Optimization: stop searching once we reach or pass R
			}
		}

		TrackedObjectState right_state;
		if (found_left && historyManager->get_object_state(first_active_future_frame, object_id, right_state)) {
			if (out_identity_epoch)
				*out_identity_epoch = right_state.identityEpoch;
			float total_dist = static_cast<float>(first_active_future_frame - last_active_frame);
			float current_dist = static_cast<float>(R - last_active_frame);
			float weight = total_dist > 0 ? (current_dist / total_dist) : 0.0f;

			cv::Rect2f interpolated_rect;
			interpolated_rect.x = (1.0f - weight) * left_state.rect.x + weight * right_state.rect.x;
			interpolated_rect.y = (1.0f - weight) * left_state.rect.y + weight * right_state.rect.y;
			interpolated_rect.width = (1.0f - weight) * left_state.rect.width + weight * right_state.rect.width;
			interpolated_rect.height = (1.0f - weight) * left_state.rect.height + weight * right_state.rect.height;

			out_mode = "Interpolated";
			return interpolated_rect;
		}
	}

	// 5. Cannot recover - confirmed but extrapolated in current frame
	if (exists_now && current_state.is_confirmed && current_state.is_extrapolated) {
		out_mode = "Extrapolated";
		if (out_identity_epoch)
			*out_identity_epoch = current_state.identityEpoch;
		return current_state.rect; 
	}

	out_mode = "Ghost";
	return cv::Rect2f(0, 0, 0, 0);
}

static void log_mask_failure(struct detect_filter *tf, const char *reason)
{
	const auto now = std::chrono::steady_clock::now();
	if (tf->lastMaskFailureLog.time_since_epoch().count() == 0 ||
	    now - tf->lastMaskFailureLog >= std::chrono::seconds(5)) {
		obs_log(LOG_ERROR,
			"Mask texture generation failed (%s). Applying full-screen masking.",
			reason);
		tf->lastMaskFailureLog = now;
	}
}

static gs_texture_t *build_gpu_mask_texture(struct detect_filter *tf,
					    const std::vector<MaskRect> &rects, uint32_t width,
					    uint32_t height)
{
	constexpr float blurWorkPadding = 300.0f;
	tf->maskForceFullscreen = false;
	if (!tf->maskTexrender || !tf->maskShapeEffect) {
		tf->maskForceFullscreen = true;
		log_mask_failure(tf, "missing graphics resource");
		return nullptr;
	}

	std::vector<MaskVertex> vertices;
	try {
		vertices = buildMaskVertices(rects, (float)tf->maskingFeather, blurWorkPadding);
	} catch (const std::bad_alloc &) {
		tf->maskForceFullscreen = true;
		log_mask_failure(tf, "vertex allocation");
		return nullptr;
	}

	gs_vertbuffer_t *vertexBuffer = nullptr;
	if (!vertices.empty()) {
		gs_render_start(true);
		for (const MaskVertex &vertex : vertices) {
			gs_texcoord(vertex.u, vertex.v, 0);
			gs_texcoord(vertex.rectWidth, vertex.rectHeight, 1);
			gs_vertex2f(vertex.x, vertex.y);
		}
		vertexBuffer = gs_render_save();
		if (!vertexBuffer) {
			tf->maskForceFullscreen = true;
			log_mask_failure(tf, "vertex buffer creation");
			return nullptr;
		}
	}

	gs_texrender_reset(tf->maskTexrender);
	if (!gs_texrender_begin(tf->maskTexrender, width, height)) {
		if (vertexBuffer)
			gs_vertexbuffer_destroy(vertexBuffer);
		tf->maskForceFullscreen = true;
		log_mask_failure(tf, "render target begin");
		return nullptr;
	}

	struct vec4 clearColor;
	vec4_zero(&clearColor);
	gs_clear(GS_CLEAR_COLOR, &clearColor, 0.0f, 0);
	gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);

	if (vertexBuffer) {
		gs_eparam_t *feather =
			gs_effect_get_param_by_name(tf->maskShapeEffect, "feather_pixels");
		gs_eparam_t *workPadding = gs_effect_get_param_by_name(tf->maskShapeEffect,
								       "blur_work_padding_pixels");
		if (feather)
			gs_effect_set_float(feather, (float)tf->maskingFeather);
		if (workPadding)
			gs_effect_set_float(workPadding, blurWorkPadding);

		gs_blend_state_push();
		// Overlapping mask quads must add up. libobs disables blending when it renders
		// this filter through a color space conversion (HDR canvas or scRGB preview).
		gs_enable_blending(true);
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ONE);
		gs_load_vertexbuffer(vertexBuffer);
		while (gs_effect_loop(tf->maskShapeEffect, "Draw"))
			gs_draw(GS_TRIS, 0, 0);
		gs_load_vertexbuffer(nullptr);
		gs_blend_state_pop();
	}

	gs_texrender_end(tf->maskTexrender);
	if (vertexBuffer)
		gs_vertexbuffer_destroy(vertexBuffer);

	gs_texture_t *maskTexture = gs_texrender_get_texture(tf->maskTexrender);
	if (!maskTexture) {
		tf->maskForceFullscreen = true;
		log_mask_failure(tf, "render target texture");
	}
	return maskTexture;
}

// Draws sourceTex with the configured masking type. maskTexture may be null when
// tf->maskForceFullscreen is set, in which case the whole frame is masked.
static void draw_masked_output(struct detect_filter *tf, gs_texture_t *sourceTex, uint32_t width,
			       uint32_t height, gs_texture_t *maskTexture)
{
	auto applyMaskTexture = [&](gs_effect_t *effect) {
		if (!effect)
			return;
		gs_eparam_t *textureParam = gs_effect_get_param_by_name(effect, "mask_texture");
		gs_eparam_t *forceParam = gs_effect_get_param_by_name(effect, "force_full_mask");
		if (textureParam && maskTexture)
			gs_effect_set_texture(textureParam, maskTexture);
		if (forceParam)
			gs_effect_set_bool(forceParam, tf->maskForceFullscreen);
	};
	applyMaskTexture(tf->maskingEffect);
	applyMaskTexture(tf->kawaseBlurEffect);
	applyMaskTexture(tf->pixelateEffect);

	// Preserve the original frame for blending
	if (!tf->baseTexture || gs_texture_get_width(tf->baseTexture) != width ||
	    gs_texture_get_height(tf->baseTexture) != height) {
		if (tf->baseTexture)
			gs_texture_destroy(tf->baseTexture);
		tf->baseTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	}
	gs_copy_texture(tf->baseTexture, sourceTex);

	gs_texture_t *tex = sourceTex;
	bool destroy_tex = false;
	const char *technique_name = "Draw";
	gs_eparam_t *maskColorParam = gs_effect_get_param_by_name(tf->maskingEffect, "color");
	if (tf->maskingType == "output_mask") {
		technique_name = "DrawMask";
	} else if (tf->maskingType == "blur") {
		tex = blur_image(tf, width, height, maskTexture, sourceTex);
		destroy_tex = true; // blur_image creates a new texture
	} else if (tf->maskingType == "pixelate") {
		tex = pixelate_image(tf, width, height, maskTexture, (float)tf->maskingBlurRadius,
				     sourceTex);
		destroy_tex = true;
	} else if (tf->maskingType == "transparent") {
		technique_name = "DrawSolidColor";
		gs_effect_set_color(maskColorParam, 0);
	} else if (tf->maskingType == "solid_color") {
		technique_name = "DrawSolidColor";
		gs_effect_set_color(maskColorParam, tf->maskingColor);
	}

	gs_eparam_t *baseImageParam = gs_effect_get_param_by_name(tf->maskingEffect, "base_image");
	if (baseImageParam)
		gs_effect_set_texture(baseImageParam, tf->baseTexture);
	gs_eparam_t *imageParam = gs_effect_get_param_by_name(tf->maskingEffect, "image");
	gs_effect_set_texture(imageParam, tex);

	while (gs_effect_loop(tf->maskingEffect, technique_name)) {
		gs_draw_sprite(tex, 0, 0, 0);
	}

	if (destroy_tex) {
		gs_texture_destroy(tex);
	}
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	// FPS-based update for lookahead and preemptive masking frames (0.5s / 0.25s)
	double fps = 60.0;
	struct obs_video_info ovi;
	if (obs_get_video_info(&ovi) && ovi.fps_den > 0) {
		fps = (double)ovi.fps_num / (double)ovi.fps_den;
	}
	int newLookahead = (int)round(0.5 * fps);
	// Compensate for the tracker's confirmation delay (minHitFrames) so the user gets a true 0.25s preemptive mask
	int newPreemptive = (int)round(0.25 * fps) + tf->minHitFrames;

	if (tf->lookaheadDelayFrames != newLookahead || tf->preemptiveMaskingFrames != newPreemptive) {
		std::lock_guard<std::mutex> lock(tf->audioMutex);
		tf->lookaheadDelayFrames = newLookahead;
		tf->preemptiveMaskingFrames = newPreemptive;
		tf->resetAudio = true;
	}

	const bool modelReady = tf->modelLoaded;
	if (tf->isDisabled || !tf->effectsLoaded || (!modelReady && !tf->maskingEnabled)) {
		clear_all_buffers(tf, true);
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}
	uint32_t width = obs_source_get_base_width(target);
	uint32_t height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	gs_texrender_reset(tf->texrender);
	if (!gs_texrender_begin(tf->texrender, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}
	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);

	if (!modelReady) {
		// No detector yet (still loading or failed to load): mask the whole frame
		// instead of passing it through unmasked.
		clear_all_buffers(tf, true);
		gs_texture_t *liveTex = gs_texrender_get_texture(tf->texrender);
		if (!liveTex) {
			obs_source_skip_video_filter(tf->source);
			return;
		}
		tf->maskForceFullscreen = true;
		draw_masked_output(tf, liveTex, width, height, nullptr);
		return;
	}

	gs_texture_t *delayed_tex = nullptr;
	uint64_t renderFrameId = 0;
	uint64_t maskRenderFrameId = 0;
	int totalDelay = tf->videoDelayFrames + tf->lookaheadDelayFrames;

	if (!tf->tickRendered) {
		tf->tickRendered = true;

		gs_texture_t *render_tex = gs_texrender_get_texture(tf->texrender);
		bool run_gpu = false;

#ifdef _WIN32
		const InferenceFrameRequirements frameRequirements =
			get_inference_frame_requirements(tf);
		const bool zeroCopyEligible =
			tf->useDirectML && render_tex && tf->modelSupportsZeroCopy &&
			tf->zeroCopyRuntimeHealthy && !requiresOwnedInferenceFrame(frameRequirements);
		if (zeroCopyEligible) {
			std::unique_lock<std::mutex> inferenceLock(tf->inferenceMutex,
								   std::try_to_lock);
			// The model can be swapped by the IO thread, so it is only touched while
			// modelMutex is held.
			std::unique_lock<std::mutex> modelLock(tf->modelMutex, std::defer_lock);
			if (!inferenceLock.owns_lock() || tf->isInferencing ||
			    tf->inferenceFrameReady) {
				// Keep the render thread non-blocking while inference owns the GPU resources.
				run_gpu = true;
			} else if (modelLock.try_lock() && tf->onnxruntimemodel) {
				if (!tf->onnxruntimemodel->is_gpu_pipeline_ready_for((int)width,
										     (int)height)) {
					ID3D11Texture2D *d3d11Tex =
						(ID3D11Texture2D *)gs_texture_get_obj(render_tex);
					if (d3d11Tex) {
						ID3D11Device *d3d11Device = nullptr;
						d3d11Tex->GetDevice(&d3d11Device);
						if (d3d11Device) {
							if (!tf->onnxruntimemodel->initialize_gpu(
								    d3d11Device, width, height))
								tf->zeroCopyRuntimeHealthy = false;
							d3d11Device->Release();
						}
					}
				}

				tf->zeroCopyPipelineReady = tf->onnxruntimemodel->is_gpu_pipeline_ready();
				if (tf->onnxruntimemodel->is_gpu_pipeline_ready_for((int)width,
										    (int)height)) {
					if (tf->onnxruntimemodel->copy_d3d11_texture(render_tex)) {
						run_gpu = true;
						tf->inferenceJob = InferenceJob{};
						tf->inferenceJob.mode = InferenceMode::GPUZeroCopy;
						tf->inferenceJob.sourceWidth = width;
						tf->inferenceJob.sourceHeight = height;
						tf->inferenceJob.frameId = tf->currentFrameId + 1;
						tf->lastQueuedInferenceFrameId =
							tf->inferenceJob.frameId;
						tf->isInferencing = true;
						tf->inferenceFrameReady = true;
						tf->inferenceCV.notify_one();
					}
				}
			}
		}
#endif

		if (!run_gpu) {
			if (tf->stagesurface) {
				uint32_t stagesurf_width = gs_stagesurface_get_width(tf->stagesurface);
				uint32_t stagesurf_height = gs_stagesurface_get_height(tf->stagesurface);
				if (stagesurf_width != width || stagesurf_height != height) {
					gs_stagesurface_destroy(tf->stagesurface);
					tf->stagesurface = nullptr;
				}
			}
			if (!tf->stagesurface) {
				tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
			}
			gs_stage_texture(tf->stagesurface, render_tex);
			uint8_t *video_data;
			uint32_t linesize;
			if (gs_stagesurface_map(tf->stagesurface, &video_data, &linesize)) {
				cv::Mat ownedFrame =
					copyMappedBGRA(video_data, linesize, width, height);
				gs_stagesurface_unmap(tf->stagesurface);
				if (ownedFrame.empty()) {
					if (tf->source)
						obs_source_skip_video_filter(tf->source);
					return;
				}
				{
					std::lock_guard<std::mutex> lock(tf->inputBGRALock);
					tf->inputBGRA = std::move(ownedFrame);
					tf->inputBGRAFrameId = tf->currentFrameId + 1;
				}
			} else {
				if (tf->source) {
					obs_source_skip_video_filter(tf->source);
				}
				return;
			}
		}

		// ---- VIDEO DELAY LOGIC ----
		tf->currentFrameId++;

		while ((int)tf->delayedTextures.size() > totalDelay) {
			gs_texture_t *old_tex = tf->delayedTextures.front();
			tf->delayedTextures.pop_front();
			tf->texturePool.push_back(old_tex);
		}
		while ((int)tf->delayedFrameIds.size() > totalDelay) {
			tf->delayedFrameIds.pop_front();
		}

		gs_texture_t *copy_tex = nullptr;
		if (totalDelay > 0) {
			for (auto it = tf->texturePool.begin(); it != tf->texturePool.end(); ++it) {
				if (gs_texture_get_width(*it) == width && gs_texture_get_height(*it) == height) {
					copy_tex = *it;
					tf->texturePool.erase(it);
					break;
				}
			}
			if (!copy_tex) {
				copy_tex = gs_texture_create(width, height, GS_BGRA, 1, nullptr, GS_RENDER_TARGET);
			}
			if (copy_tex) {
				gs_copy_texture(copy_tex, render_tex);
				tf->delayedTextures.push_back(copy_tex);
				tf->delayedFrameIds.push_back(tf->currentFrameId);
			}
		} else {
			for (auto tex : tf->delayedTextures) gs_texture_destroy(tex);
			tf->delayedTextures.clear();
			tf->delayedFrameIds.clear();
			for (auto tex : tf->texturePool) gs_texture_destroy(tex);
			tf->texturePool.clear();
		}

		delayed_tex = render_tex;
		renderFrameId = tf->currentFrameId;
		maskRenderFrameId = tf->currentFrameId;
		if (totalDelay > 0 && !tf->delayedTextures.empty()) {
			delayed_tex = tf->delayedTextures.front();
			renderFrameId = tf->delayedFrameIds.front();
			if ((int)tf->delayedFrameIds.size() > tf->videoDelayFrames) {
				maskRenderFrameId = tf->delayedFrameIds[tf->videoDelayFrames];
			} else {
				maskRenderFrameId = tf->delayedFrameIds.back();
			}
		}
		// ---------------------------

		tf->cachedRenderTex = delayed_tex;
		tf->cachedRenderFrameId = renderFrameId;
		tf->cachedMaskRenderFrameId = maskRenderFrameId;
	} else {
		delayed_tex = tf->cachedRenderTex;
		renderFrameId = tf->cachedRenderFrameId;
		maskRenderFrameId = tf->cachedMaskRenderFrameId;
	}

	// if preview, masking, debug stats, or debug mode is enabled, render the image
	if (tf->preview || tf->maskingEnabled || tf->debugMode || tf->enableFaceStats) {
		cv::Mat outputBGRA;
		cv::Mat localOutputBGRA;
		bool skip_preview = false;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->previewHistory.empty()) {
				skip_preview = true;
			} else {
				auto it = tf->previewHistory.upper_bound(maskRenderFrameId);
				if (it != tf->previewHistory.begin()) {
					--it; // Closest available frame <= maskRenderFrameId
					localOutputBGRA = it->second;
				} else {
					localOutputBGRA = tf->previewHistory.begin()->second;
				}
				
				if ((uint32_t)localOutputBGRA.cols != width ||
				    (uint32_t)localOutputBGRA.rows != height) {
					skip_preview = true;
				}
			}
		}
		if (!skip_preview) {
			outputBGRA = localOutputBGRA.clone();
			
			// Mask status display ([Confirmed] / [Reverse Masking] etc.) - Depends on preview option
			if (tf->preview && tf->stateHistory) {
				std::set<int> active_ids = tf->stateHistory->get_all_active_ids(maskRenderFrameId, tf->currentFrameId);
				for (int id : active_ids) {
					std::string mode_str;
					cv::Rect2f rect = getFinalRenderBox(id, maskRenderFrameId, tf->stateHistory.get(), tf->currentFrameId, tf->preemptiveMaskingFrames, mode_str);
					if (rect.width > 0 && rect.height > 0) {
						std::string text = "[" + mode_str + "]";
						cv::Scalar color(0, 255, 255, 255); // Yellow
						if (mode_str == "Confirmed") color = cv::Scalar(0, 255, 0, 255); // Green
						else if (mode_str == "Interpolated") color = cv::Scalar(255, 165, 0, 255); // Orange
						else if (mode_str == "Reverse Masking") color = cv::Scalar(255, 0, 255, 255); // Magenta
						else if (mode_str == "Extrapolated") color = cv::Scalar(0, 0, 255, 255); // Red
						
						int baseline = 0;
						cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
						cv::Point textOrg((int)rect.x, (int)rect.y - 10);
						if (textOrg.y < textSize.height) textOrg.y = (int)(rect.y + rect.height) + textSize.height + 10;
						
						// Draw text outline for visibility
						cv::putText(outputBGRA, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255), 4, cv::LINE_AA);
						cv::putText(outputBGRA, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
					}
				}
			}
		}

		gs_texture_t *tex = delayed_tex;

		if (tf->maskingEnabled) {
			std::vector<MaskRect> maskRects;
			int current_excluded_count = 0;
			bool maskCollectionFailed = false;
			auto faceSnapshot = tf->faceState.snapshot();
			try {
				if (tf->stateHistory) {
					std::set<int> active_ids =
						tf->stateHistory->get_all_active_ids(
							maskRenderFrameId, tf->currentFrameId);
					maskRects.reserve(active_ids.size());
					for (int id : active_ids) {
						std::string mode_str;
						uint64_t identityEpoch = 0;
						cv::Rect2f rect = getFinalRenderBox(
							id, maskRenderFrameId,
							tf->stateHistory.get(), tf->currentFrameId,
							tf->preemptiveMaskingFrames, mode_str,
							&identityEpoch);
						if (rect.width <= 0 || rect.height <= 0 ||
						    mode_str == "Ghost")
							continue;

						if (tf->enableFaceExclusion && faceSnapshot &&
						    FaceStateStore::isExempt(
							    *faceSnapshot, (uint64_t)id,
							    identityEpoch,
							    tf->preserveFaceExclusionOnRecovery)) {
							current_excluded_count++;
							continue;
						}

						float expansion = 0.0f;
						if (tf->maskingDynamicExpansion) {
							expansion =
								tf->maskingDynamicExpansionBase +
								(rect.height *
								 tf->maskingDynamicExpansionRatio);
						} else if (tf->maskingDilateIterations > 0) {
							expansion = (float)tf->maskingDilateIterations;
						}
						if (expansion > 0.0f) {
							rect.x -= expansion;
							rect.y -= expansion;
							rect.width += expansion * 2.0f;
							rect.height += expansion * 2.0f;
						}
						maskRects.push_back(
							{rect.x, rect.y, rect.width, rect.height});
					}
				}
			} catch (const std::bad_alloc &) {
				maskCollectionFailed = true;
				tf->maskForceFullscreen = true;
				log_mask_failure(tf, "rectangle allocation");
			}

			if (tf->enableFaceExclusion) {
				if (tf->last_excluded_count != current_excluded_count) {
					tf->last_excluded_count = current_excluded_count;
					emit_face_exclusion_update(current_excluded_count);
				}
			} else {
				if (tf->last_excluded_count != 0) {
					tf->last_excluded_count = 0;
					emit_face_exclusion_update(0);
				}
			}

			// Without detector coverage of this frame (warm-up after a reset, or an
			// inference stall longer than the look-ahead) an empty mask would show
			// everyone, so the whole frame is masked instead.
			const uint64_t coverageStart = tf->coverageStartFrameId;
			const bool covered = coverageStart != 0 && coverageStart <= maskRenderFrameId &&
					     tf->lastCoveredFrameId >= maskRenderFrameId;
			gs_texture_t *maskTexture =
				(maskCollectionFailed || !covered)
					? nullptr
					: build_gpu_mask_texture(tf, maskRects, width, height);
			if (!covered ||
			    shouldForceFullscreenMask(!maskCollectionFailed, maskTexture != nullptr))
				tf->maskForceFullscreen = true;
			draw_masked_output(tf, delayed_tex, width, height, maskTexture);
		} else {
			gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
			gs_eparam_t *defaultImageParam = gs_effect_get_param_by_name(default_effect, "image");
			if (defaultImageParam) gs_effect_set_texture(defaultImageParam, tex);
			while (gs_effect_loop(default_effect, "Draw")) {
				gs_draw_sprite(tex, 0, 0, 0);
			}
		}
		
		if (!skip_preview && (tf->preview || tf->debugMode || tf->enableFaceStats)) {
			if (!tf->previewTexture || gs_texture_get_width(tf->previewTexture) != width || gs_texture_get_height(tf->previewTexture) != height) {
				if (tf->previewTexture) gs_texture_destroy(tf->previewTexture);
				tf->previewTexture = gs_texture_create(width, height, GS_BGRA, 1,
							      (const uint8_t **)&outputBGRA.data, GS_DYNAMIC);
			} else {
				gs_texture_set_image(tf->previewTexture, outputBGRA.data, width * 4, false);
			}
			
			gs_blend_state_push();
			gs_enable_blending(true);
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
	} else if (totalDelay > 0) {
		gs_texture_t *tex = delayed_tex;
		gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		gs_eparam_t *defaultImageParam = gs_effect_get_param_by_name(default_effect, "image");
		if (defaultImageParam) gs_effect_set_texture(defaultImageParam, tex);
		while (gs_effect_loop(default_effect, "Draw")) {
			gs_draw_sprite(tex, 0, 0, 0);
		}
	} else {
		// Just render the original video
		gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
		gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		obs_source_skip_video_filter(tf->source);
	}
	return;
}

struct obs_audio_data *detect_filter_audio(void *data, struct obs_audio_data *audio)
{
	struct detect_filter *tf = reinterpret_cast<struct detect_filter *>(data);
	// Video is only delayed while a model is loaded, so audio must not be delayed either.
	if (!tf || tf->isDisabled || !tf->modelLoaded || !obs_source_enabled(tf->source))
		return audio;

	std::lock_guard<std::mutex> lock(tf->audioMutex);

	uint32_t sample_rate = 48000;
	struct obs_audio_info audio_info;
	if (obs_get_audio_info(&audio_info)) {
		sample_rate = audio_info.samples_per_sec;
	}

	double fps = 60.0;
	struct obs_video_info ovi;
	if (obs_get_video_info(&ovi) && ovi.fps_den > 0) {
		fps = (double)ovi.fps_num / (double)ovi.fps_den;
	}

	uint64_t target_delay_samples = (uint64_t)round((tf->videoDelayFrames + tf->lookaheadDelayFrames) * sample_rate / fps);
	size_t target_delay_bytes = target_delay_samples * sizeof(float);

	if (tf->resetAudio) {
		for (int c = 0; c < MAX_AUDIO_CHANNELS; c++) {
			if (audio->data[c]) {
				float *samples = reinterpret_cast<float*>(audio->data[c]);
				float start_val = tf->lastSamples[c][1];
				uint32_t fade_len = (audio->frames >= 64) ? 64 : audio->frames;
				for (uint32_t i = 0; i < fade_len; i++) {
					float ratio = (float)i / (float)fade_len;
					samples[i] = start_val * (1.0f - ratio);
				}
				if (audio->frames > fade_len) {
					std::memset(samples + fade_len, 0, (audio->frames - fade_len) * sizeof(float));
				}
				tf->lastSamples[c][0] = 0.0f;
				tf->lastSamples[c][1] = 0.0f;
			}
		}

		for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
			circlebuf_free(&tf->audioBuffers[i]);
			circlebuf_init(&tf->audioBuffers[i]);
		}

		if (target_delay_bytes > 0) {
			std::vector<uint8_t> zeros(target_delay_bytes, 0);
			for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
				circlebuf_push_back(&tf->audioBuffers[i], zeros.data(), target_delay_bytes);
			}
		}

		tf->resetAudio = false;
	}

	if (target_delay_bytes == 0) {
		return audio;
	}

	for (int c = 0; c < MAX_AUDIO_CHANNELS; c++) {
		if (!audio->data[c])
			continue;

		size_t frame_bytes = audio->frames * sizeof(float);

		if (audio->frames > 1) {
			float *samples = reinterpret_cast<float*>(audio->data[c]);
			tf->lastSamples[c][0] = samples[audio->frames - 2];
			tf->lastSamples[c][1] = samples[audio->frames - 1];
		}

		circlebuf_push_back(&tf->audioBuffers[c], audio->data[c], frame_bytes);

		if (tf->audioBuffers[c].size >= (target_delay_bytes + frame_bytes)) {
			circlebuf_pop_front(&tf->audioBuffers[c], audio->data[c], frame_bytes);
		} else {
			std::memset(audio->data[c], 0, frame_bytes);
		}
	}

	return audio;
}
