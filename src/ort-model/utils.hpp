#ifndef _EdgeYOLO_CPP_UTILS_HPP
#define _EdgeYOLO_CPP_UTILS_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "types.hpp"

static std::vector<std::string> read_class_labels_file(file_name_t file_name)
{
	std::vector<std::string> class_names;
	std::ifstream ifs(file_name);
	std::string buff;
	if (ifs.fail()) {
		return class_names;
	}
	while (getline(ifs, buff)) {
		if (buff == "")
			continue;
		class_names.push_back(buff);
	}
	return class_names;
}

const float color_list[80][3] = {
	{0.000f, 0.447f, 0.741f}, {0.850f, 0.325f, 0.098f}, {0.929f, 0.694f, 0.125f},
	{0.494f, 0.184f, 0.556f}, {0.466f, 0.674f, 0.188f}, {0.301f, 0.745f, 0.933f},
	{0.635f, 0.078f, 0.184f}, {0.300f, 0.300f, 0.300f}, {0.600f, 0.600f, 0.600f},
	{1.000f, 0.000f, 0.000f}, {1.000f, 0.500f, 0.000f}, {0.749f, 0.749f, 0.000f},
	{0.000f, 1.000f, 0.000f}, {0.000f, 0.000f, 1.000f}, {0.667f, 0.000f, 1.000f},
	{0.333f, 0.333f, 0.000f}, {0.333f, 0.667f, 0.000f}, {0.333f, 1.000f, 0.000f},
	{0.667f, 0.333f, 0.000f}, {0.667f, 0.667f, 0.000f}, {0.667f, 1.000f, 0.000f},
	{1.000f, 0.333f, 0.000f}, {1.000f, 0.667f, 0.000f}, {1.000f, 1.000f, 0.000f},
	{0.000f, 0.333f, 0.500f}, {0.000f, 0.667f, 0.500f}, {0.000f, 1.000f, 0.500f},
	{0.333f, 0.000f, 0.500f}, {0.333f, 0.333f, 0.500f}, {0.333f, 0.667f, 0.500f},
	{0.333f, 1.000f, 0.500f}, {0.667f, 0.000f, 0.500f}, {0.667f, 0.333f, 0.500f},
	{0.667f, 0.667f, 0.500f}, {0.667f, 1.000f, 0.500f}, {1.000f, 0.000f, 0.500f},
	{1.000f, 0.333f, 0.500f}, {1.000f, 0.667f, 0.500f}, {1.000f, 1.000f, 0.500f},
	{0.000f, 0.333f, 1.000f}, {0.000f, 0.667f, 1.000f}, {0.000f, 1.000f, 1.000f},
	{0.333f, 0.000f, 1.000f}, {0.333f, 0.333f, 1.000f}, {0.333f, 0.667f, 1.000f},
	{0.333f, 1.000f, 1.000f}, {0.667f, 0.000f, 1.000f}, {0.667f, 0.333f, 1.000f},
	{0.667f, 0.667f, 1.000f}, {0.667f, 1.000f, 1.000f}, {1.000f, 0.000f, 1.000f},
	{1.000f, 0.333f, 1.000f}, {1.000f, 0.667f, 1.000f}, {0.333f, 0.000f, 0.000f},
	{0.500f, 0.000f, 0.000f}, {0.667f, 0.000f, 0.000f}, {0.833f, 0.000f, 0.000f},
	{1.000f, 0.000f, 0.000f}, {0.000f, 0.167f, 0.000f}, {0.000f, 0.333f, 0.000f},
	{0.000f, 0.500f, 0.000f}, {0.000f, 0.667f, 0.000f}, {0.000f, 0.833f, 0.000f},
	{0.000f, 1.000f, 0.000f}, {0.000f, 0.000f, 0.167f}, {0.000f, 0.000f, 0.333f},
	{0.000f, 0.000f, 0.500f}, {0.000f, 0.000f, 0.667f}, {0.000f, 0.000f, 0.833f},
	{0.000f, 0.000f, 1.000f}, {0.000f, 0.000f, 0.000f}, {0.143f, 0.143f, 0.143f},
	{0.286f, 0.286f, 0.286f}, {0.429f, 0.429f, 0.429f}, {0.571f, 0.571f, 0.571f},
	{0.714f, 0.714f, 0.714f}, {0.857f, 0.857f, 0.857f}, {0.000f, 0.447f, 0.741f},
	{0.314f, 0.717f, 0.741f}, {0.50f, 0.5f, 0.0f}};

static void draw_objects(cv::Mat bgr, const std::vector<Object> &objects,
			 const std::vector<std::string> &class_names)
{

	for (size_t i = 0; i < objects.size(); i++) {
		const Object &obj = objects[i];

		int color_index = obj.label % 80;
		cv::Scalar color = cv::Scalar(color_list[color_index][0] * 255.0,
					      color_list[color_index][1] * 255.0,
					      color_list[color_index][2] * 255.0, 255.0);
		// Reproduce original bug/logic where alpha 0 was included in the mean
		float c_mean = (color_list[color_index][0] + color_list[color_index][1] + color_list[color_index][2]) / 4.0f;
		cv::Scalar txt_color;
		if (c_mean > 0.5) {
			txt_color = cv::Scalar(0, 0, 0, 255);
		} else {
			txt_color = cv::Scalar(255, 255, 255, 255);
		}

		if (obj.isUnconfirmed) {
			color = cv::Scalar(128, 128, 128, 255); // Gray
			txt_color = cv::Scalar(255, 255, 255, 255);
		} else if (obj.isExempt) {
			color = cv::Scalar(0, 0, 255, 255); // Red
			txt_color = cv::Scalar(255, 255, 255, 255);
		}

		cv::rectangle(bgr, obj.rect, color, 2);

		char base_text[128];
		snprintf(base_text, sizeof(base_text), "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

		std::string status_text = "";
		cv::Scalar status_bg_color = color * 0.7;
		status_bg_color[3] = 255;
		cv::Scalar status_txt_color = txt_color;

		if (!obj.customText.empty()) {
			status_text = obj.customText;
			if (status_text.find("Exempted") != std::string::npos) {
				status_bg_color = cv::Scalar(0, 180, 0, 255); // Green
				status_txt_color = cv::Scalar(255, 255, 255, 255);
			} else if (status_text.find("Not Me") != std::string::npos) {
				status_bg_color = cv::Scalar(0, 0, 200, 255); // Red
				status_txt_color = cv::Scalar(255, 255, 255, 255);
			} else if (status_text.find("Checking") != std::string::npos) {
				status_bg_color = cv::Scalar(0, 165, 255, 255); // Orange
				status_txt_color = cv::Scalar(255, 255, 255, 255);
			} else if (status_text.find("Too Small") != std::string::npos) {
				status_bg_color = cv::Scalar(128, 128, 128, 255); // Gray
				status_txt_color = cv::Scalar(255, 255, 255, 255);
			}
		} else if (obj.isUnconfirmed) {
			status_text = "Unconfirmed / 미확정";
			status_bg_color = cv::Scalar(128, 128, 128, 255);
			status_txt_color = cv::Scalar(255, 255, 255, 255);
		} else if (obj.isExempt) {
			status_text = "Exempted / 제외됨";
			status_bg_color = cv::Scalar(0, 180, 0, 255);
			status_txt_color = cv::Scalar(255, 255, 255, 255);
		}

		int baseLine = 0;
		cv::Size base_size = cv::getTextSize(base_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

		int x = (int)(obj.rect.x);
		int y = (int)(obj.rect.y + 1);
		if (y > bgr.rows) y = bgr.rows;

		// Draw base_text
		cv::Scalar base_bg_color = color * 0.7;
		base_bg_color[3] = 255;
		cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(base_size.width, base_size.height + baseLine)), base_bg_color, -1);
		cv::putText(bgr, base_text, cv::Point(x, y + base_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);

		int current_y_offset = base_size.height + baseLine;

		// Draw status_text (face recognition status) if any
		if (!status_text.empty()) {
			int statusBaseLine = 0;
			cv::Size status_size = cv::getTextSize(status_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &statusBaseLine);
			cv::rectangle(bgr, cv::Rect(cv::Point(x, y + current_y_offset), cv::Size(status_size.width, status_size.height + statusBaseLine)), status_bg_color, -1);
			cv::putText(bgr, status_text, cv::Point(x, y + current_y_offset + status_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, status_txt_color, 1);
			current_y_offset += status_size.height + statusBaseLine;
		}

		// Draw trackingState if any
		if (!obj.trackingState.empty()) {
			cv::Scalar track_bg_color = cv::Scalar(128, 128, 128, 255); // Default Gray (BGR)
			cv::Scalar track_txt_color = cv::Scalar(255, 255, 255, 255); // White text
			
			if (obj.trackingState.find("Stable") != std::string::npos) {
				track_bg_color = cv::Scalar(0, 150, 0, 255); // Green (BGR)
			} else if (obj.trackingState.find("New") != std::string::npos) {
				track_bg_color = cv::Scalar(0, 140, 255, 255); // Orange (BGR)
			} else if (obj.trackingState.find("Unseen") != std::string::npos) {
				track_bg_color = cv::Scalar(100, 100, 100, 255); // Dark Gray (BGR)
			} else if (obj.trackingState.find("Recovered") != std::string::npos) {
				track_bg_color = cv::Scalar(180, 0, 180, 255); // Purple (BGR)
			}
			
			int trackBaseLine = 0;
			cv::Size track_size = cv::getTextSize(obj.trackingState, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &trackBaseLine);
			cv::rectangle(bgr, cv::Rect(cv::Point(x, y + current_y_offset), cv::Size(track_size.width, track_size.height + trackBaseLine)), track_bg_color, -1);
			cv::putText(bgr, obj.trackingState, cv::Point(x, y + current_y_offset + track_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, track_txt_color, 1);
			current_y_offset += track_size.height + trackBaseLine;
		}

		// write the id of the object (no background)
		char id_text[64];
		snprintf(id_text, sizeof(id_text), "ID: %d", (int)obj.id);
		int idBaseLine = 0;
		cv::Size id_size = cv::getTextSize(id_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &idBaseLine);
		// Draw ID text with a thin black border for readability on light backgrounds
		cv::putText(bgr, id_text, cv::Point(x, y + current_y_offset + id_size.height),
			    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0, 255), 2, cv::LINE_AA);
		cv::putText(bgr, id_text, cv::Point(x, y + current_y_offset + id_size.height),
			    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1, cv::LINE_AA);
	}
}

#endif
