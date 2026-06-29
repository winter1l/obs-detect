# ⚠️ Stalled ⚠️ This project is not under active development

## OBS Detect - Object Detection and Masking Filter

<div align="center">

[![GitHub](https://img.shields.io/github/license/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/occ-ai/obs-detect/push.yaml)](https://github.com/occ-ai/obs-detect/actions/workflows/push.yaml)
[![Total downloads](https://img.shields.io/github/downloads/occ-ai/obs-detect/total)](https://github.com/occ-ai/obs-detect/releases)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/releases)
[![Discord](https://img.shields.io/discord/1200229425141252116)](https://discord.gg/KbjGU2vvUz)

</div>

A plugin for [OBS Studio](https://obsproject.com/) that allows you to detect many types of objects in any source, track them and apply masking.

If you like this work, which is given to you completely free of charge, please consider supporting it by sponsoring us on GitHub:

- https://github.com/sponsors/royshil
- https://github.com/sponsors/umireon

This work uses the great contributions from [EdgeYOLO-ROS](https://github.com/fateshelled/EdgeYOLO-ROS) and [PINTO-Model-Zoo](https://github.com/PINTO0309/PINTO_model_zoo). The Hungarian algorithm is taken from https://github.com/Gluttton/munkres-cpp under the GPLv2 license.

## 🌟 Major Updates (Since `a4f4928`)

> **🤖 AI-Driven Development**: 본 리포지토리의 주요 업데이트 및 알고리즘 고도화 작업은 전적으로 사용자 피드백과 AI(LLM) 어시스턴트와의 페어 프로그래밍(Pair Programming)을 통해 설계, 코딩 및 디버깅되었습니다.

본 리포지토리는 원본(`a4f4928`) 버전에서 크게 발전하여 다음과 같은 핵심 기능들이 추가 및 개선되었습니다.

### 1. 얼굴 기반 특정 인물 추적 제외 (Face-based Personal Exclusion)
- **Zero-training 얼굴 매칭**: `YuNet` (얼굴 탐지)과 `SFace` (얼굴 특징 추출 및 비교) 모델을 결합하여, 제외할 인물(본인 등)의 정면 사진 파일만 지정하면 해당 인물을 추적 및 마스킹 대상에서 실시간으로 제외합니다.
- **백그라운드 비동기 처리**: 무거운 얼굴 특징 추출 및 매칭 연산을 렌더링 스레드가 아닌 전용 백그라운드 스레드에서 비동기 처리하여 OBS의 60FPS 렌더링 프레임 저하를 완벽하게 방어합니다.
- **다중 참조 이미지 지원**: 한 장의 사진뿐만 아니라 여러 장의 참조 이미지(Reference Images)를 디렉토리로 지정하여 얼굴 인식률을 극대화할 수 있습니다.

### 2. 향상된 트래킹 알고리즘 및 고스트 복구 (Advanced Tracking & Ghost Recovery)
- **Ghost Recovery (고스트 복구)**: 대상이 카메라의 급격한 무빙이나 가려짐으로 인해 일시적으로 탐지되지 않더라도, 기존 궤적과 중심점 거리를 비교하는 공간 매칭을 통해 즉시 대상을 되찾고 1프레임의 지연 없이 순간이동하여 추적을 재개합니다.
- **Dynamic Kalman Filter Noise**: 추적 대상의 화면 내 크기 비율(area ratio)에 따라 칼만 필터의 측정 노이즈(Measurement Noise)를 실시간으로 다이내믹하게 조절합니다. 전신 등 큰 대상은 부드럽게 쫓아가며, 얼굴 등 1% 미만의 작은 대상은 과거 관성을 버리고 즉각적으로 빠릿빠릿하게(Responsive) 추적합니다.
- **텔레포트 관성 보존**: 대상을 잃어버렸다가 복구하는 과정에서 칼만 필터 예측이 튀지 않도록 속도 관성을 보존하여 매끄러운 카메라 무빙을 지원합니다.

### 3. 멀티 모델 지원 및 호환성 보장
- **다양한 Face 모델 지원**: `YOLOv8n-Face`, `YOLOv8s-Face`, `YuNet` 모델을 지원합니다.
- **YOLO Fallback 시스템**: 눈코입 랜드마크(Landmarks)를 출력하지 못하는 YOLOv8-Face 계열 모델(Lindevs 포맷) 사용 시 SFace 정렬(Align)이 실패하는 현상을 막기 위해, 백그라운드에서 임시로 YuNet을 돌려 진짜 눈코입 좌표를 찰떡같이 추출해주는 Fallback 메커니즘을 적용했습니다. 어떠한 모델을 사용해도 높은 인식률(0.7 이상)을 보장합니다.

### 4. 최적화 및 디버깅 UI (Optimization & Debugging)
- **렌더링 및 동기화 모드**: 고사양 PC를 위해 1프레임의 지연조차 허용하지 않는 Zero-Delay Sync Mode 기능과, 텍스처 복사 오버헤드를 최소화한 GPU 최적화가 포함되어 있습니다.
- **직관적인 상태 오버레이**: 화면에 추적 박스의 현재 상태(`New`, `Stable`, `Recovered`, `Unseen`)를 텍스트와 색상 코드로 명확하게 표시해 주어 플러그인 작동 상황을 즉각적으로 파악할 수 있습니다.
- **마스크 동적 확장**: 단순 `cv::dilate`를 쓰지 않고 해상도 비례 수학적 크기 확장을 통해 마스킹 처리 시 CPU 부하를 제거했습니다.
- **한국어 지원**: `ko-KR.ini` 언어팩을 완벽하게 업데이트하여 번역 품질을 대폭 향상시켰습니다.

## Usage

<div align="center">
<a href="https://youtu.be/LrbUrvaGreQ"><img width="40%" src="https://github.com/occ-ai/obs-detect/assets/441170/b8e7367e-c1b0-4c7e-b0df-af45ead87199" /></a>&nbsp;
<a href="https://youtu.be/zmdq1bPVYs0"><img width="40%" src="https://github.com/occ-ai/obs-detect/assets/441170/2eb08589-1695-4a40-877e-4985c2b5270f" /></a>
</div>

- Add the "Detect" filter to any source with an image (Media, Browser, VLC, Image, etc.)
- Enable "Masking" or "Tracking"

Use Detect to track your pet, or blur out people in your video!

More information and usage tutorials to follow soon.

## Features

Current features:

- Detect over 80 categories of objects, using an efficient model ([EdgeYOLO](https://github.com/LSH9832/edgeyolo))
- 3 Model sizes: Small, Medium and Large
- Face detection model, fast and efficient ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet))
- Load custom ONNX detection models from disk
- Filter by: Minimal Detection confidence, Object category (e.g. only "Person"), Object Minimal Size
- Masking: Blur, Pixelate, Solid color, Transparent, output binary mask (combine with other plugins!)
- Tracking: Single object / Biggest / Oldest / All objects, Zoom factor, smooth transition
- SORT algorithm for tracking smoothness and continuity
- Save detections to file in real-time, for integrations e.g. with Streamer.bot

Roadmap features:
- Precise object mask, beyond bounding box
- Multiple object category selection (e.g. Dog + Cat + Duck)
- Make available detection information for other plugins through settings

## Train and use a custom detection model

Follow the instructions in [docs/train_model.md](docs/train_model.md) to train and use your own custom model.

## Building

The plugin was built and tested on Mac OSX  (Intel & Apple silicon), Windows and Linux.

Start by cloning this repo to a directory of your choice.

### Mac OSX

Using the CI pipeline scripts, locally you would just call the zsh script. By default this builds a universal binary for both Intel and Apple Silicon. To build for a specific architecture please see `.github/scripts/.build.zsh` for the `-arch` options.

```sh
$ ./.github/scripts/build-macos -c Release
```

#### Install
The above script should succeed and the plugin files (e.g. `obs-ocr.plugin`) will reside in the `./release/Release` folder off of the root. Copy the `.plugin` file to the OBS directory e.g. `~/Library/Application Support/obs-studio/plugins`.

To get `.pkg` installer file, run for example
```sh
$ ./.github/scripts/package-macos -c Release
```
(Note that maybe the outputs will be in the `Release` folder and not the `install` folder like `pakage-macos` expects, so you will need to rename the folder from `build_x86_64/Release` to `build_x86_64/install`)

### Linux (Ubuntu)

Use the CI scripts again
```sh
$ ./.github/scripts/build-linux.sh
```

Copy the results to the standard OBS folders on Ubuntu
```sh
$ sudo cp -R release/RelWithDebInfo/lib/* /usr/lib/x86_64-linux-gnu/
$ sudo cp -R release/RelWithDebInfo/share/* /usr/share/
```
Note: The official [OBS plugins guide](https://obsproject.com/kb/plugins-guide) recommends adding plugins to the `~/.config/obs-studio/plugins` folder.

### Windows

Use the CI scripts again, for example:

```powershell
> .github/scripts/Build-Windows.ps1 -Target x64
```

The build should exist in the `./release` folder off the root. You can manually install the files in the OBS directory.
