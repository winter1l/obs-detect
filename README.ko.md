# ⚠️ 중단됨 ⚠️ 이 프로젝트는 현재 활발히 개발되고 있지 않습니다.

## OBS Detect - 객체 감지 및 마스킹 필터

<div align="center">

[![GitHub](https://img.shields.io/github/license/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/blob/main/LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/occ-ai/obs-detect/push.yaml)](https://github.com/occ-ai/obs-detect/actions/workflows/push.yaml)
[![Total downloads](https://img.shields.io/github/downloads/occ-ai/obs-detect/total)](https://github.com/occ-ai/obs-detect/releases)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/occ-ai/obs-detect)](https://github.com/occ-ai/obs-detect/releases)
[![Discord](https://img.shields.io/discord/1200229425141252116)](https://discord.gg/KbjGU2vvUz)

</div>

어떤 소스에서든 다양한 종류의 객체를 감지하고, 추적하며, 마스킹을 적용할 수 있는 [OBS Studio](https://obsproject.com/) 플러그인입니다.

완전히 무료로 제공되는 이 프로젝트가 마음에 드신다면, GitHub 스폰서를 통해 후원을 고려해 주세요:

- https://github.com/sponsors/royshil
- https://github.com/sponsors/umireon

이 프로젝트는 [EdgeYOLO-ROS](https://github.com/fateshelled/EdgeYOLO-ROS) 및 [PINTO-Model-Zoo](https://github.com/PINTO0309/PINTO_model_zoo)의 훌륭한 기여물을 사용합니다. 헝가리안 알고리즘(Hungarian algorithm)은 GPLv2 라이선스 하에 https://github.com/Gluttton/munkres-cpp 의 코드를 가져와 사용했습니다.

## 🌟 주요 업데이트 (버전 `a4f4928` 이후)

> **🤖 AI 기반 개발**: 본 리포지토리의 주요 업데이트 및 알고리즘 고도화 작업은 전적으로 사용자 피드백과 AI(LLM) 어시스턴트와의 페어 프로그래밍(Pair Programming)을 통해 설계, 코딩 및 디버깅되었습니다.

본 리포지토리는 원본(`a4f4928`) 버전에서 크게 발전하여 다음과 같은 핵심 기능들이 추가 및 개선되었습니다.

### 1. 얼굴 기반 특정 인물 추적 제외 (Face-based Personal Exclusion)
- **Zero-training 얼굴 매칭**: 별도의 얼굴 데이터 학습 없이 `YuNet` (얼굴 탐지)과 `SFace` (얼굴 특징 추출 및 비교) 모델을 결합하여, 제외할 인물의 정면 사진 파일만 지정하면 해당 인물을 추적 및 마스킹 대상에서 실시간으로 제외하는 기술을 도입했습니다.
- **독립 비동기 스레드 처리**: 연산 부하가 큰 얼굴 특징 추출 및 매칭 연산을 렌더링 파이프라인에서 완벽히 분리하고, 전용 백그라운드 스레드(`inference_thread_loop`)에서 비동기로 수행하도록 구현하여 OBS의 60FPS 렌더링 프레임 저하를 완전히 방어했습니다.
- **다중 참조 이미지 디렉토리 지원**: 단일 이미지 지정 방식을 개선하여, 여러 장의 참조 이미지(Reference Images)가 저장된 디렉토리를 통째로 선택할 수 있도록 지원합니다. 이를 통해 다양한 조명, 각도, 표정에서의 얼굴 매칭 인식률을 크게 높였습니다.
- **로버스트 파일 로딩**: `cv::imread` 대신 `stb_image` 라이브러리를 채택하여, 경로 상에 한글이나 공백 등 다국어 문자가 섞여 있거나 이미지 포맷이 다양하더라도 중단 없이 원활하게 얼굴 이미지를 로드할 수 있도록 안정성을 극대화했습니다.
- **YOLO Fallback 시스템**: 눈코입 랜드마크(Landmarks) 좌표 출력을 제공하지 않거나 포맷이 다른 일부 YOLOv8-Face 계열 모델(예: Lindevs 포맷) 사용 시, SFace 정렬(Align) 단계가 실패하지 않도록 백그라운드에서 임시로 YuNet을 동작시켜 진짜 랜드마크 좌표를 동적으로 보정 및 주입해주는 Fallback 메커니즘을 적용했습니다.
- **캐싱 및 오버헤드 최소화**: 사용자가 설정을 변경할 때마다 얼굴 이미지를 디스크에서 매번 재로딩하는 리소스 낭비를 방지하기 위해 캐싱을 도입하였고, 동일한 프레임에 대한 불필요한 재탐색을 방지했습니다.

### 2. 향상된 트래킹 알고리즘 및 고스트 복구 (Advanced Tracking & Ghost Recovery)
- **Ghost Recovery (고스트 복구)**: 추적 대상이 급격한 카메라 무빙, 급작스러운 프레임 아웃, 일시적인 가려짐(Occlusion) 등으로 탐지되지 않더라도, 헝가리안 알고리즘 매칭 이후 공간적 전역 매칭을 추가로 작동시킵니다. 이를 통해 과거 궤적 정보를 바탕으로 대상을 순식간에 재조회하여 튀는 현상(마스크 깜빡임) 없이 실시간 복구(Teleport) 후 추적을 이어 나갑니다.
- **관성 및 공분산 보존**: 고스트 복구 시 칼만 필터의 예측 모델이 튀거나 불안정해지지 않도록, 복구 시점의 이전 속도 관성(Velocity Inertia) 및 공분산(Covariance) 정보를 필터 내부로 정교하게 주입해 주는 물리 보존 로직을 설계했습니다.
- **화면 경계 이탈 예외 처리**: 대상이 완전히 화면 밖 경계 영역으로 이탈하는 시점에는 무리한 고스트 복구가 작동하지 않도록 오작동 제어(Block)를 추가하여 잘못된 마스크가 불필요하게 그려지는 오탐지를 방지했습니다.
- **다이내믹 칼만 필터 측정 노이즈 (Dynamic Kalman Filter Noise)**: 객체가 화면 내에서 차지하는 면적 비율(Area Ratio)에 따라 칼만 필터의 측정 노이즈(Measurement Noise) 변수를 실시간으로 비례 보정합니다. 화면 내에서 아주 작게 표시되는 얼굴 등은 과거 프레임의 관성 반영률을 극도로 낮추어 빠르고 기민하게(Responsive) 움직임을 따라가고, 전신처럼 크게 표시되는 객체는 노이즈 반영을 억제하여 부드러운(Smooth) 궤적을 그리도록 설계했습니다.

### 3. 룩어헤드(Look-ahead) 및 선제 마스킹 (Reverse Masking)
- **지연 큐 동기화 (Look-ahead Delay)**: 입력되는 비디오 프레임을 지연 큐에 버퍼링(최대 20프레임 설정 가능)하고, 실시간 추론 스레드는 지연되지 않은 미래 프레임을 미리 확인하여 추론 정보를 산출합니다.
- **역방향 선형 외삽 (Reverse Linear Extrapolation)**: 미래 시점의 추론 데이터에서 도출된 객체의 속도 벡터를 역산하여, 화면에 객체가 실제로 나타나기 전부터 미리 마스크를 씌워주는 선제 마스킹(Reverse Masking)을 구현하여 모델 추론 지연 및 마스크 번쩍임 현상을 완벽히 해결했습니다.
- **보간(Interpolation) 연산**: 지연 큐 동기화 과정에서 AI 추론 성능 저하로 생길 수 있는 미세한 렌더링 깜빡임(Blinking)을 방지하고자 누락 프레임 간의 선형 보간 처리를 통해 안정성을 확보했습니다.
- **오디오 동기화 및 팝 노이즈 방지**: 비디오의 프레임 지연에 비례하여 오디오 오프셋도 정밀 동기화하였으며, 지연 버퍼 리사이징 시 발생하는 오디오 틱 노이즈(Pop-Noise)를 최소화하기 위한 정밀 크로스페이딩(Crossfading) 필터를 내장했습니다.

### 4. GPU Zero-Copy 및 고성능 렌더링
- **GPU Zero-Copy Pipeline**: Direct3D 11/12 및 DirectML(DML) 환경과의 상호 운용성을 확보하여 CPU와 GPU 간의 불필요한 텍스처 복사 및 메모리 정렬 오버헤드를 근본적으로 차단했습니다. 비디오 메모리(VRAM) 내에서 모든 렌더링 연산을 수행하여 4K 고해상도 소스 필터 적용 시의 전송 지연을 최소화했습니다.
- **렌더 캐시 텍스처 갱신 개선**: 렌더링 스레드에서 캐싱되는 텍스처 버퍼에 `GS_DYNAMIC` 속성을 바인딩하여, 특정 정적 소스 혹은 비디오 갱신 시 화면이 멈추거나 마스크 렌더가 중단되는 동결(Frozen source) 버그를 원천 해결했습니다.
- **동적 크기 마스크 확장 (Dynamic Scale Expansion)**: 성능 부하가 심한 `cv::dilate` 연산을 제거하고, 객체 해상도와 카메라 거리 비율에 매칭되는 크기 스케일링 수학적 사각형 확장 연산으로 마스크 영역을 확장하도록 직접 제작하여 연산 리소스를 대폭 아꼈습니다.

### 5. UI/UX 정밀화 및 디버깅 오버레이
- **상태 시각화 오버레이**: 디버그 모드가 켜지면 각 추적 박스의 내부 상태 기계(`New`, `Stable`, `Recovered`, `Unseen`) 및 제외(Exempt) 상태를 직관적인 색상 코드와 텍스트 오버레이로 화면에 직접 표시합니다.
- **다국어 경로 로깅 및 상세 디버그**: 로그를 파일로 내보낼 때 한글이 포함된 윈도우 사용자 경로에서도 깨짐 없이 저장되도록 `std::filesystem::u8path`를 채택하고, 이어쓰기(Append) 기능과 프레임 번호, 칼만 필터 공분산 수치 정밀 출력을 추가했습니다.
- **ONNX IR 버전 강제 패치**: 플러그인이 탑재한 OBS ONNX Runtime 엔진 환경의 호환성을 맞추기 위해, 로드되는 외부 ONNX 모델의 IR(Intermediate Representation) 버전을 검사하여 필요 시 버전 7로 자동 강제 변환/패치하는 로직을 통합했습니다.
- **한국어 로컬라이제이션 업데이트**: 플러그인 UI의 세부 설정값들을 명확하게 파악할 수 있도록 `ko-KR.ini` 언어팩의 오역을 고치고 번역 수준을 높였습니다.

## 사용법

<div align="center">
<a href="https://youtu.be/LrbUrvaGreQ"><img width="40%" src="https://github.com/occ-ai/obs-detect/assets/441170/b8e7367e-c1b0-4c7e-b0df-af45ead87199" /></a>&nbsp;
<a href="https://youtu.be/zmdq1bPVYs0"><img width="40%" src="https://github.com/occ-ai/obs-detect/assets/441170/2eb08589-1695-4a40-877e-4985c2b5270f" /></a>
</div>

- 이미지가 포함된 모든 소스(미디어, 브라우저, VLC, 이미지 등)에 "Detect" 필터를 추가합니다.
- "마스킹(Masking)" 또는 "추적(Tracking)"을 활성화합니다.

Detect를 사용하여 반려동물을 추적하거나 비디오에서 사람들을 흐리게 처리(블러)해 보세요!

더 자세한 정보와 사용법 튜토리얼은 곧 추가될 예정입니다.

## 주요 기능

현재 지원되는 기능:

- 효율적인 모델([EdgeYOLO](https://github.com/LSH9832/edgeyolo))을 사용하여 80개 이상의 객체 카테고리를 감지
- 3가지 모델 크기: Small, Medium, Large
- 빠르고 효율적인 얼굴 감지 모델 ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet))
- 디스크에서 커스텀 ONNX 감지 모델 로드 가능
- 필터링 기준: 최소 감지 신뢰도(Confidence), 객체 카테고리(예: "Person"만 감지), 객체 최소 크기
- 마스킹 옵션: 블러(Blur), 픽셀화(Pixelate), 단색(Solid color), 투명(Transparent), 바이너리 마스크 출력 (다른 플러그인과 조합 가능!)
- 추적 옵션: 단일 객체 / 가장 큰 객체 / 가장 오래된 객체 / 모든 객체, 줌 배율, 부드러운 전환
- 추적의 부드러움과 연속성을 위한 SORT 알고리즘 적용
- Streamer.bot 등과의 연동을 위해 실시간 감지 결과를 파일로 저장

로드맵 기능(예정):
- 바운딩 박스를 넘어서는 정밀한 객체 마스크
- 여러 객체 카테고리 동시 선택 (예: 개 + 고양이 + 오리)
- 설정을 통해 다른 플러그인이 감지 정보에 접근할 수 있도록 연동

## 커스텀 감지 모델 학습 및 사용

자신만의 커스텀 모델을 학습하고 사용하려면 [docs/train_model.md](docs/train_model.md)의 안내를 따르세요.

## 빌드 방법

이 플러그인은 macOS (Intel 및 Apple Silicon), Windows, Linux 환경에서 빌드 및 테스트되었습니다.

먼저 원하는 디렉토리에 이 리포지토리를 클론하세요.

### macOS

CI 파이프라인 스크립트를 사용하여, 로컬에서는 zsh 스크립트를 실행하기만 하면 됩니다. 기본적으로 Intel 및 Apple Silicon 모두를 위한 유니버설 바이너리가 빌드됩니다. 특정 아키텍처용 빌드 옵션은 `.github/scripts/.build.zsh`의 `-arch` 옵션을 참고하세요.

```sh
$ ./.github/scripts/build-macos -c Release
```

#### 설치
위 스크립트가 성공적으로 실행되면 루트 경로 아래의 `./release/Release` 폴더에 플러그인 파일(예: `obs-ocr.plugin`)이 생성됩니다. 이 `.plugin` 파일을 OBS 디렉토리(예: `~/Library/Application Support/obs-studio/plugins`)로 복사하세요.

.pkg 설치 패키지 파일을 생성하려면 다음 명령을 실행하세요:
```sh
$ ./.github/scripts/package-macos -c Release
```
(참고: 출력이 `package-macos` 스크립트가 예상하는 `install` 폴더가 아닌 `Release` 폴더에 있을 수 있습니다. 이 경우 폴더 이름을 `build_x86_64/Release`에서 `build_x86_64/install`로 변경해야 할 수 있습니다.)

### Linux (Ubuntu)

CI 스크립트를 사용해 빌드합니다:
```sh
$ ./.github/scripts/build-linux.sh
```

빌드 결과를 우분투의 표준 OBS 폴더로 복사합니다:
```sh
$ sudo cp -R release/RelWithDebInfo/lib/* /usr/lib/x86_64-linux-gnu/
$ sudo cp -R release/RelWithDebInfo/share/* /usr/share/
```
참고: 공식 [OBS 플러그인 가이드](https://obsproject.com/kb/plugins-guide)에서는 플러그인을 `~/.config/obs-studio/plugins` 폴더에 추가하는 것을 권장합니다.

### Windows

마찬가지로 CI 스크립트를 사용하여 빌드합니다. 예:

```powershell
> .github/scripts/Build-Windows.ps1 -Target x64
```

빌드된 결과물은 루트 경로 아래의 `./release` 폴더에 생성됩니다. 생성된 파일들을 OBS 설치 디렉토리에 수동으로 설치할 수 있습니다.
