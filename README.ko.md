## OBS Detect - 객체 감지 및 마스킹 필터 (fork)

> [!NOTE]
> 본 리포지토리는 [원본(royshil/obs-detect)](https://github.com/royshil/obs-detect)을 포크하여 대부분을 AI로 수정하였습니다.
>
> 특히 **야외 방송 시 일반 행인의 초상권을 신속하고 안정적으로 보호**하는 것에 초점을 두고 최적화되었습니다.

어떤 소스에서든 다양한 종류의 객체를 감지하고, 추적하며, 마스킹을 적용할 수 있는 [OBS Studio](https://obsproject.com/) 플러그인입니다.

이 프로젝트의 원본 개발자분들을 응원하고 싶으시다면, GitHub 스폰서를 통해 후원을 고려해 주세요:

- https://github.com/sponsors/royshil
- https://github.com/sponsors/umireon

이 프로젝트는 [EdgeYOLO-ROS](https://github.com/fateshelled/EdgeYOLO-ROS) 및 [PINTO-Model-Zoo](https://github.com/PINTO0309/PINTO_model_zoo), [yolov8-face](https://github.com/lindevs/yolov8-face)의 훌륭한 기여물을 사용합니다. 헝가리안 알고리즘(Hungarian algorithm)은 GPLv2 라이선스 하에 https://github.com/Gluttton/munkres-cpp 의 코드를 가져와 사용했습니다.

## 사용법

1. 이미지가 포함된 모든 소스(미디어, 브라우저, VLC, 이미지 등)에 "Detect" 필터를 추가합니다.
2. "마스킹"을 활성화하고 블러나 모자이크로 선택합니다.
3. 세부 설정을 조절합니다.

> [!TIP]
> 효과의 기본값이 **'Person'** 객체 카테고리(사람 추적)에 알맞게 최적화되어있습니다.

### ['특정 대상 마스킹 제외 (얼굴 인식)' 사용법 보기](docs/FACE_GUIDE_KR.md)

### [옵션 설명 보기]()

## [원본](https://github.com/royshil/obs-detect)과 무엇이 다른가?

> 이 포크 리포지토리는 사람 10%와 AI 90%로 수정되었습니다.

### 1. 특정 인물 마스킹 제외
> 스트리머 본인 또한 마스킹(블러, 모자이크)되는 불편을 해소하기 위해, 얼굴을 분석하여 일치하는 경우에 마스킹하지 않는 기능이 추가되었습니다.
- **Zero-training 얼굴 매칭**: `YuNet` (얼굴 탐지)과 `SFace` (얼굴 특징 추출 및 비교) 모델을 결합하여, 제외할 인물의 얼굴 사진 폴더를 지정하면 해당 인물의 유사도를 분석하여 추적 및 마스킹 대상에서 실시간으로 제외합니다.
- **다중 얼굴 지원**: 얼굴 사진 폴더에는 여러 각도와 상황에서 촬영한 이미지를 넣어 정확도를 올리고, 여러 명으로 추가할 수도 있습니다.

### 2. 향상된 트래킹 알고리즘 및 고스트 복구
> 급격한 움직임에서도 안정적으로 트래킹하도록 개선하였습니다.
- **[yolov8-face](https://github.com/lindevs/yolov8-face) 모델 추가**: 더욱 안정적으로 얼굴을 추적하는 [yolov8-face](https://github.com/lindevs/yolov8-face) 모델을 추가했습니다.
- **고스트 복구 기능 추가**: 급격한 카메라 무빙이나 객체의 급격한 움직임으로 인해 트래킹을 놓치더라도, 헝가리안 알고리즘 매칭 이후 공간적 전역 매칭을 추가로 작동시킵니다. 이를 통해 과거 궤적 정보를 바탕으로 대상을 순식간에 재조회하여 튀는 현상(마스크 깜빡임) 없이 실시간 복구(Teleport) 후 추적을 이어 나갑니다.
- **화면 경계 이탈 예외 처리**: 대상이 완전히 화면 밖 경계 영역으로 이탈하는 시점에는 무리한 고스트 복구가 작동하지 않도록 예외 처리하여, 오탐지를 방지했습니다.
- **크기 비례 추적 감도**: 객체 크기 비율에 따라 칼만 필터의 측정 노이즈 변수를 실시간으로 비례 보정합니다. 화면 내에서 아주 작게 표시되는 얼굴 등은 과거 프레임의 관성 반영률을 낮추어 빠르게 움직임을 따라가고, 전신처럼 크게 인식되는 객체는 노이즈 반영을 억제하여 부드럽게 추적하도록 설계했습니다.

### 3. 0.5초 지연으로 미래 프레임 확보
> 0.5초 버퍼를 추가하여, 트래킹이 더 안정적이고 얼굴이 잠깐 노출되는 초상권 침해를 방지합니다.
- **지연 시간 버퍼 (Look-ahead Delay)**: 비디오와 오디오를 0.5초 지연시키고, 객체 추적은 지연되지 않은 미래 프레임을 확보합니다. 이를 통해 추적 안정성(고스팅 방지, 튀는 추적 방지)을 높였습니다.
- **인식 전에 마스킹**: 모델이 인식할 정도로 객체가 등장하기 0.25초 전에 미리 마스크를 씌워주는 선제 마스킹을 구현했습니다. 이를 통해 객체가 잠깐 노출되는 현상을 해결했습니다.
- **재인식된 객체 자연스럽게 복구**: 행인이 가로수나 가로등 뒤를 통과하여 트래킹을 놓치는 상황 발생 시, 유실 직전 좌표와 복구 후 좌표 사이의 궤적을 선형으로 보간하여 자연스러운 마스킹 흐름을 유지합니다.

### 4. 최적화
> 당신의 컴퓨터가 조금 덜 힘들어합니다.
- **GPU Zero-Copy Pipeline**: Direct3D 11/12 및 DirectML 환경과의 상호 운용성을 확보하여 CPU와 GPU 간의 불필요한 텍스처 복사 및 메모리 정렬 오버헤드를 차단했습니다. 메모리 대역폭을 절약하여 1080P 및 4K 고해상도 환경에서의 연산 지연을 줄였습니다.
- **백그라운드 비동기 추론 (Async Inference Workloads)**: GPU 및 AI 모델 추론 루프와 얼굴 특징 추출(SFace) 비교 연산 등 무거운 작업을 비디오 렌더링 스레드와 분리된 별도의 백그라운드 스레드에서 비동기 처리합니다. 추론 프레임이 출렁이더라도 OBS 메인 화면이 끊기거나 드랍되는 현상을 방지합니다.
- **렌더 캐시 텍스처 갱신 개선**: 정적 소스 혹은 비디오 프레임 갱신 주기 변동 시 화면이 굳거나 마스크 렌더가 중단되는 동결(Frozen source) 버그를 `GS_DYNAMIC` 속성 바인딩 수정을 통해 해결했습니다.
- **동적 크기 마스크 확장 (Dynamic Scale Expansion)**: 리소스 소모가 심한 기존 `cv::dilate` 연산을 제거하고, 객체 해상도와 카메라 거리 비율에 매칭되는 크기 스케일링 수학적 사각형 확장 연산을 자체 구현하여 리소스를 보존했습니다.

### 5. UI/UX 정밀화 및 디버깅 오버레이 추가
- **상태 시각화 오버레이**: 디버그 모드가 켜지면 각 추적 박스의 내부 상태 기계(`New`, `Stable`, `Recovered`, `Unseen`) 및 제외(Exempt) 상태를 오버레이로 화면에 표시합니다.
- **한국어 추가**: 개발자가 영어를 못하는 한국인이어서 `ko-KR.ini` 언어 팩을 추가했습니다.
- **비디오-객체 싱크 미세 조정**: 비디오 렌더 시점과 객체 추적 박스의 미세한 싱크 편차를 보정할 수 있도록 오프셋 조절 기능을 추가했습니다.

## 주요 기능

현재 지원되는 기능:

- 효율적인 모델([EdgeYOLO](https://github.com/LSH9832/edgeyolo))을 사용하여 80개 이상의 객체 카테고리를 감지
- 3가지 모델 크기: Small, Medium, Large
- 빠르고 효율적인 얼굴 감지 모델 ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet))
- 커스텀 ONNX 감지 모델 로드 가능
- 필터링 기준: 최소 감지 신뢰도(Confidence), 객체 카테고리(예: "Person"만 감지), 객체 최소 크기
- 마스킹 옵션: 블러(Blur), 픽셀화(Pixelate), 단색(Solid color), 투명(Transparent), 바이너리 마스크 출력 (다른 플러그인과 조합 가능!)
- 추적 옵션: 단일 객체 / 가장 큰 객체 / 가장 오래된 객체 / 모든 객체, 줌 배율, 부드러운 전환
- 추적의 부드러움과 연속성을 위한 SORT 알고리즘 적용
- Streamer.bot 등과의 연동을 위해 실시간 감지 결과를 파일로 저장

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
