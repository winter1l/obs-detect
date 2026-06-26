# OBS-Detect Project Rules & Knowledge Base

이 파일은 obs-detect 프로젝트의 구조, 설계 방향 및 특이사항을 기록한 문서입니다. 에이전트(AI)는 이 프로젝트에서 작업할 때 항상 아래의 내용을 숙지하고 준수해야 합니다.

## 1. 렌더링 및 비동기(Async) 구조 (중요)
* **목표 프레임률 방어**: OBS의 렌더링 파이프라인(60FPS)이 인퍼런스 지연으로 인해 끊기는 현상을 막기 위해, 모델 인퍼런스는 백그라운드 스레드(`inference_thread_loop`)에서 비동기로 처리됩니다.
* **텍스처 복사 최적화**: 렌더링 스레드(`detect_filter_video_render`)에서는 GPU 상에서 렌더링을 유지하기 위해 CPU-GPU 간 메모리 복사(`cv::Mat` 클론)를 최소화하고, `gs_texrender_get_texture(tf->texrender)`를 사용하여 베이스 프레임을 가져옵니다.
* **동기화 모드 (Zero-Delay)**: 고사양 PC 사용자를 위해 렌더링 프레임 틱에서 직접 인퍼런스를 수행하는 동기화 모드(`tf->syncMode`) 기능이 존재합니다. 이는 1~2프레임의 지연을 완벽하게 제거하지만, 렌더링 틱을 차단하므로 사양이 부족하면 전체 OBS 프레임 드랍을 유발할 수 있습니다.

## 2. 트래킹 및 마스킹 기능
* **마스킹 동적 확장 (Dynamic Expansion)**: 마스크 확장 기능은 단순히 `cv::dilate`를 쓰지 않고, 대상 객체의 화면 내 크기 비율에 맞춰 동적으로 확장(Scale)하도록 직접 `cv::rectangle` 수학 연산으로 구현되어 있어 성능 부하가 거의 없습니다.
* **SORT 기반 대상 고정 및 예외 처리**: 대상이 잠시 사라지더라도 유지하는 유령 방지(Min Hit Frames), 제외 대기 시간(Exclude Delay), 제외 해제 면적 완화 비율(Safety Drop-off Ratio), 단일 대상일 때만 제외(Exclude Single Only) 등의 커스텀 옵션 로직이 구현되어 있습니다.

## 3. 하드웨어 및 빌드 환경
* **하드웨어 가속**: CUDA는 환경 호환성 문제로 기본 옵션에서 제외되었으며, 현재는 **CPU** 및 **GPU (DirectML)** 최적화를 위주로 동작합니다.
* **빌드 시스템**: CMake를 기반으로 하며, 빌드 명령어는 `cmake --build build --config Release` 형식을 따릅니다.
* **디버그 모드**: `Debug Mode`를 켜면 현재의 인퍼런스 레이턴시와 FPS를 OBS 화면에 텍스트로 출력합니다.

## 4. 로컬라이제이션 (언어팩)
* UI 프로퍼티를 추가할 경우, 반드시 `data/locale/en-US.ini` 및 `data/locale/ko-KR.ini` 두 파일 모두에 번역을 추가해야 합니다.
* `ko-KR.ini`를 편집할 때는 인코딩 깨짐을 방지하기 위해 파일 저장 시 반드시 UTF-8(또는 UTF-8 BOM) 인코딩을 보존해야 합니다.
