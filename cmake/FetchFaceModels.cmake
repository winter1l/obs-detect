# FetchFaceModels.cmake
# Downloads YOLOv8-Face and SCRFD ONNX models for face tracking

set(MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/models")
file(MAKE_DIRECTORY ${MODELS_DIR})

# URL Configuration
set(YOLOV8N_FACE_URL "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx")
set(YOLOV8S_FACE_URL "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.onnx")

set(YOLOV8N_FACE_FILE "${MODELS_DIR}/yolov8n-face.onnx")
set(YOLOV8S_FACE_FILE "${MODELS_DIR}/yolov8s-face.onnx")

# Helper to find Python
find_program(PYTHON_EXECUTABLE NAMES python3 python)

# Download YOLOv8n-Face
if(NOT EXISTS ${YOLOV8N_FACE_FILE})
    message(STATUS "Downloading YOLOv8n-Face ONNX model...")
    file(DOWNLOAD ${YOLOV8N_FACE_URL} ${YOLOV8N_FACE_FILE}
         SHOW_PROGRESS
         STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download YOLOv8n-Face. Please download it manually to ${YOLOV8N_FACE_FILE}")
    else()
        # Fix IR version for compatibility with OBS ONNX Runtime (max IR=9)
        if(PYTHON_EXECUTABLE)
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c
                "import onnx; m=onnx.load('${YOLOV8N_FACE_FILE}'); m.ir_version=7; onnx.save(m,'${YOLOV8N_FACE_FILE}')"
                RESULT_VARIABLE FIX_RESULT)
            if(FIX_RESULT EQUAL 0)
                message(STATUS "YOLOv8n-Face IR version patched to 7.")
            else()
                message(WARNING "Could not patch YOLOv8n-Face IR version. Model may not load in OBS.")
            endif()
        endif()
    endif()
else()
    message(STATUS "YOLOv8n-Face ONNX model already exists.")
endif()

# Download YOLOv8s-Face
if(NOT EXISTS ${YOLOV8S_FACE_FILE})
    message(STATUS "Downloading YOLOv8s-Face ONNX model...")
    file(DOWNLOAD ${YOLOV8S_FACE_URL} ${YOLOV8S_FACE_FILE}
         SHOW_PROGRESS
         STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download YOLOv8s-Face. Please download it manually to ${YOLOV8S_FACE_FILE}")
    else()
        if(PYTHON_EXECUTABLE)
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c
                "import onnx; m=onnx.load('${YOLOV8S_FACE_FILE}'); m.ir_version=7; onnx.save(m,'${YOLOV8S_FACE_FILE}')"
                RESULT_VARIABLE FIX_RESULT)
            if(FIX_RESULT EQUAL 0)
                message(STATUS "YOLOv8s-Face IR version patched to 7.")
            else()
                message(WARNING "Could not patch YOLOv8s-Face IR version. Model may not load in OBS.")
            endif()
        endif()
    endif()
else()
    message(STATUS "YOLOv8s-Face ONNX model already exists.")
endif()

