# FetchFaceModels.cmake
# Downloads YOLOv8-Face and SCRFD ONNX models for face tracking

set(MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/models")
file(MAKE_DIRECTORY ${MODELS_DIR})

# URL Configuration
set(YOLOV8N_FACE_URL "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx")
set(YOLOV8S_FACE_URL "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.onnx")
set(SCRFD_500M_URL "https://github.com/deepinsight/insightface/raw/master/detection/scrfd/onnx/scrfd_500m_kps.onnx") # Note: This might not be a direct link, but acts as a fallback.

set(YOLOV8N_FACE_FILE "${MODELS_DIR}/yolov8n-face.onnx")
set(YOLOV8S_FACE_FILE "${MODELS_DIR}/yolov8s-face.onnx")
set(SCRFD_500M_FILE "${MODELS_DIR}/scrfd_500m_kps.onnx")

# Download YOLOv8n-Face
if(NOT EXISTS ${YOLOV8N_FACE_FILE})
    message(STATUS "Downloading YOLOv8n-Face ONNX model...")
    file(DOWNLOAD ${YOLOV8N_FACE_URL} ${YOLOV8N_FACE_FILE}
         SHOW_PROGRESS
         STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download YOLOv8n-Face. Please download it manually to ${YOLOV8N_FACE_FILE}")
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
    endif()
else()
    message(STATUS "YOLOv8s-Face ONNX model already exists.")
endif()

# Download SCRFD 500M
if(NOT EXISTS ${SCRFD_500M_FILE})
    message(STATUS "Downloading SCRFD 500M ONNX model...")
    file(DOWNLOAD ${SCRFD_500M_URL} ${SCRFD_500M_FILE}
         SHOW_PROGRESS
         STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download SCRFD 500M. Please download it manually to ${SCRFD_500M_FILE}")
    endif()
else()
    message(STATUS "SCRFD 500M ONNX model already exists.")
endif()
