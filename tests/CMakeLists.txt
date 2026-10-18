add_executable(obs-detect-safety-tests ../src/FaceInferenceQueue.cpp ../src/FaceReferenceSelection.cpp
                                       ../src/FaceStateStore.cpp ../src/FrameUtils.cpp ../src/MaskGeometry.cpp
                                       ../src/sort/StateHistory.cpp
                                       test_safety.cpp)

target_compile_features(obs-detect-safety-tests PRIVATE cxx_std_17)
target_include_directories(obs-detect-safety-tests PRIVATE ../src)
target_compile_definitions(obs-detect-safety-tests PRIVATE OBS_DETECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
find_package(Threads REQUIRED)
target_link_libraries(obs-detect-safety-tests PRIVATE Threads::Threads)

if(TARGET OpenCV)
  target_link_libraries(obs-detect-safety-tests PRIVATE OpenCV)
else()
  target_link_libraries(obs-detect-safety-tests PRIVATE "${OpenCV_LIBRARIES}")
  target_include_directories(obs-detect-safety-tests SYSTEM PRIVATE "${OpenCV_INCLUDE_DIRS}")
endif()

add_test(NAME obs-detect-safety COMMAND obs-detect-safety-tests)
