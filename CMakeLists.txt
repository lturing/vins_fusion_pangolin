cmake_minimum_required(VERSION 2.8)
project(vins_fusion)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE relwithdebinfo)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}  -march=native")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -march=native")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()


find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/camera_models/include/camodocal
${PROJECT_SOURCE_DIR}/camera_models/include/camodocal/calib
${PROJECT_SOURCE_DIR}/camera_models/include/camodocal/camera_models
${PROJECT_SOURCE_DIR}/camera_models/include/camodocal/chessboard
${PROJECT_SOURCE_DIR}/camera_models/include/camodocal/gpl
${PROJECT_SOURCE_DIR}/camera_models/include/camodocal/sparse_graph
${PROJECT_SOURCE_DIR}/vins_estimator/src/estimator
${PROJECT_SOURCE_DIR}/vins_estimator/src/factor
${PROJECT_SOURCE_DIR}/vins_estimator/src/featureTracker
${PROJECT_SOURCE_DIR}/vins_estimator/src/initial
${PROJECT_SOURCE_DIR}/vins_estimator/src/utility
${PROJECT_SOURCE_DIR}/loop_fusion/src

${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${CERES_INCLUDE_DIRS}
${Sophus_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
camera_models/src/calib/CameraCalibration.cc
camera_models/src/camera_models/Camera.cc
camera_models/src/camera_models/CameraFactory.cc
camera_models/src/camera_models/CataCamera.cc
camera_models/src/camera_models/CostFunctionFactory.cc
camera_models/src/camera_models/EquidistantCamera.cc
camera_models/src/camera_models/PinholeCamera.cc
camera_models/src/camera_models/PinholeFullCamera.cc
camera_models/src/camera_models/ScaramuzzaCamera.cc
camera_models/src/chessboard/Chessboard.cc
camera_models/src/gpl/EigenQuaternionParameterization.cc
camera_models/src/gpl/gpl.cc
camera_models/src/sparse_graph/Transform.cc
vins_estimator/src/estimator/estimator.cpp
vins_estimator/src/estimator/feature_manager.cpp
vins_estimator/src/estimator/parameters.cpp
vins_estimator/src/factor/marginalization_factor.cpp
vins_estimator/src/factor/pose_local_parameterization.cpp
vins_estimator/src/factor/projection_factor.cpp
vins_estimator/src/factor/projectionOneFrameTwoCamFactor.cpp
vins_estimator/src/factor/projectionTwoFrameOneCamFactor.cpp
vins_estimator/src/factor/projectionTwoFrameTwoCamFactor.cpp
vins_estimator/src/featureTracker/feature_tracker.cpp
vins_estimator/src/initial/initial_alignment.cpp
vins_estimator/src/initial/initial_ex_rotation.cpp
vins_estimator/src/initial/initial_sfm.cpp
vins_estimator/src/initial/solve_5pts.cpp
vins_estimator/src/utility/utility.cpp

loop_fusion/src/keyframe.cpp
loop_fusion/src/pose_graph.cpp
loop_fusion/src/viewer.cpp

camera_models/include/camodocal/calib/CameraCalibration.h
camera_models/include/camodocal/camera_models/Camera.h
camera_models/include/camodocal/camera_models/CameraFactory.h
camera_models/include/camodocal/camera_models/CataCamera.h
camera_models/include/camodocal/camera_models/CostFunctionFactory.h
camera_models/include/camodocal/camera_models/EquidistantCamera.h
camera_models/include/camodocal/camera_models/PinholeCamera.h
camera_models/include/camodocal/camera_models/PinholeFullCamera.h
camera_models/include/camodocal/camera_models/ScaramuzzaCamera.h
camera_models/include/camodocal/chessboard/Chessboard.h
camera_models/include/camodocal/chessboard/ChessboardCorner.h
camera_models/include/camodocal/chessboard/ChessboardQuad.h
camera_models/include/camodocal/chessboard/Spline.h
camera_models/include/camodocal/gpl/EigenQuaternionParameterization.h
camera_models/include/camodocal/gpl/EigenUtils.h
camera_models/include/camodocal/gpl/gpl.h
camera_models/include/camodocal/sparse_graph/Transform.h
vins_estimator/src/estimator/estimator.h
vins_estimator/src/estimator/feature_manager.h 
vins_estimator/src/estimator/parameters.h
vins_estimator/src/factor/imu_factor.h
vins_estimator/src/factor/initial_bias_factor.h
vins_estimator/src/factor/initial_pose_factor.h
vins_estimator/src/factor/integration_base.h
vins_estimator/src/factor/marginalization_factor.h 
vins_estimator/src/factor/pose_local_parameterization.h
vins_estimator/src/factor/projection_factor.h
vins_estimator/src/factor/projectionOneFrameTwoCamFactor.h 
vins_estimator/src/factor/projectionTwoFrameOneCamFactor.h
vins_estimator/src/factor/projectionTwoFrameTwoCamFactor.h
vins_estimator/src/featureTracker/feature_tracker.h 
vins_estimator/src/initial/initial_alignment.h 
vins_estimator/src/initial/initial_ex_rotation.h 
vins_estimator/src/initial/initial_sfm.h 
vins_estimator/src/initial/solve_5pts.h 
vins_estimator/src/utility/tic_toc.h
vins_estimator/src/utility/utility.h 

loop_fusion/src/keyframe.h 
loop_fusion/src/pose_graph.h 
loop_fusion/src/viewer.h 
)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${CERES_LIBRARIES}
${PROJECT_SOURCE_DIR}/ThirdParty/DBoW/lib/libDBoW2.so
${Boost_LIBRARIES}
-lboost_serialization
-lcrypto
)


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples)

add_executable(stereo_kitti
        examples/KITTIGPSTest.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

