cmake_minimum_required(VERSION 2.8)
project(DBoW2)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")

find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.0 not found.")
endif()



include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/DBoW
${PROJECT_SOURCE_DIR}/DUtils
${PROJECT_SOURCE_DIR}/DVision
${OpenCV_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    DBoW/BowVector.cpp
    DBoW/FBrief.cpp
    DBoW/FeatureVector.cpp
    DBoW/QueryResults.cpp
    DBoW/ScoringObject.cpp
    DUtils/Random.cpp
    DUtils/Timestamp.cpp
    DVision/BRIEF.cpp
    DBoW/VocabularyBinary.cpp

    DBoW/BowVector.h
    DBoW/DBoW2.h
    DBoW/FBrief.h
    DBoW/FClass.h
    DBoW/FeatureVector.h 
    DBoW/QueryResults.h
    DBoW/ScoringObject.h
    DBoW/TemplatedDatabase.h
    DBoW/TemplatedVocabulary.h
    DUtils/DException.h
    DUtils/DUtils.h
    DUtils/Random.h 
    DUtils/Timestamp.h 
    DVision/BRIEF.h 
    DVision/DVision.h
    DBoW/VocabularyBinary.hpp

)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
)
