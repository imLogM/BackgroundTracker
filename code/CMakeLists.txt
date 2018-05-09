cmake_minimum_required(VERSION 2.8)

project(Tracker_B13_R3)
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wextra -pedantic -Wno-long-long")
endif()

FIND_PACKAGE( OpenCV 2 REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR}/src/)

FILE(GLOB_RECURSE sourcefiles "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(tracker_B13 ${sourcefiles})
target_link_libraries(tracker_B13 ${OpenCV_LIBS})




