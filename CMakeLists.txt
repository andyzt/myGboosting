cmake_minimum_required(VERSION 3.5)
project(myGboosting)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
message(${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE CURRENT_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/*.cpp
        ${PROJECT_SOURCE_DIR}/*.cc
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.hxx)

message(${CURRENT_SOURCE_FILES})

set(SOURCE_FILES ${SOURCE_FILES} ${CURRENT_SOURCE_FILES})

set(CMAKE_CXX_STANDARD 14)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -pthread -Wall -Wextra -Wpedantic")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(myGboosting ${SOURCE_FILES})
#TARGET_LINK_LIBRARIES(myGboosting proto ${PROTOBUF_LIBRARY})
