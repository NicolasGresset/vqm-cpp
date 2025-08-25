cmake_minimum_required(VERSION 3.10)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio)

set(SOURCES
    src/main.cpp
    src/VQM.cpp
    src/Options.cpp
    src/FeatureExtractor.cpp
)

add_executable(main ${SOURCES})

target_link_libraries(main PRIVATE ${OpenCV_LIBS})

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
