cmake_minimum_required(VERSION 3.16)
project(DBoW2)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)

# Dependencies
find_package(OpenCV REQUIRED)

# Create shared library
set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h   
  DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

add_library(${PROJECT_NAME} SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})


# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
        include/DBoW2/
        include/
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBRARIES}
)

# Config file for find_package support
configure_file(DBoW2/DBoW2.cmake.in
        "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

# Install the shared library
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Install the header files
install(DIRECTORY DBoW2/ DESTINATION include/DBoW2 FILES_MATCHING PATTERN "*.h")
install(DIRECTORY DUtils/ DESTINATION include/DUtils FILES_MATCHING PATTERN "*.h")

# Optionally install the config file
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake"
        DESTINATION lib/cmake/DBoW2/)