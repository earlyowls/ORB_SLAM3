#!/bin/bash

delete_if_exists() {
  local folder=$1
  build_folder="${folder}/build"
  bin_folder="${folder}/bin"
  lib_folder="${folder}/lib"
  if [ -d "$build_folder" ]; then
    rm -rf "$build_folder"
  fi
  if [ -d "$bin_folder" ]; then
    rm -rf "$bin_folder"
  fi
  if [ -d "$lib_folder" ]; then
    rm -rf "$lib_folder"
  fi
}

build_library() {
  library_name="$1"
  source_folder="$2"
  verbose="$3"
  force_build="$4"

  build_folder="$source_folder/build"
  bin_folder="$source_folder/bin"
  lib_folder="$source_folder/lib"

  if [ "$force_build" = true ]; then
  	delete_if_exists ${source_folder}
  fi

  if [ "$verbose" = true ]; then
    echo "[ORB-SLAM3][build.sh] Compile ${library_name} ... "
    cmake -G Ninja -B $build_folder -S $source_folder -DCMAKE_PREFIX_PATH=$source_folder -DCMAKE_INSTALL_PREFIX=$source_folder
    cmake --build $build_folder --config Release -j 4
  else
    echo "[ORB-SLAM3][build.sh] Compile ${library_name} (output disabled) ... "
    cmake -G Ninja -B $build_folder -S $source_folder -DCMAKE_PREFIX_PATH=$source_folder -DCMAKE_INSTALL_PREFIX=$source_folder > /dev/null 2>&1
    cmake --build $build_folder --config Release -j 4 > /dev/null 2>&1
  fi
}

# Check inputs
force_build=false
verbose=false
for input in "$@"
do
    if [ "$input" = "-f" ]; then
  	force_build=true
    fi
    if [ "$input" = "-v" ]; then
  	verbose=true
    fi
done

# Baseline Dir
LIBRARY_PATH=$(realpath "$0")
LIBRARY_DIR=$(dirname "LIBRARY_PATH")

## Build DBoW2
library_name="DBoW2"
source_folder="${LIBRARY_DIR}/Thirdparty/${library_name}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}

## Build g2o
library_name="g2o"
source_folder="${LIBRARY_DIR}/Thirdparty/${library_name}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}

## Build Sophus
library_name="Sophus"
source_folder="${LIBRARY_DIR}/Thirdparty/${library_name}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}

## Build ORB-SLAM3
library_name="ORB-SLAM3"
source_folder="${LIBRARY_DIR}"
build_library ${library_name} ${source_folder} ${verbose} ${force_build}

## Uncompress vocabulary
echo "[ORB-SLAM3][build.sh] Uncompress vocabulary ... "
vocabulary_folder="${LIBRARY_DIR}/Vocabulary"
if [ ! -f "${vocabulary_folder}/ORBvoc.txt" ]; then
	tar -xf "${LIBRARY_DIR}/Vocabulary/ORBvoc.txt.tar.gz" -C "${LIBRARY_DIR}/Vocabulary"
fi

