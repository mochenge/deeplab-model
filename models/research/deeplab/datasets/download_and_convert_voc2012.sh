#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_voc2012_data.py
#     - download_and_convert_voc2012.sh
#     - remove_gt_colormap.py
#     + pascal_voc_seg
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

#---------------------------------------------------#
#   数据集下载并转换：
#   PASCAL VOC2012数据集下载，生成mask数据集，并转换为
#   Tensorflow的TFRecord格式.
#   download_convert_voc2012.sh
#---------------------------------------------------#


# 在命令行终端中按e立刻退出
# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./pascal_voc_seg"
# WORK_DIR="E://PycharmProject/DeepLab_model/datasets/pascal_voc_seg"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack VOC 2012 dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  tar -xf "${FILENAME}"
}

# Download the images.
BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
FILENAME="VOCtrainval_11-May-2012.tar"

# 下载和解压缩图像文件
download_and_uncompress "${BASE_URL}" "${FILENAME}"

# 回到当前目录 
cd "${CURRENT_DIR}"

# Root path for PASCAL VOC 2012 dataset.
PASCAL_ROOT="${WORK_DIR}/VOCdevkit/VOC2012"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${PASCAL_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/SegmentationClassRaw"

#---------------------------------------------------#
#   mask的灰度值转换
#   将background的灰度值设置为0，object1,2,3的灰度值设置为1，2，3
#   去除mask的colormap
#   SegmentationClass: 此处存放的是带有colormap的单通道png图像
#   SegmentationClassRaw: 此处存放的是去除colormap的mask图像
#---------------------------------------------------#
echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"




# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PASCAL_ROOT}/ImageSets/Segmentation"


#---------------------------------------------------#
#   转换成TFRecord格式数据集
#   此处需要：确保输入图像为jpg格式，输入的mask为png格式
#   list_folder中为txt格式的
#---------------------------------------------------#
echo "Converting PASCAL VOC 2012 dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
