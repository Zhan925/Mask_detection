# Mask_detection

## What Does This Program Do?
This Program re-trained the ssd detectnet network and it can recognize human face and detect if they are wearing the mask properly or not.

## Prerequisite
NVIDIA Jetson Nano Developer Kit\
USB Web Camera or Raspberry Pi Camera V2\
NVIDIA JetPack 4.2.1 or later\
You should also follow the Hello AI World Tutorial(https://github.com/dusty-nv/jetson-inference) to set the docker environment and know how to run the deep learning model in the docker environment

## Installation
1. Set up the jetson-inference\
You can set up the jetson-inference following this tutorial https://github.com/dusty-nv/jetson-inference

2. Get to the right directory\
$ cd ~/jetson-inference/python/training/detection/ssd

3. Install this application and the dependent modules\
$ git clone https://github.com/Zhan925/Mask_detection

4. Set the environment variable\
$ export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

5. Set the camera\
If you are using the USB camera, just ignore this step. If you are using Rashbery Pi camera:\
$ cd Mask_detection\
$ vim my-detection.py\
modify the files following the comments


## Usage
1. Go to the jetson-inference\
$ cd ~/jetson-inference

2. Run the docker\
$  docker/run.sh --volume ~/jetson-inference/python/training/detection/ssd/Mask_detection:/jetson-inference/python/training/detection/ssd

3. Go to the directory\
$ cd /jetson-inference/python/training/detection/ssd/Mask_detection

4. Run the program\
$  python3 my-detection.py\
Note: it takes time for the first-run, so just be patient.

## Demonstration
https://youtu.be/oY5Ss2yr6O8
## Liscence

\#\
\# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.\
\#
\# NOTICE TO LICENSEE:\
\#\
\# This source code and/or documentation ("Licensed Deliverables") are\
\# subject to NVIDIA intellectual property rights under U.S. and\
\# international Copyright laws.\
\#\
\# These Licensed Deliverables contained herein is PROPRIETARY and\
\# CONFIDENTIAL to NVIDIA and is being provided under the terms and\
\# conditions of a form of NVIDIA software license agreement by and\
\# between NVIDIA and Licensee ("License Agreement") or electronically\
\# accepted by Licensee.  Notwithstanding any terms or conditions to\
\# the contrary in the License Agreement, reproduction or disclosure\
\# of the Licensed Deliverables to any third party without the express\
\# written consent of NVIDIA is prohibited.\
\#\
\# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE\
\# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE\
\# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS\
\# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.\
\# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED\
\# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,\
\# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.\
\# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE\
\# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY\
\# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY\
\# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,\
\# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS\
\# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE\
\# OF THESE LICENSED DELIVERABLES.\
\#\
\# U.S. Government End Users.  These Licensed Deliverables are a\
\# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT\
\# 1995), consisting of "commercial computer software" and "commercial\
\# computer software documentation" as such terms are used in 48\
\# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government\
\# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and\
\# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all\
\# U.S. Government End Users acquire the Licensed Deliverables with\
\# only those rights set forth herein.\
\#\
\# Any use of the Licensed Deliverables in individual and commercial\
\# software must include, in the user documentation and internal\
\# comments to the code, the above Disclaimer and U.S. Government End\
\# Users Notice.\
\#



