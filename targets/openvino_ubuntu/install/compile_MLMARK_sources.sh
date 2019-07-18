#!/bin/bash

# Copyright (c) 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

NUM_ARGS=$#
usage() {
    echo "Usage:"
    echo -e "\t compile_MLMARK_sources.sh </path/to/MLMARK>"
    echo -e "\t compile_MLMARK_sources.sh -dir </path/to/MLMARK>"
    echo -e "\t compile_MLMARK_sources.sh -h [PRINT HELP MESSAGE]"
    echo -e "Assumes:\n\t--- you have installed openVINO in /opt/intel/computer_vision_sdk/ or /opt/intel/openvino/"
    echo -e "\t--- you have cloned MLMARK"
    exit 1
}

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

#============================= parse command line options and set MLMARK install directory ===============================================
key="$1"
case $key in
    -h | -help | --help)
     usage
    ;;    
    -dir | --dir)
    flag=$key
    MLMARK_DIR="$2"
    ;;
esac


if [[ ${NUM_ARGS} -lt 1 ]]; then
  MLMARK_DIR="$( dirname $( dirname $( dirname $PWD ) ) )"
  if [ -d ${MLMARK_DIR} ] && [ -d "${MLMARK_DIR}/targets" ]; then
     echo -e "\e[1;33mMLMARK install directory not provided. Installing to ${MLMARK_DIR}\e[0m"
  else
     echo -e "\e[1;31mCannot determine MLMARK distribution directory. Please pass the source directory with -dir </path/to/MLMARK/folder>.\e[0m"
     usage
  fi
  
  OPENVINO_DIR="/opt/intel/computer_vision_sdk"
  if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
     echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
  else
     OPENVINO_DIR="/opt/intel/openvino"
     if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
        echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
     else
        echo -e "\e[1;31mCannot find OpenVINO in default install locations /opt/intel/openvino or /opt/intel/computer_vision_sdk\e[0m"
        usage
     fi
  fi
  
fi



if [[ ${NUM_ARGS} == 1 ]]; then
   MLMARK_DIR=$key
  if  [ ! -d ${MLMARK_DIR} ]; then   
     echo -e "\e[1;31m\n\nTarget folder ${key} does not exist.\n\e[0m"
     usage
  fi

  OPENVINO_DIR="/opt/intel/computer_vision_sdk"
  if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
     echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
  else
     OPENVINO_DIR="/opt/intel/openvino"
     if [ -d ${OPENVINO_DIR} ] && [ -d ${OPENVINO_DIR}/bin ]; then
        echo -e "\e[1;33mOpenVINO install directory not provided. But we found an existing installation at ${OPENVINO_DIR}. This will be used\e[0m"
     else
        echo -e "\e[1;31mCannot find OpenVINO in default install locations /opt/intel/openvino or /opt/intel/computer_vision_sdk\e[0m"
        usage
     fi
  fi

elif [[ ${NUM_ARGS} == 2 ]]; then
  MLMARK_DIR="$1"
  if  [ ! -d ${MLMARK_DIR} ]; then
     echo -e "\e[1;31m\n\nTarget folder ${key} does not exist.\n\e[0m"
     usage
  fi
  
  OPENVINO_DIR="$2"
  if [ ! -d ${OPENVINO_DIR} ]; then
     echo -e "\e[1;31m\n\nOpenvino install directory ${OPENVINO_DIR} not found.\n\e[0m"
     usage
  fi

elif [[ "${NUM_ARGS}" -ge "2" ]] && [ "$flag" == "-dir" ] && [ ! -d ${MLMARK_DIR} ]; then

   echo -e "\e[1;31m\n\nTarget folder ${MLMARK_DIR} does not exist.\n\e[0m"
   usage

elif [[ "${NUM_ARGS}" -ge "2" ]] && [ "${1}" != "-dir" ]; then
   MLMARK_DIR=${1}

   echo -e "\e[1;31mProvided install directory ${MLMARK_DIR} does not exist.\n\e[0m"
   usage

elif [[ ${NUM_ARGS} -gt 2 ]]; then
   echo -e "\e[1;31m\nCannot parse inputs\n\e[0m"
   usage
fi  

# Last sanity check
if [ ! -d "${MLMARK_DIR}/targets" ]; then
   echo -e "\e[1;31mProvided install directory ${MLMARK_DIR} does not have Modules subdir.\e[0m"
   usage
fi
if [ ! -d ${OPENVINO_DIR}/bin ]; then
   echo -e "\e[1;31m\n\nThe provided OpenVINO install directory must contain a 'bin' folder. Please check that you have installed openvino correctly.\n\e[0m"
   usage
fi


RUN_AGAIN="Then run the script again\n\n"
DASHES="\n\n==================================================\n\n"
CUR_PATH=$PWD

if [ -d "${OPENVINO_DIR}/deployment_tools/tools/" ]; then
   OPENVINO_BUILD="R1"
else
   OPENVINO_BUILD="R5"
fi

if [ ! -e $OPENVINO_DIR ]; then
   echo -e "\e[1;33m\nDid not find OpenVINO installed in ${OPENVINO_DIR}.\n\e[0m"
   echo -e "\e[1;0mPlease install OpenVINO distribution in /opt/intel\n\e[0m"

   exit 1
fi

OPENVINO_CV_DEP_DIR="${OPENVINO_DIR}/install_dependencies"

MLMARK_MODELS="${MLMARK_DIR}/models/openvino/"
MLMARK_FROZEN_GRAPHS="${MLMARK_DIR}/models/tensorflow"
MLMARK_CAFFE="${MLMARK_DIR}/models/caffe"
MLMARK_PLUGIN="${MLMARK_DIR}/targets/openvino_ubuntu/lib/"
MLMARK_BIN="${MLMARK_DIR}/targets/openvino_ubuntu/workloads/common/"
MLMARK_SOURCES="${MLMARK_DIR}/targets/openvino_ubuntu/workloads/common/src/"


#========================= Install dependencies =======================================================

# Step 1. Install Dependencies

printf "${DASHES}"
printf "Installing dependencies"
printf "${DASHES}"
#------------------------------------------------------------------------------------------------------
if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
    IFS='=' read -ra arr <<< "$(cat /etc/lsb-release | grep DISTRIB_RELEASE)" # get the release version
    RELEASE=${arr[1]}
fi

if [[ $DISTRO == "centos" ]]; then
    if command -v python3.5 >/dev/null 2>&1; then
        PYTHON_BINARY=python3.5
    fi
    if command -v python3.6 >/dev/null 2>&1; then
        PYTHON_BINARY=python3.6
    fi
    if [ -z "$PYTHON_BINARY" ]; then
        sudo -E yum install -y https://centos7.iuscommunity.org/ius-release.rpm
        #sudo -E yum install -y python36u easy_install python36u-pip
        sudo -E yum install -y python36u python36u-pip libgfortran3 build-essential libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base libpng12-dev python-pil
        sudo -E pip3.6 install virtualenv
        PYTHON_BINARY=python3.6
    fi
elif [[ $DISTRO == "ubuntu" ]]; then
    sudo -E apt -y install python3-pip libgfortran3 build-essential libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base
    PYTHON_BINARY=python3

    if [[ $RELEASE == "16.04" ]]; then
       echo -e "\e[0;32m Installing PIL and png packages for Ubuntu 16.04.\e[0m"
       sudo -E apt -y install libpng12-dev python-imaging

    else
       echo -e "\e[0;32m Installing PIL and png packages for Ubuntu 18.04.\e[0m"
       sudo -E apt -y install python-pil libpng-dev
    fi

else
   echo -e "\e[0;31mUnsupported operating system.\e[0m"
   exit
fi

#------------------------------------------------------------------------------------------------------

printf "${DASHES}"
printf "Installing cv sdk dependencies\n\n"
cd ${OPENVINO_CV_DEP_DIR}

if [ "${OPENVINO_BUILD}" == "R5" ]; then
   sudo -E ./install_cv_sdk_dependencies.sh; 
else
   sudo -E bash install_openvino_dependencies.sh
fi

cd ${CUR_PATH}

#========================= Setup Model Optimizer =======================================================
# Step 2. Enter OpenVINO environment and Configure Model Optimizer

printf "${DASHES}"
printf "Setting OpenVINO environment and Configuring Model Optimizer"
printf "${DASHES}"

if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
            printf "\n\nINTEL_CVSDK_DIR environment variable is not set. Trying to run ./setvars.sh to set it. \n"

    if [ -e "${OPENVINO_DIR}/bin/setupvars.sh" ]; then # for Intel CV SDK package
        SETVARS_PATH="${OPENVINO_DIR}/bin/setupvars.sh"
    else
        echo -e "\e[0;31mError: setvars.sh is not found\n\e[0m"
	exit 1
    fi
    if ! source ${SETVARS_PATH} ; then
        printf "Unable to run ./setvars.sh. Please check its presence. ${RUN_AGAIN}"
        exit 1
    fi
fi

OPENVINO_DT_DIR="${OPENVINO_DIR}/deployment_tools"
OPENVINO_IE_DIR="${OPENVINO_DT_DIR}/inference_engine/"
OPENVINO_MO_DIR="${OPENVINO_DT_DIR}/model_optimizer/"
MO_PATH="${OPENVINO_MO_DIR}/mo.py"

printf "${DASHES}"
printf "Install Model Optimizer dependencies"
cd "${OPENVINO_MO_DIR}/install_prerequisites"
bash install_prerequisites.sh

cd ${CUR_PATH}
#========================= Download and Convert Caffepre-trained models ===========================================
# Step 3a. Download and Convert Caffe pretrained Models

printf "${DASHES}"
printf "Downloading and Converting pretrained models"
printf "${DASHES}"

CAFFE_MODEL_NAMES=("resnet-50")

PRECISION_LIST=("FP16" "FP32")

for idx in "${!CAFFE_MODEL_NAMES[@]}"
  do
    
    MODEL_NAME=${CAFFE_MODEL_NAMES[idx]}
    echo -e "\n\e[0;32m Generating IRs for ${MODEL_NAME}\n\e[0m"
    IR_PATH="$CUR_PATH/${MODEL_NAME}"
    if [ -d ${IR_PATH} ];then rm -r ${IR_PATH}; fi;

	if [ $MODEL_NAME == "resnet-50" ]
	   then
		MEAN_VALUES="data[104.0,117.0,123.0]"
		SCALE_VALUES="data[1.0]"
		MODEL_DEST="resnet50"
		MODEL_PATH="${MLMARK_CAFFE}/${MODEL_DEST}/ResNet-50-model.caffemodel"
		INPUT_PROTO="${MLMARK_CAFFE}/${MODEL_DEST}/ResNet-50-deploy.prototxt"
		INPUT_SHAPE="[1,3,224,224]"
	fi

	for PRECISION in ${PRECISION_LIST[@]}
	  do
		precision=${PRECISION,,}
		OUTPUT_NAME=${IR_PATH}/${MODEL_NAME}_${precision}

		printf "Run $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $IR_PATH --model_name $MODEL_DEST --data_type "${PRECISION}" --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES\n\n"
		$PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --input_proto ${INPUT_PROTO}  --output_dir $IR_PATH --model_name ${OUTPUT_NAME} --data_type "${PRECISION}" --input_shape $INPUT_SHAPE --mean_values $MEAN_VALUES --scale_values $SCALE_VALUES

	done

	
    cp ${IR_PATH}/*.xml ${MLMARK_MODELS}/${MODEL_DEST}/
	cp ${IR_PATH}/*.bin ${MLMARK_MODELS}/${MODEL_DEST}/

done


TF_MODEL_NAMES=("mobilenet" "mobilenet-ssd")

BATCH_LIST=(1 2 4 8 16 32) # For ssd_mobilenet (tf model) inference on GPU, IRs have to be generated for specific batch

for idx in "${!TF_MODEL_NAMES[@]}"
  do
    MODEL_NAME=${TF_MODEL_NAMES[idx]}
    echo -e "\n\e[0;32m Generating IRs for ${MODEL_NAME}\n\e[0m"
	if [ $MODEL_NAME == "mobilenet-ssd" ]
	     then
		#MODEL_PATH="${TAR_DIR}/frozen_inference_graph.pb"
		MODEL_DEST="ssdmobilenet"
		MODEL_PATH="${MLMARK_FROZEN_GRAPHS}/${MODEL_DEST}/frozen_graph.pb"
		PIPELINE_CONFIG="${MLMARK_FROZEN_GRAPHS}/${MODEL_DEST}/pipeline.config"

		ALL_PARAMETERS="--tensorflow_use_custom_operations_config ${OPENVINO_DT_DIR}/model_optimizer/extensions/front/tf/ssd_v2_support.json --output=detection_boxes,detection_scores,num_detections --tensorflow_object_detection_api_pipeline_config ${PIPELINE_CONFIG}"
		for PRECISION in "${PRECISION_LIST[@]}"
		   do
		       precision=${PRECISION,,}
		       OUTPUT_NAME="${MODEL_DEST}_${precision}"
		       printf "Run $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $MODEL_DEST --model_name ${OUTPUT_NAME} --data_type ${precision} ${ALL_PARAMETERS}\n\n"
		       $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir ${MODEL_DEST} --model_name ${OUTPUT_NAME} --data_type ${PRECISION} ${ALL_PARAMETERS} --reverse_input_channels

			 for BATCH in "${BATCH_LIST[@]}"
			    do

			       precision=${PRECISION,,}
			       OUTPUT_NAME="${MODEL_DEST}_${precision}_b${BATCH}"
			       printf "Run $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir $MODEL_DEST --model_name ${OUTPUT_NAME} --data_type ${precision} ${ALL_PARAMETERS} --batch ${BATCH}\n\n"
			       $PYTHON_BINARY $MO_PATH --input_model $MODEL_PATH --output_dir ${MODEL_DEST} --model_name ${OUTPUT_NAME} --data_type ${PRECISION} ${ALL_PARAMETERS} --batch ${BATCH} --reverse_input_channels
			 done
		done

	elif [ $MODEL_NAME == "mobilenet" ]
	then
		MODEL_DEST="mobilenet"
		MODEL_PATH="${MLMARK_FROZEN_GRAPHS}/${MODEL_DEST}/frozen_graph.pb"
		INPUT_SHAPE="[1,224,224,3]"
		MEAN_VALUES="input[128.0,128.0,128.0]"
		SCALE_VALUES="input[128.0]"

		for PRECISION in ${PRECISION_LIST[@]}
		   do
			precision=${PRECISION,,} # Lower case
			OUTPUT_NAME=${MODEL_DEST}v1_${precision}
			printf "\n\nRun $PYTHON_BINARY $MO_PATH --input_model ${MODEL_PATH} --output_dir ${MODEL_DEST} --model_name ${OUTPUT_NAME} --data_type ${PRECISION} --input_shape ${INPUT_SHAPE}\n\n"
			$PYTHON_BINARY $MO_PATH --input_model ${MODEL_PATH} --output_dir ${MODEL_DEST} --model_name ${OUTPUT_NAME} --data_type ${PRECISION} --input_shape ${INPUT_SHAPE} --scale_values ${SCALE_VALUES} --mean_values ${MEAN_VALUES} --reverse_input_channels
		done              
	fi


    if [ ! -e ${MLMARK_MODELS}/${MODEL_DEST} ]; then
       printf "\n\nTarget folder ${MLMARK_MODELS}/${MODEL_DEST} does not exist. It will be created."
       mkdir ${MLMARK_MODELS}/${MODEL_DEST}
    fi

    cp -r ${MODEL_DEST}/* ${MLMARK_MODELS}/${MODEL_DEST}/
    rm -r ${MODEL_DEST}

done

#========================= Build Classification and Detection binaries ================================================================================
# Step 4. Build samples
printf "${DASHES}"
printf "Building MLMARK sources ${MLMARK_SOURCES}"
printf "${DASHES}"

if ! command -v cmake &>/dev/null; then
    echo -e "\e[0;32m\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it and run again.\n\e[0m"
    exit 1
fi

# copy sources here
MLMARK_TMP_SRC="$CUR_PATH/MLMARK_sources/"
BUILD_DIR="$CUR_PATH/MLMARK_compiled"
COMPILED_APP_DIR="${BUILD_DIR}/intel64/Release"
if [ -d "${MLMARK_TMP_SRC}" ]; then rm -Rf $MLMARK_TMP_SRC; fi
if [ -d "${BUILD_DIR}" ]; then rm -Rf $BUILD_DIR; fi

mkdir ${MLMARK_TMP_SRC}
cp -r ${MLMARK_SOURCES}/* ${MLMARK_TMP_SRC}
# cp -r ${OPENVINO_DIR}/inference_engine/samples/thirdparty ${MLMARK_TMP_SRC}
# cp -r ${OPENVINO_DIR}/inference_engine/samples/common ${MLMARK_TMP_SRC}

if [ ! -e "${COMPILED_APP_DIR}/image_classification" ]; then
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    cmake -DCMAKE_BUILD_TYPE=Release -DBIN_FOLDER="intel64" ${MLMARK_TMP_SRC}
    make -j8
else
    printf "\n\nTarget folder ${BUILD_DIR} already exists. Skipping samples building."
    printf "If you want to rebuild samples, remove the entire ${BUILD_DIR} folder. ${RUN_AGAIN}"
fi

if [ ! -e $COMPILED_APP_DIR ]; then
   printf "\n\nTarget folder ${COMPILED_APP_DIR} does not exists.\n"
   exit 1
else
   cp ${COMPILED_APP_DIR}/../image_classification_async ${MLMARK_BIN}
   cp ${COMPILED_APP_DIR}/../object_detection_ssd ${MLMARK_BIN}
   cp ${COMPILED_APP_DIR}/../object_detection_ssd_async ${MLMARK_BIN}

   # these are found after compiling MLMARK
   cp ${COMPILED_APP_DIR}/lib/libformat_reader.so ${MLMARK_PLUGIN}/
   cp ${COMPILED_APP_DIR}/lib/libcpu_extension.so ${MLMARK_PLUGIN}/

fi

#========================= Copy libraries ================================================================================
# Step 6. Finally copy plugins folder

#printf ${DASHES}
#printf "Copying OpenVINO Libraries"

PLUGIN_DIR="$CUR_PATH/plugin"
if [ -d "${PLUGIN_DIR}" ]; then rm -Rf $PLUGIN_DIR; fi
mkdir ${PLUGIN_DIR}

if [ $DISTRO == "centos" ];then 
   OPERATING_SYSTEM="centos_7.4"
   OS_VERSION="centos7"
elif [ $DISTRO == "ubuntu" ];then
   OPERATING_SYSTEM="ubuntu_${RELEASE}" # Options: ubuntu_16.04 centos_7.4
   if [ ${RELEASE} == "16.04" ]; then
      OS_VERSION="ubuntu16"
   elif [ ${RELEASE} == "18.04" ]; then
      OS_VERSION="ubuntu18"
   fi
fi

# Due to directory structure change in R1 releases
if [ "${OPENVINO_BUILD}" == "R1" ]; then
   OPERATING_SYSTEM=""
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNN64.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx2.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx512.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHeteroPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libinference_engine.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libMKLDNNPlugin.so $PLUGIN_DIR
   # cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libmyriadPlugin.so $PLUGIN_DIR

   cp $OPENVINO_IE_DIR/external/mkltiny_lnx/lib/* $PLUGIN_DIR
   
   # Copy tbb libraries
   find ${OPENVINO_IE_DIR}/external/tbb/lib/ -type f -name 'libtbb.so*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
   find ${OPENVINO_IE_DIR}/external/tbb/lib/ -type f -name 'libtbbmalloc.so*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   # Copy HDDL libraries
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHDDLPlugin.so $PLUGIN_DIR
   find ${OPENVINO_IE_DIR}/external/hddl/lib/ -type f -name 'lib*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   # openCV
   find ${OPENVINO_DIR}/opencv/lib/ -type f -name 'libopencv_core*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -type f -name 'libopencv_imgcodecs*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -type f -name 'libopencv_imgproc*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   cp ${OPENVINO_DIR}/python/python3.5/openvino/inference_engine/ie_api.so ${PLUGIN_DIR}

else

   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNN64.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libcpu_extension_avx2.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libclDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHeteroPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libinference_engine.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libMKLDNNPlugin.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libmyriadPlugin.so $PLUGIN_DIR

   cp $OPENVINO_IE_DIR/external/omp/lib/libiomp5.so $PLUGIN_DIR
   cp $OPENVINO_IE_DIR/external/mkltiny_lnx/lib/libmkl_tiny_omp.so $PLUGIN_DIR

   # HDDL libraries
   if [ $DISTRO == "ubuntu" ];then
      cp $OPENVINO_IE_DIR/lib/$OPERATING_SYSTEM/intel64/libHDDLPlugin.so $PLUGIN_DIR
      find ${OPENVINO_IE_DIR}/external/hddl/lib/ -type f -name 'lib*' -exec cp '{}' ${PLUGIN_DIR}/ ';'
   fi

   # openCV
   find ${OPENVINO_DIR}/opencv/lib/ -type f -name 'libopencv_core*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -type f -name 'libopencv_imgcodecs*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   find ${OPENVINO_DIR}/opencv/lib/ -type f -name 'libopencv_imgproc*' -exec cp '{}' ${PLUGIN_DIR}/ ';'

   cp ${OPENVINO_DIR}/python/python3.5/${OS_VERSION}/openvino/inference_engine/ie_api.so ${PLUGIN_DIR}

fi

cp -r ${PLUGIN_DIR}/* ${MLMARK_PLUGIN}/

#printf ${DASHES}
#======================= Remove temp directories =================================================
# Step 7. Post-install Clean-up

cd ${CUR_PATH}
MLMARK_compiled="${CUR_PATH}/MLMARK_compiled"
MLMARK_sources="${CUR_PATH}/MLMARK_sources"
plugin="${CUR_PATH}/plugin"
models="${CUR_PATH}/models"
caffe_models="${CUR_PATH}/caffe_models"

if [ -d "${MLMARK_compiled}" ]; then rm -Rf ${MLMARK_compiled}; fi
if [ -d "${MLMARK_sources}" ]; then rm -Rf ${MLMARK_sources}; fi
if [ -d "${plugin}" ]; then rm -Rf ${plugin}; fi
if [ -d "${models}" ]; then rm -Rf ${models}; fi
if [ -d "${caffe_models}" ]; then rm -Rf ${caffe_models}; fi

printf ${DASHES}
echo "OpenVINO Directory: ${OPENVINO_DIR}" > ${MLMARK_DIR}/harness/OpenVINO_BUILD.txt
echo -e "\e[1;32mSetup completed successfully.\e[0m"
printf ${DASHES}
exit 1

