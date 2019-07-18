#!/bin/bash
DASHES="\n\n==========================================================\n\n"

#=========================== Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2 =============================
printf ${DASHES}
printf "Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2"
printf ${DASHES}

CUR_DIR=$PWD

# USB rules
sudo usermod -a -G users "$(whoami)"
if [ ! -e "${CUR_DIR}/97-usbboot.rules" ];then 
  echo -e "\e[0;31mUSB boot rules not found. Please include boot rules file\e[0m";
  exit
fi
sudo cp 97-usbboot.rules /etc/udev/rules.d
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo ldconfig

echo -e "\e[1;32mIntel Compute Stick support added.\e[0m"
