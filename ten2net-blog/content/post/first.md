+++
title = "Opencv在Ubuntu上的安装过程"
draft = false
date = "2016-12-21T09:52:09+08:00"

+++


# 一、[阅读](https://www.raben.com/content/opencv-installation-ubuntu-1204)
# 二、安装依赖
-	To install OpenCV 2.4.2 or 2.4.3 on the Ubuntu 12.04 operating system, first install a developer environment to build OpenCV.
	apt-get -y install build-essential cmake pkg-config
-	Install Image I/O libraries
	apt-get -y install libjpeg62-dev 
	apt-get -y install libtiff4-dev libjasper-dev
-	Install the GTK dev library
	apt-get -y install  libgtk2.0-dev
-	Install Video I/O libraries
	apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
-	Optional - install support for Firewire video cameras
	apt-get -y install libdc1394-22-dev
-	Optional - install video streaming libraries
	apt-get -y install libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev 
-	Optional - install the Python development environment and the Python Numerical library
	apt-get -y install python-dev python-numpy
-	Optional - install the parallel code processing library (the Intel tbb library)
	apt-get -y install libtbb-dev
-	Optional - install the Qt dev library
	apt-get -y install libqt4-dev
# 三、安装opencv
-	wget https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.13/opencv-2.4.13.zip
-	unzip opencv-2.4.13.zip
-	cd opencv-2.4.13
- 	#（可选，若出现list_filterout错误）修改samples/gpu/CMakeLists.txt 文件的106、109、110、111、112五行代码如下：
```cpp
	 if(NOT HAVE_OPENGL)
	 #   list_filterout(install_list ".*opengl.cpp")
	  endif()
	  if(NOT HAVE_CUDA)
	 #   list_filterout(install_list ".*opticalflow_nvidia_api.cpp")
	 #   list_filterout(install_list ".*cascadeclassifier_nvidia_api.cpp")
	 #   list_filterout(install_list ".*driver_api_multi.cpp")
	 #   list_filterout(install_list ".*driver_api_stereo_multi.cpp")
	  endif()
```
-	mkdir build
-	cd build
-	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D WITH_FFMPEG=OFF    -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON     -D BUILD_EXAMPLES=OFF -D WITH_QT=OFF -D WITH_OPENGL=OFF ..

-	make
-	make install

# 四、问题
	### 解决1394问题
	ln /dev/null /dev/raw1394

# 五、测试
-	python
-	>>>import cv2
-	  不报错即表示安装成功




附：安装gnome后出现中文乱码的问题
  apt-get install gnome-language-selector
  然后在Xterm中执行
  #gnome-language-selector
