
ARG ROS_VERSION="melodic"
FROM ros:${ROS_VERSION}
#package install
RUN apt-get update && apt-get install -y libc++-10-dev libc++abi-10-dev curl sudo wget build-essential ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev\
  && apt-get update;\
  apt-get clean;\
  rm -rf /var/lib/apt/lists/*;
#ROS install
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list';\
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -;\
  apt-get update;\
  apt-get install -y ros-melodic-desktop-full libpcl-dev ros-melodic-pcl-* ros-melodic-cv-bridge-*;echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc;\
  /bin/bash -c 'source ~/.bashrc';apt-get install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential;\
  apt-get install -y python-rosdep;rosdep init;rosdep update;

#CERES
RUN /bin/bash -c 'source ~/.bashrc';cd /root;wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz;tar zxf ceres-solver-2.1.0.tar.gz;\
  mkdir ceres-bin; cd ceres-bin;cmake ../ceres-solver-2.1.0;make -j$(nproc);make test;make install;

# ext calib
COPY . /workspace/src/targetless_calib/
WORKDIR /workspace/


RUN /bin/bash -c 'sudo mv /usr/include/flann/ext/lz4.h /usr/include/flann/ext/lz4.h.bak &&\
  sudo mv /usr/include/flann/ext/lz4hc.h /usr/include/flann/ext/lz4.h.bak &&\
  sudo ln -s /usr/include/lz4.h /usr/include/flann/ext/lz4.h &&\
  sudo ln -s /usr/include/lz4hc.h /usr/include/flann/ext/lz4hc.h';\
  /bin/bash -c 'source /opt/ros/melodic/setup.bash &&\
  cd /workspace &&\
  catkin_make && source ~/.bashrc'


