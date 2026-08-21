cmake_minimum_required(VERSION 3.0.2)
project(tracer_http_interface)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()

# Install Python scripts so rosrun can find them
catkin_install_python(PROGRAMS
  scripts/tracer_http_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)