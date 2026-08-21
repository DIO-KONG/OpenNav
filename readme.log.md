2026.7.7
done testing master-slam wrapper, success.

2026.7.6
testing vllm object detection
constructing, collecting mock data

2026.7.3&7.4&7.5
updated driver to 595, cuda 13.2 
setting up environment for vllm
finished master-slam environment/testing at 7.4
finished vllm deployment setting up at 7.5


2026.7.2
take camera image
finished http interfaces
added masterslam, using conda env named ``move ``
installed vllm at c
# base movement preparation
1. run`tracer_base_node`
```
sudo modprobe gs_usb
$rosrun tracer_bringup bringup_can2usb.bash
roslaunch tracer_bringup tracer_robot_base.launch
```
2. start base http server
```
# if needed, conda deactivate multiple times
python tracer_http_interface/scripts/tracer_http_server.py

# can use this python /home/agilex/miniconda3/envs/aloha/bin/python

```
or use ros to run it
```bash
roslaunch tracer_http_interface tracer_http_interface.launch
```

# camera preparation
```
source ~/cobot_magic/camera_ws/devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

# run demo
get base information at ``tracer_http_interface/rw_api.py``
get rgb information at ``RGBRosConnector.py``

the entry of demo is at ``moveit.py``




2026.7.1

all movement control is in ``~/agilex_ws``


added tracer(base) http interfaces, but now can only move 1sec, bug in http server
```
source ~/AGILEX_WS/devel/setup.bash

```
added camera acquirement

To move the base using http: src codes are at ``tracer_ros/tracer_http_interface``
To acquire front camera(orbbec astra) information: ros1 launch ``multi camera``(src at ``/home/agilex/cobot_magic/camera_ws``), 
based on the [documentation](https://agilexsupport.yuque.com/staff-hso6mo/toh64r/tcpvae9wrb5xnivn?singleDoc)``
here's what you need to do
```

source ~/cobot_magic/camera_ws/devel/setup.bash
a
# 1.4 终端运行rostopic list查看ros话题 
rostopic list
# 1.5 一次打印上面ros话题的内容，保证采集数据前, 传感器数据正常
rostopic echo 话题名
```
you'll find rostopic ``camera_f/color/image_raw``
use my demo to retrieve the image at ``viewer/saver.py``
use my live script to see live rgb-streaming at ``rosrun image_view image_view image:=/camera_f/color/image_raw``
make sure the conda env has openssl==1.1.1. ``conda install openssl==1.1.1.1``

confirmed RGB-D camera laucn, by ``roslaunch astra_camera multi_cemera.launch``
confirmed rostopic as /cemra_f/color/image_raw
they are using ros1.


confirmed tracer2.0 using ros and sdk
found the core action sender is in ``tracer_ros/tracer_base/src/tracer_messager.cpp``

