1. ✅️, 7.4, test base api using curl
2. ✅️, 7.4, test keyboard.py
3. ✅️, 7.5, mast3r-slam environment setting up
4. ✅️, 7.5, test camera video on master-slam, GPU util 36.5%, FPS: 14.2608, speed kinda slow!!
5. ✅️, 7.6, vllm deploy Qwen3.5, Qwen3, QwenVL 2.5
6. ✅️, 7.6, quatization of QwenVL3, Qwen3.5, QwenVL2.5, FP16 -> fp8 (OOM if regular deployment)
7. ✅️, 7.7, masterslam wrapper integration
8. ✅️, 7.7, Qwen3.5 Qwen3 detection service testing by wenhao
9. ✅️, 7.8, Qwen+mobile sam pipeline
10. ✅️, 7.8, (localhost), communication between VLM and robots.
11. ✅️, 7.8, serve Qwen and video slam togeter by wenhao
12. ✅️, 7.9, serve Qwen and real-time slam togeter by wenhao
13. 🎯, video recording (remote, keyboard) test real-world environment on mast3r-slam by wenhao
14. ✅️, 7.9, device wlan setup
14. ✅️, Qwen on real-world sensor testing, QA
15. ✅️, mock navigation process
16. 🎯, boost master-slam speed, fps drop down quickly, dont know why!
17. ✅️, nav state machine backbone
18. ⌛️, path planning, path following testing.
19. ✅️, 7.13, smooth movement, preventing shaking (multiprocessing, pure persuit algorithm)
20. ⌛️, secure slam safety
21. ✅️, 7.13, local map cropping, escaping corner
22. ✅️, 7.15, point cloud filtering
22. ✅️, 7.19, map history of visited by wenhao
23. ✅️, 7.16, dynamic map height trauncation by wenhao
25. ❓️, using phone video instead of astras
26. ✅️, 7.20, local frontier searching by wenhao
27. ✅️, 7.21, frontier with RRT ensurance
28. ✅️, 7.21,  global frontier searching by wenhao
29. ✅️, 7.21,  global frontier with global RRT ensurance by wenhao
30. ✅️, 7.17, async for speed and smooth control
31. ✅️, 7.23, integration of vlm detection to nav logic
32. fix RRT periodic replan bug
33. added odometry data for escaping phase
34. added final adjust phase in real-world scenario
35. fixed pure pursuit algorithm bug when goal is too near
36. tried wider scene in 11th floor
37. merged mock process to fully automatic navigation process
38. added log and episode data for automatic navigation process
39. added log replay and visualizer by wenhao
40. boost efficiency of replay visualizer
41. added astra-depth-only SLAM system
42. tried astra depth SLAM system with odom data by wenhao
43. clean log system




# experiences on SLAM

## when will the slam get "relocation mode"
1. ppl in obs but who suddenly vanishes.
2. too close to the obstacle while rotation
3. facing the plain wall

## how to get a "good" demo
1. don't start the navigation if you find the PC is not well positioned.
2. try not to face the wall, if really need to, 
    make the rotation slow, make sure TRACKING MODE is on.



# download model using hf-mirror
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com


# restart todesk
sudo systemctl restart todeskd.service

