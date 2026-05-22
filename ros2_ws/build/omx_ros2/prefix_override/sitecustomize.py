import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/robot/Documents/Grupo_pvsc_alas4_lopt_iams/Robotic_Arny/ros2_ws/install/omx_ros2'
