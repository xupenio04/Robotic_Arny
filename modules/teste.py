import numpy as np
from omx_kinematic import omxKinematicClass

l1 = 0.036
l2 = 0.040
l3 = 0.040
l4 = 0.040
l5 = 0.124
l6 = 0.040
l7 = 0.130

robot = omxKinematicClass(l1, l2, l3, l4, l5, l6, l7)
q_test = [1.57, 1.57 , 0.0,  -1.57,  0.0]

frames = robot.forward_kinematics(*q_test)
labels = ['Base', 'J1', 'J2', 'J2_aux', 'J3', 'J4', 'J5', 'Tool']

for label, T in zip(labels, frames):
    x, y, z = T[0,3], T[1,3], T[2,3]
    print(f"{label:10s}  x={x:.4f}  y={y:.4f}  z={z:.4f}")