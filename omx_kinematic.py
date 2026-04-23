import numpy as np
import math
import homogeneous_matrix as hm
import matplotlib.pyplot as plt


class omxKinematicClass():

    def __init__(self, l1, l2, l3, l4, l5, l6, l7):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6
        self.l7 = l7

    def forward_kinematics(self, theta1, theta2, theta3, theta4, theta5):
        
        T01 = hm.T_xyz(0, 0, self.l1) @ hm.R_z(theta1)
        T12 = hm.T_xyz(0, 0, self.l2) @ hm.R_x(np.pi/2) @ hm.R_z(theta2)
        T23 = hm.T_xyz(self.l3, 0, 0) @ hm.R_z(theta3) 
        T_aux = hm.T_xyz(0, -self.l4, 0) 
        T34 = hm.T_xyz(self.l5, 0, 0) @ hm.R_z(theta4)
        T45 = hm.T_xyz(self.l6, 0, 0) @ hm.R_y(np.pi/2) @ hm.R_z(theta5)  
        T56 = hm.T_xyz(0, 0, self.l7)  

        T0 = np.eye(4)
        T1 = T01
        T2 = T01 @ T12
        T3 = T2 @ T23
        T_extra_joint = T3 @ T_aux
        T4 = T_extra_joint @ T34
        T5 = T4 @ T45
        T6 = T5 @ T56

        return [T0, T1, T2, T3, T_extra_joint, T4, T5, T6]