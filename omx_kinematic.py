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

    def forward_kinematics(self, theta1, theta2, theta3, theta4, theta5, theta6):
        T01 = hm.R_z(theta1) @ hm.T_xyz(0, 0, self.l1)
        T12 = hm.R_x(theta2) @ hm.T_xyz(0, 0, self.l2)
        T23 = hm.R_z(theta3) @ hm.T_xyz(self.l3, -self.l4, 0)
        T34 = hm.R_z(theta4) @ hm.T_xyz(self.l5, 0, 0)
        T45 = hm.R_y(theta5) @ hm.T_xyz(self.l6, 0, 0)
        T56 = hm.R_z(theta6) @ hm.T_xyz(0, 0, self.l7)  
    
        return T01 @ T12 @ T23 @ T34 