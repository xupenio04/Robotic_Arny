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
    
# =====================================================
# CINEMÁTICA INVERSA MELHORADA
# Mais precisa + mais estável
# =====================================================

def numerical_jacobian(robot, q, eps=1e-6):
    """
    Jacobiano numérico da posição do tool tip
    Retorna matriz 3 x N
    """
    n = len(q)
    J = np.zeros((3, n))

    T0 = robot.forward_kinematics(*q)

    if isinstance(T0, list):
        T0 = T0[-1]

    p0 = T0[:3, 3]

    for i in range(n):

        q2 = np.array(q, dtype=float)
        q2[i] += eps

        T1 = robot.forward_kinematics(*q2)

        if isinstance(T1, list):
            T1 = T1[-1]

        p1 = T1[:3, 3]

        J[:, i] = (p1 - p0) / eps

    return J


def inverse_kinematics(robot,
                       T_desired,
                       q0=None,
                       max_iter=1500,
                       tol=1e-5,
                       alpha=0.15,
                       damping=0.05):
    """
    IK melhorada usando Damped Least Squares
    """

    n = 5

    if q0 is None:
        q = np.zeros(n)
    else:
        q = np.array(q0, dtype=float)

    if isinstance(T_desired, list):
        T_desired = T_desired[-1]

    T_desired = np.array(T_desired, dtype=float)

    target = T_desired[:3, 3]

    for _ in range(max_iter):

        T = robot.forward_kinematics(*q)

        if isinstance(T, list):
            T = T[-1]

        p = T[:3, 3]

        error = target - p

        # convergência
        if np.linalg.norm(error) < tol:
            return q

        # Jacobiano
        J = numerical_jacobian(robot, q)

        # =================================================
        # Damped Least Squares
        # dq = Jᵀ (J Jᵀ + λ² I)^-1 e
        # =================================================
        I = np.eye(3)

        J_dls = J.T @ np.linalg.inv(J @ J.T + (damping**2) * I)

        dq = alpha * (J_dls @ error)

        q = q + dq

        # normaliza ângulos entre -pi e pi
        q = (q + np.pi) % (2*np.pi) - np.pi

    raise ValueError("IK não convergiu")