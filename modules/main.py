import math
import numpy as np
import time

from omx_driver import OmxDriver
from trajectory_generator import TrajectoryGenerator

# ── Parâmetros do gerador de trajetória ───────────────────────────────────────

N_JOINTS = 6
VMAX     = [math.radians(20)] * N_JOINTS   # 20 °/s por junta
AMAX     = [math.radians(10)] * N_JOINTS   # 10 °/s² por junta
TS       = 0.01                             # período de controle (s)




def main():

    driver = OmxDriver()

    if not driver._motors:
        print("Nenhum motor encontrado. Verifique a conexão e tente novamente.")
        return

    while True:

        angle = []

        for i in range (N_JOINTS):
            print(f"Digite o ângulo desejado para a junta {i+1} (graus):")
            angle_input = math.radians(float(input()))

            if angle_input == "sair":
                driver.__del__()
                return

            if i == 5:
                print("Deseja abrir a garra")
                command = input()
                if command == "sim":
                    driver.open_gripper(TS)
                else:
                    driver.close_gripper(TS)

            angle.append(angle_input)
            print(angle)

        qi = driver.read_joint_positions()
        qf = angle

        #print(driver.read_joint_positions())
        #(print(qi[id_joint-11]))
        #print(qf)

        Tg = TrajectoryGenerator(N_JOINTS, VMAX, AMAX, ts=TS)
        traj, _, _ = Tg.compute_trajectory(qi, qf)

        print(driver.read_joint_positions())

        #driver.execute_trajectory_one_joint(traj, TS, id_joint-11)

        driver.execute_trajectory(traj, TS)

if __name__ == "__main__":
    main()

