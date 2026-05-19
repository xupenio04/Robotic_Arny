import math
import numpy as np

from omx_driver import OmxDriver
from trajectory_generator import TrajectoryGenerator

# ── Parâmetros do gerador de trajetória ───────────────────────────────────────

N_JOINTS = 6
VMAX     = [math.radians(20)] * N_JOINTS   # 20 °/s por junta
AMAX     = [math.radians(10)] * N_JOINTS   # 10 °/s² por junta
TS       = 0.01                             # período de controle (s)


def main():

    driver = OmxDriver()

    if not driver.motors:
        print("Nenhum motor encontrado. Verifique a conexão e tente novamente.")
        return

    while True:

        print("Digite a junta que deseja mover (1-6) ou 'sair' para encerrar:")
        id_joint = int(input()) + 10

        if id_joint < 10 or id_joint > N_JOINTS + 10:
            print("Junta inválida.")
            continue

        print("Digite o ângulo desejado (graus):")
        angle = math.radians(float(input()))

        qi = 0
        qf = angle

        Tg = TrajectoryGenerator(1, [VMAX[id_joint-11]], [AMAX[id_joint-11]], ts=TS)
        traj, _, _ = Tg.compute_trajectory([qi], [qf])
        driver.execute_trajectory_one_joint(traj, TS, id_joint-11)

if __name__ == "__main__":
    main()


# IndexError: index 1 is out of bounds for axis 0 with size 1 
# points[joint_idx] in execute_trajectory_one_joint() está tentando acessar um índice que não existe, porque traj é uma lista de listas, onde cada sublista tem apenas um elemento (a posição da junta). Para corrigir isso, basta acessar o primeiro elemento da sublista: