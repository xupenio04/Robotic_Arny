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

        in_garra = input()

        if(in_garra == "sim"):
            driver.open_gripper()
        elif(in_garra == "não"):
            driver.close_gripper()
        
if __name__ == "__main__":
    main()

