from dynamixel_easy_sdk import *
import time
import math

# ============================================================================
# CONFIGURAÇÃO DOS MOTORES
# ============================================================================
#
# D       -> direção cinemática (+1 ou -1)
# S       -> resolução do encoder
# offset  -> zero mecânico do motor
# raw_min/max -> limites de segurança
#
# IMPORTANTE:
# Mantivemos raw_min < raw_max para evitar ambiguidades.
# A direção já é tratada por D.
#
# ============================================================================

MOTOR_CONFIG = [
    {"id": 11, "D":  1, "S": 4096, "offset":  2048, "raw_min": 1024, "raw_max": 3200},
    {"id": 12, "D": -1, "S": 4096, "offset":  3072, "raw_min": 750, "raw_max": 3270 },
    {"id": 13, "D": -1, "S": 4096, "offset":  1024, "raw_min": 730, "raw_max": 3200},
    {"id": 14, "D": -1, "S": 4096, "offset":  2048, "raw_min": 700, "raw_max": 3300},
    {"id": 15, "D":  1, "S": 4096, "offset":  2048, "raw_min":    0, "raw_max": 4096},
    {"id": 16, "D":  1, "S": 4096, "offset": 2048, "raw_min": 2048, "raw_max": 3200},
]

PORT = "/dev/ttyACM0"
BAUD_RATE = 1000000



def raw_to_joint(raw: int, D: int, S: int, offset: int) -> float:
    """
    Encoder -> junta [rad]
    """
    return D * (raw - offset) * (2.0 * math.pi / S)


def joint_to_raw(theta: float, D: int, S: int, offset: int) -> int:
    """
    Junta [rad] -> encoder

    ALTERAÇÃO IMPORTANTE:
    ---------------------
    Adicionado wrap modular (% S).

    Isso evita:
    - valores negativos
    - overflow do encoder
    - comandos inválidos

    Fundamental para Dynamixel.
    """

    raw = int(round(
        D * theta * (S / (2.0 * math.pi)) + offset
    ))

    return raw % S


# ============================================================================
# CLAMP DE SEGURANÇA
# ============================================================================

def clamp_raw(raw: int,
              raw_min: int,
              raw_max: int,
              motor_id: int) -> int:

    clamped = max(raw_min, min(raw_max, raw))

    if clamped != raw:
        print(
            f"[OmxDriver] WARNING: motor {motor_id} "
            f"raw command {raw} clamped to {clamped} "
            f"(limits [{raw_min}, {raw_max}])."
        )

    return clamped


# ============================================================================
# DRIVER
# ============================================================================

class OmxDriver:

    def __init__(
        self,
        port: str = PORT,
        baud_rate: int = BAUD_RATE,
        motor_config: list = MOTOR_CONFIG,
    ):

        self._motor_config = motor_config

        self._connector = Connector(port, baud_rate)

        self._group_executor = self._connector.createGroupExecutor()

        # --------------------------------------------------------------------
        # Criação dos motores
        # --------------------------------------------------------------------

        self._motors = [
            self._connector.createMotor(cfg["id"])
            for cfg in self._motor_config
        ]

        print(self._motor_config)

        # --------------------------------------------------------------------
        # Configuração do modo de operação
        # --------------------------------------------------------------------

        for motor in self._motors:
            motor.disableTorque()

        for motor in self._motors:
            motor.setOperatingMode(OperatingMode.EXTENDED_POSITION)
            

        for motor in self._motors:
            motor.enableTorque()

        print(
            f"[OmxDriver] Connected on {port} @ {baud_rate} Bd — "
            f"{len(self._motors)} motors ready."
        )

    def shutdown(self):

        for motor in self._motors:
            motor.disableTorque()

        print("[OmxDriver] Torque disabled — driver shut down.")

    def __del__(self):
        self.shutdown()

    # =========================================================================
    # LEITURA DAS JUNTAS
    # =========================================================================

    def read_joint_positions(self) -> list[float]:

        # --------------------------------------------------------------------
        # Stage sync read
        # --------------------------------------------------------------------

        for motor in self._motors:
            self._group_executor.addCmd(
                motor.stageGetPresentPosition()
            )

        raw_positions = self._group_executor.executeRead()

        self._group_executor.clearStagedReadCommands()

        # --------------------------------------------------------------------
        # Conversão encoder -> junta
        # --------------------------------------------------------------------

        joint_positions = [
            raw_to_joint(
                raw_positions[i],
                self._motor_config[i]["D"],
                self._motor_config[i]["S"],
                self._motor_config[i]["offset"],
            )
            for i in range(len(self._motors))
        ]

        return joint_positions



    def open_gripper(self, ts: float) -> None:

        next_time = time.perf_counter()

        for step_idx in range(10):

            raw_cmd = self._motor_config[5]["raw_max"]

            self._group_executor.addCmd(
                self._motors[5].stageSetGoalPosition(raw_cmd)
            )

            self._group_executor.executeWrite()

            self._group_executor.clearStagedWriteCommands()

            next_time += ts

            remaining = next_time - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

            elif remaining < -0.005:
                print(
                    f"[OmxDriver] WARNING: "
                    f"gripper open step {step_idx} overran by "
                    f"{-remaining * 1e3:.2f} ms."
                )
    
    def close_gripper(self, ts: float) -> None:

        next_time = time.perf_counter()

        for step_idx in range(10):

            raw_cmd = self._motor_config[5]["raw_min"]

            self._group_executor.addCmd(
                self._motors[5].stageSetGoalPosition(raw_cmd)
            )

            self._group_executor.executeWrite()

            self._group_executor.clearStagedWriteCommands()

            next_time += ts

            remaining = next_time - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

            elif remaining < -0.005:
                print(
                    f"[OmxDriver] WARNING: "
                    f"gripper close step {step_idx} overran by "
                    f"{-remaining * 1e3:.2f} ms."
                )



    # =========================================================================
    # EXECUÇÃO DE TRAJETÓRIA MULTI-JUNTA
    # =========================================================================

    def execute_trajectory(
        self,
        traj: list[list[float]],
        ts: float
    ) -> None:

       
        next_time = time.perf_counter()

        for step_idx, points in enumerate(traj):

            # ----------------------------------------------------------------
            # Validação de tamanho
            # ----------------------------------------------------------------

            if len(points) != len(self._motors):
                raise ValueError(
                    f"Trajectory point size ({len(points)}) "
                    f"!= number of motors ({len(self._motors)})."
                )
            
            print(len(points))

            # ----------------------------------------------------------------
            # Stage dos comandos
            # ----------------------------------------------------------------

            for i, motor in enumerate(self._motors):

                raw_cmd = joint_to_raw(
                    points[i],
                    self._motor_config[i]["D"],
                    self._motor_config[i]["S"],
                    self._motor_config[i]["offset"],
                )

                raw_cmd = clamp_raw(
                    raw_cmd,
                    self._motor_config[i]["raw_min"],
                    self._motor_config[i]["raw_max"],
                    motor_id=self._motor_config[i]["id"],
                )

                self._group_executor.addCmd(
                    motor.stageSetGoalPosition(raw_cmd)
                )

            # ----------------------------------------------------------------
            # Sync write
            # ----------------------------------------------------------------

            self._group_executor.executeWrite()

            print(self._group_executor.executeWrite())

            self._group_executor.clearStagedWriteCommands()

            # ----------------------------------------------------------------
            # Controle temporal absoluto
            # ----------------------------------------------------------------

            next_time += ts

            remaining = next_time - time.perf_counter()

            # ----------------------------------------------------------------
            # Pequenos overruns são normais
            # ----------------------------------------------------------------

            if remaining > 0:
                time.sleep(remaining)

            elif remaining < -0.005:
                print(
                    f"[OmxDriver] WARNING: "
                    f"step {step_idx} overran by "
                    f"{-remaining * 1e3:.2f} ms."
                )

    # =========================================================================
    # EXECUÇÃO DE UMA ÚNICA JUNTA
    # =========================================================================

    def execute_trajectory_one_joint(
        self,
        traj: list[list[float]],
        ts: float,
        joint_idx: int
    ) -> None:

        motor = self._motors[joint_idx]

        cfg = self._motor_config[joint_idx]

        next_time = time.perf_counter()

        #print(traj)

        for step_idx, points in enumerate(traj):

            # ----------------------------------------------------------------
            # Segurança
            # ----------------------------------------------------------------

            if len(points) == 0:
                raise ValueError("Empty trajectory point.")

            # ----------------------------------------------------------------
            # Conversão junta -> encoder
            # ----------------------------------------------------------------



            raw_cmd = joint_to_raw(
                points[0],
                cfg["D"],
                cfg["S"],
                cfg["offset"]
            )


            raw_cmd = clamp_raw(
                raw_cmd,
                cfg["raw_min"],
                cfg["raw_max"],
                cfg["id"]
            )

            # ----------------------------------------------------------------
            # Stage + write
            # ----------------------------------------------------------------

            self._group_executor.addCmd(
                motor.stageSetGoalPosition(raw_cmd)
            )

            self._group_executor.executeWrite()

            print(f"sent raw_cmd={raw_cmd}")
            self._group_executor.clearStagedWriteCommands()

            # ----------------------------------------------------------------
            # Controle temporal absoluto
            # ----------------------------------------------------------------

            next_time += ts

            remaining = next_time - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

            elif remaining < -0.005:
                print(
                    f"[OmxDriver] WARNING: "
                    f"step {step_idx} overran by "
                    f"{-remaining * 1e3:.2f} ms."
                )
