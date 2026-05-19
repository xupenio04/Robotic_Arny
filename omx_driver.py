from dynamixel_easy_sdk import *
import time
import math

MOTOR_CONFIG = [
    {"id": 11, "D":  1, "S": 4096, "offset":     0, "raw_min": -1536, "raw_max":  1536},  # Joint 1
    {"id": 12, "D": -1, "S": 4096, "offset":  3072, "raw_min":  3072, "raw_max":  1024},  # Joint 2
    {"id": 13, "D": -1, "S": 4096, "offset":  1024, "raw_min":  1024, "raw_max":  2560},  # Joint 3
    {"id": 14, "D": -1, "S": 4096, "offset":  2048, "raw_min":  1024, "raw_max":  3072},  # Joint 4
    {"id": 15, "D":  1, "S": 4096, "offset": -2048, "raw_min": -4096, "raw_max":     0},  # Joint 5
    {"id": 16, "D":  1, "S": 4096, "offset": -2048, "raw_min":  2048, "raw_max":  2300},  # Joint 6
]

PORT      = "/dev/ttyACM0"
BAUD_RATE = 1000000


def raw_to_joint(raw: int, D: int, S: int, offset: int) -> float:
    return D * (raw - offset) * (2.0 * math.pi / S)


def joint_to_raw(theta: float, D: int, S: int, offset: int) -> int:
    return int(round(D * theta * (S / (2.0 * math.pi)) + offset))


def clamp_raw(raw: int, raw_min: int, raw_max: int, motor_id: int) -> int:
    lo, hi = min(raw_min, raw_max), max(raw_min, raw_max)
    clamped = max(lo, min(hi, raw))
    if clamped != raw:
        print(
            f"[OmxDriver] WARNING: motor {motor_id} raw command {raw} "
            f"clamped to {clamped} (limits [{raw_min}, {raw_max}])."
        )
    return clamped


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

        # 3. Create motor objects (order matches MOTOR_CONFIG / joint ordering)
        self._motors = [
            self._connector.createMotor(cfg["id"])
            for cfg in self._motor_config
        ]

        # 4. Configure operating mode (torque must be OFF while changing mode)
        for motor in self._motors:
            motor.disableTorque()

        for motor in self._motors:
            motor.setOperatingMode(OperatingMode.POSITION)

        for motor in self._motors:
            motor.enableTorque()

        print(f"[OmxDriver] Connected on {port} @ {baud_rate} Bd — "
              f"{len(self._motors)} motors ready.")

    def __del__(self):
        try:
            for motor in self._motors:
                motor.disableTorque()
            print("[OmxDriver] Torque disabled — driver shut down.")
        except Exception:
            pass 

    
    def read_joint_positions(self) -> list[float]:
        """Read current motor encoders and return joint angles [rad].

        Returns
        -------
        list[float]
            Joint angles in radians, ordered consistently with the kinematic
            model (joint 1 … joint N).
        """
        # Stage read commands for every motor
        for motor in self._motors:
            self._group_executor.addCmd(motor.stageGetPresentPosition())

        # Execute a single synchronised read
        raw_positions = self._group_executor.executeRead()
        self._group_executor.clearStagedReadCommands()

        # Convert raw encoder values → joint angles [rad]
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

    # ──────────────────────────────────────────
    #  Trajectory execution  (Section 5.4)
    # ──────────────────────────────────────────

    def execute_trajectory(self, traj: list[list[float]], ts: float) -> None:

        for step_idx, points in enumerate(traj):
            t_start = time.perf_counter()

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
                self._group_executor.addCmd(motor.stageSetGoalPosition(raw_cmd))

            self._group_executor.executeWrite()
            self._group_executor.clearStagedWriteCommands()

            elapsed = time.perf_counter() - t_start
            remaining = ts - elapsed
            if remaining < 0:
                print(f"[OmxDriver] WARNING: step {step_idx} overran ts by "
                    f"{-remaining * 1e3:.2f} ms.")
            else:
                time.sleep(remaining)


    def execute_trajectory_one_joint(self, traj: list[list[float]], ts: float, joint_idx: int) -> None:

        motor = self._motors[joint_idx]
        cfg   = self._motor_config[joint_idx]

        for step_idx, points in enumerate(traj):
            t_start = time.perf_counter()

            raw_cmd = joint_to_raw(points[0], cfg["D"], cfg["S"], cfg["offset"])
            raw_cmd = clamp_raw(raw_cmd, cfg["raw_min"], cfg["raw_max"], cfg["id"])

            self._group_executor.addCmd(motor.stageSetGoalPosition(raw_cmd))
            self._group_executor.executeWrite()
            self._group_executor.clearStagedWriteCommands()

            elapsed   = time.perf_counter() - t_start
            remaining = ts - elapsed
            if remaining < 0:
                print(f"[OmxDriver] WARNING: step {step_idx} overran ts by {-remaining * 1e3:.2f} ms.")
            else:
                time.sleep(remaining)
