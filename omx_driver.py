from dynamixel_easy_sdk import *
import time
import math

MOTOR_CONFIG = [
    {"id": 1, "D":  1, "S": 4096, "offset":     0, "raw_min": -1536, "raw_max":  1536},  # Joint 1
    {"id": 2, "D": -1, "S": 4096, "offset":  3072, "raw_min":  3072, "raw_max":  1024},  # Joint 2
    {"id": 3, "D": -1, "S": 4096, "offset":  1024, "raw_min":  1024, "raw_max":  2560},  # Joint 3
    {"id": 4, "D": -1, "S": 4096, "offset":  2048, "raw_min":  1024, "raw_max":  3072},  # Joint 4
    {"id": 5, "D":  1, "S": 4096, "offset": -2048, "raw_min": -4096, "raw_max":     0},  # Joint 5
    {"id": 6, "D":  1, "S": 4096, "offset": -2048, "raw_min":  2048, "raw_max":  2300},  # Joint 6
]

PORT      = "/dev/ttyACM0"
BAUD_RATE = 57600


def raw_to_joint(raw: int, D: int, S: int, offset: int) -> float:
    """Convert raw encoder count to joint angle [rad].

    θ = D * (raw - offset) * (2π / S)
    """
    return D * (raw - offset) * (2.0 * math.pi / S)


def joint_to_raw(theta: float, D: int, S: int, offset: int) -> int:
    """Convert joint angle [rad] to raw encoder count.

    raw = D * θ * (S / 2π) + offset
    """
    return int(round(D * theta * (S / (2.0 * math.pi)) + offset))


def clamp_raw(raw: int, raw_min: int, raw_max: int, motor_id: int) -> int:
    """Clamp a raw encoder command to the motor's safe hardware limits.

    Note: raw_min may be greater than raw_max for motors whose encoder range
    wraps around (e.g. motor 2: min=3072, max=1024 in the opposite direction).
    In that case the valid range is [raw_max, raw_min] and clamping is applied
    symmetrically.

    A warning is printed whenever clamping occurs.
    """
    lo, hi = min(raw_min, raw_max), max(raw_min, raw_max)
    clamped = max(lo, min(hi, raw))
    if clamped != raw:
        print(
            f"[OmxDriver] WARNING: motor {motor_id} raw command {raw} "
            f"clamped to {clamped} (limits [{raw_min}, {raw_max}])."
        )
    return clamped


class OmxDriver:
    """Driver for the OpenManipulator-X robotic arm using Dynamixel motors."""

    # ──────────────────────────────────────────
    #  Initialisation / teardown
    # ──────────────────────────────────────────

    def __init__(
        self,
        port: str = PORT,
        baud_rate: int = BAUD_RATE,
        motor_config: list = MOTOR_CONFIG,
    ):
        """Connect to the control board, configure and enable all motors.

        Steps (as required by the hardware):
          1. Establish serial connection with the control board.
          2. Create a group executor for synchronised read/write.
          3. Instantiate each motor object.
          4. Disable torque → set POSITION operating mode → re-enable torque.
        """
        self._motor_config = motor_config

        # 1. Connect to control board
        self._connector = Connector(port, baud_rate)

        # 2. Group executor for synchronised operations
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
        """Disable torque on all motors when the driver is destroyed."""
        try:
            for motor in self._motors:
                motor.disableTorque()
            print("[OmxDriver] Torque disabled — driver shut down.")
        except Exception:
            pass  # Silently ignore errors during teardown

    # ──────────────────────────────────────────
    #  Joint state acquisition  (Section 5.3)
    # ──────────────────────────────────────────

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

    def execute_trajectory(self, trajectory: list[list[float]], ts: float) -> None:
        """Execute a pre-computed joint-space trajectory.

        Parameters
        ----------
        trajectory : list[list[float]]
            Sequence of joint-position vectors [rad], one per time step.
            Each inner list must have exactly as many elements as there are
            motors/joints.
        ts : float
            Control period [s].  Commands are sent at this fixed interval.
        """
        n_motors = len(self._motors)

        for step_index, joint_positions in enumerate(trajectory):
            t_start = time.perf_counter()

            if len(joint_positions) != n_motors:
                raise ValueError(
                    f"Step {step_index}: expected {n_motors} joint values, "
                    f"got {len(joint_positions)}."
                )

            # Convert joint angles [rad] → raw encoder counts, then clamp
            raw_commands = []
            for i in range(n_motors):
                cfg = self._motor_config[i]
                raw = joint_to_raw(
                    joint_positions[i], cfg["D"], cfg["S"], cfg["offset"]
                )
                raw = clamp_raw(raw, cfg["raw_min"], cfg["raw_max"], cfg["id"])
                raw_commands.append(raw)

            # Stage and send all commands simultaneously
            for i, motor in enumerate(self._motors):
                self._group_executor.addCmd(
                    motor.stageSetGoalPosition(raw_commands[i])
                )

            self._group_executor.executeWrite()
            self._group_executor.clearStagedWriteCommands()

            # ── Timing: busy-wait for the remainder of ts ──────────────────
            elapsed = time.perf_counter() - t_start
            remaining = ts - elapsed

            if remaining < 0:
                print(
                    f"[OmxDriver] WARNING: step {step_index} overran ts by "
                    f"{-remaining * 1e3:.2f} ms."
                )
            else:
                time.sleep(remaining)