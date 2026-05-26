import rclpy
from rclpy.node import Node
import math

from omx_ros2.omx_kinematic import omxKinematicClass

from omx_interfaces.srv import (
    GenerateTrajectory,
    ExecuteTrajectory,
    SetGripper
)

# ============================================================================
# PARÂMETROS DO ROBÔ
# ============================================================================

L1 = 0.036
L2 = 0.040
L3 = 0.040
L4 = 0.040
L5 = 0.124
L6 = 0.040
L7 = 0.130

JOINT_MIN = [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi]
JOINT_MAX = [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2,  math.pi]

FORBIDDEN_REGIONS = [
    {"z_max": 0.01},
]

DEFAULT_TS = 0.01

# ============================================================================
# GRIPPER MAP
# ============================================================================

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0


# ============================================================================
# CONFIGURAÇÕES DE TRAJETÓRIA
# ============================================================================

CONFIG_REST      = [0.0,  0.0,  0.0,  0.0,  0.0]
CONFIG_ABOVE_OBJ = [0.4, -0.3,  0.3,  0.2,  0.0]
CONFIG_GRASP     = [0.4, -0.5,  0.5,  0.3,  0.0]
CONFIG_TRANSPORT = [0.0,  0.0,  0.0,  0.0,  0.0]
CONFIG_ABOVE_DST = [-0.4, -0.3, 0.3,  0.2,  0.0]
CONFIG_PLACE     = [-0.4, -0.5, 0.5,  0.3,  0.0]


# ============================================================================
# NODE
# ============================================================================

class TaskSupervisorNode(Node):

    def __init__(self):
        super().__init__('task_supervisor_node')

        self.robot = omxKinematicClass(L1, L2, L3, L4, L5, L6, L7)

        self.gen_client = self.create_client(
            GenerateTrajectory, 'generate_trajectory'
        )

        self.exec_client = self.create_client(
            ExecuteTrajectory, 'execute_trajectory'
        )

        self.grip_client = self.create_client(
            SetGripper, 'gripper_control'
        )

        self.get_logger().info("TaskSupervisorNode iniciado")

        for client, name in [
            (self.gen_client, 'generate_trajectory'),
            (self.exec_client, 'execute_trajectory'),
            (self.grip_client, 'gripper_control'),
        ]:
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'Aguardando {name}...')

    # ====================================================================
    # TRAJETÓRIA
    # ====================================================================

    def analyze_trajectory(self, traj_msg) -> bool:
        for i, point in enumerate(traj_msg.points):
            for j, pos in enumerate(point.positions):
                if pos < JOINT_MIN[j] or pos > JOINT_MAX[j]:
                    self.get_logger().error(f'Limite violado ponto {i}')
                    return False
        return True

    def check_collision(self, traj_msg) -> bool:
        for i, point in enumerate(traj_msg.points):
            q = list(point.positions)
            frames = self.robot.forward_kinematics(*q)
            T = frames[-1]

            x, y, z = T[0, 3], T[1, 3], T[2, 3]

            for region in FORBIDDEN_REGIONS:
                if z <= region["z_max"]:
                    self.get_logger().error(f'Colisão no ponto {i}')
                    return False

        return True

    # ====================================================================
    # SERVIÇOS
    # ====================================================================

    def _call_generate(self, qi, qf, ts=DEFAULT_TS):
        req = GenerateTrajectory.Request()
        req.qi = list(qi)
        req.qf = list(qf)
        req.ts = ts

        future = self.gen_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _call_execute(self, traj_msg, ts=DEFAULT_TS):
        req = ExecuteTrajectory.Request()
        req.trajectory = traj_msg
        req.ts = ts

        future = self.exec_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    # ============================================================
    # GRIPPER CORRIGIDO (float64 position)
    # ============================================================

    def _call_gripper(self, position: float):
        req = SetGripper.Request()
        req.position = float(position)

        future = self.grip_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        if result is not None:
            self.get_logger().info(
                f"Gripper: success={result.success}, message={result.message}"
            )

        return result

    # ====================================================================
    # MOVIMENTO
    # ====================================================================

    def _safe_move(self, qi, qf, label=''):
        self.get_logger().info(f'[{label}] Gerando traj...')

        result = self._call_generate(qi, qf)

        if result is None or not result.success:
            return False

        traj = result.trajectory

        if not self.analyze_trajectory(traj):
            return False

        if not self.check_collision(traj):
            return False

        exec_result = self._call_execute(traj)

        return exec_result is not None and exec_result.success

    # ====================================================================
    # PICK AND PLACE
    # ====================================================================

    def pick_and_place(self):

        self.get_logger().info("=== PICK AND PLACE ===")

        # abre
        self._call_gripper(GRIPPER_OPEN)

        if not self._safe_move(CONFIG_REST, CONFIG_ABOVE_OBJ, "REST->OBJ"):
            return False

        if not self._safe_move(CONFIG_ABOVE_OBJ, CONFIG_GRASP, "OBJ->GRASP"):
            return False

        # fecha
        self._call_gripper(GRIPPER_CLOSE)

        if not self._safe_move(CONFIG_GRASP, CONFIG_TRANSPORT, "GRASP->TRANS"):
            return False

        if not self._safe_move(CONFIG_TRANSPORT, CONFIG_ABOVE_DST, "TRANS->DST"):
            return False

        if not self._safe_move(CONFIG_ABOVE_DST, CONFIG_PLACE, "DST->PLACE"):
            return False

        # abre
        self._call_gripper(GRIPPER_OPEN)

        if not self._safe_move(CONFIG_PLACE, CONFIG_REST, "PLACE->REST"):
            return False

        self.get_logger().info("=== FINALIZADO ===")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = TaskSupervisorNode()

    try:
        node.pick_and_place()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()