import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from builtin_interfaces.msg import Duration

from omx_ros2.omx_driver import OmxDriver

# Serviços definidos em omx_interfaces
from omx_interfaces.srv import ExecuteTrajectory, SetGripper


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']


class OmxDriverNode(Node):

    def __init__(self):
        super().__init__('omx_driver_node')

        # ----------------------------------------------------------------
        # Driver
        # ----------------------------------------------------------------
        self.driver = OmxDriver()

        # ----------------------------------------------------------------
        # Publisher de joint states
        # ----------------------------------------------------------------
        self.publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        # ----------------------------------------------------------------
        # Serviço de execução de trajetória
        # ----------------------------------------------------------------
        self.execute_srv = self.create_service(
            ExecuteTrajectory,
            'execute_trajectory',
            self.execute_trajectory_callback
        )

        # ----------------------------------------------------------------
        # Serviço do gripper (SET GRIPPER)
        # ----------------------------------------------------------------
        self.gripper_srv = self.create_service(
            SetGripper,
            'gripper_control',
            self.set_gripper_callback
        )

        self.get_logger().info('OmxDriverNode iniciado.')

    # ====================================================================
    # Joint states
    # ====================================================================

    def timer_callback(self):
        try:
            positions = self.driver.read_joint_positions()
        except Exception as e:
            self.get_logger().error(f'Erro ao ler posições: {e}')
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(positions[:5])

        self.publisher.publish(msg)

    # ====================================================================
    # Execute trajectory
    # ====================================================================

    def execute_trajectory_callback(self, request, response):
        try:
            traj_msg = request.trajectory

            if not traj_msg.points:
                response.success = False
                return response

            traj = [list(p.positions) for p in traj_msg.points]

            # Extrai time_from_start de cada ponto em segundos
            time_from_start = [
                p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                for p in traj_msg.points
            ]

            self.driver.execute_trajectory(traj, time_from_start)
            response.success = True

        except Exception as e:
            self.get_logger().error(f'Erro na execução: {e}')
            response.success = False

        return response

    # ====================================================================
    # SET GRIPPER (novo serviço)
    # ====================================================================

    def set_gripper_callback(self, request, response):
        """
        request.position:
            0.0 -> fechado
            1.0 -> aberto
        """
        try:
            pos = float(request.position)

            if pos > 0.5:
                self.get_logger().info('Abrindo gripper...')
                self.driver.open_gripper(0.1)
            else:
                self.get_logger().info('Fechando gripper...')
                self.driver.close_gripper(0.1)

            response.success = True
            response.message = "OK"

        except Exception as e:
            self.get_logger().error(f'Erro no gripper: {e}')
            response.success = False
            response.message = str(e)

        return response

    # ====================================================================
    # Cleanup
    # ====================================================================

    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OmxDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()