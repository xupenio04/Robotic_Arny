
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Bool
# from builtin_interfaces.msg import Duration

# from omx_ros2.omx_driver import OmxDriver

# # Serviços definidos em omx_interfaces
# from omx_interfaces.srv import ExecuteTrajectory, GripperControl


# JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']


# class OmxDriverNode(Node):

#     def __init__(self):
#         super().__init__('omx_driver_node')

#         # ----------------------------------------------------------------
#         # Driver
#         # ----------------------------------------------------------------
#         self.driver = OmxDriver()

#         # ----------------------------------------------------------------
#         # 6.4 — Publisher de joint states
#         # ----------------------------------------------------------------
#         self.publisher = self.create_publisher(
#             JointState,
#             'joint_states',
#             10
#         )

#         # Timer: publica a cada 0.1 s (10 Hz)
#         self.timer = self.create_timer(0.1, self.timer_callback)

#         # ----------------------------------------------------------------
#         # 6.6 — Serviço de execução de trajetória
#         # ----------------------------------------------------------------
#         self.execute_srv = self.create_service(
#             ExecuteTrajectory,
#             'execute_trajectory',
#             self.execute_trajectory_callback
#         )

#         # ----------------------------------------------------------------
#         # 6.7 — Serviço de controle do gripper
#         # ----------------------------------------------------------------
#         self.gripper_srv = self.create_service(
#             GripperControl,
#             'gripper_control',
#             self.gripper_control_callback
#         )

#         self.get_logger().info('OmxDriverNode iniciado.')

#     # ====================================================================
#     # 6.4 — Callback do timer: lê e publica joint states
#     # ====================================================================

#     def timer_callback(self):
#         try:
#             positions = self.driver.read_joint_positions()
#         except Exception as e:
#             self.get_logger().error(f'Erro ao ler posições: {e}')
#             return

#         msg = JointState()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.name = JOINT_NAMES
#         # Apenas as 5 juntas do braço (motor 6 é o gripper)
#         msg.position = list(positions[:5])

#         self.publisher.publish(msg)

#     # ====================================================================
#     # 6.6 — Callback do serviço de execução de trajetória
#     # ====================================================================

#     def execute_trajectory_callback(self, request, response):
#         """
#         Recebe uma JointTrajectory e executa ponto a ponto no driver.
#         O campo time_from_start do primeiro ponto define o ts.
#         """
#         try:
#             traj_msg: JointTrajectory = request.trajectory
#             ts = request.ts  # período de controle em segundos

#             if not traj_msg.points:
#                 self.get_logger().warn('Trajetória vazia recebida.')
#                 response.success = False
#                 return response

#             # Converte JointTrajectory -> list[list[float]]
#             traj = [
#                 list(point.positions)
#                 for point in traj_msg.points
#             ]

#             self.get_logger().info(
#                 f'Executando trajetória com {len(traj)} pontos, ts={ts} s.'
#             )

#             self.driver.execute_trajectory(traj, ts)

#             response.success = True
#             self.get_logger().info('Trajetória executada com sucesso.')

#         except Exception as e:
#             self.get_logger().error(f'Erro na execução da trajetória: {e}')
#             response.success = False

#         return response

#     # ====================================================================
#     # 6.7 — Callback do serviço de controle do gripper
#     # ====================================================================

#     def gripper_control_callback(self, request, response):
#         """
#         request.open == True  -> abre o gripper
#         request.open == False -> fecha o gripper
#         """
#         try:
#             ts = request.ts if hasattr(request, 'ts') else 0.1

#             if request.open:
#                 self.get_logger().info('Abrindo gripper...')
#                 self.driver.open_gripper(ts)
#             else:
#                 self.get_logger().info('Fechando gripper...')
#                 self.driver.close_gripper(ts)

#             response.success = True

#         except Exception as e:
#             self.get_logger().error(f'Erro no controle do gripper: {e}')
#             response.success = False

#         return response

#     # ====================================================================
#     # Destrutor
#     # ====================================================================

#     def __del__(self):
#         if hasattr(self, 'driver'):
#             self.driver.shutdown()


# def main(args=None):
#     rclpy.init(args=args)
#     node = OmxDriverNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()



import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np

from omx_ros2.trajectory_generator import TrajectoryGenerator

# Serviço definido em omx_interfaces
from omx_interfaces.srv import GenerateTrajectory


JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']

# Número de juntas do braço (sem o gripper)
N_JOINTS = 5

# Limites dinâmicos padrão (rad/s e rad/s²)
DEFAULT_VMAX = [0.35] * N_JOINTS   # ~20 °/s
DEFAULT_AMAX = [0.17] * N_JOINTS   # ~10 °/s²


class TrajectoryGeneratorNode(Node):

    def __init__(self):
        super().__init__('trajectory_generator_node')

        # ----------------------------------------------------------------
        # Gerador de trajetórias
        # ----------------------------------------------------------------
        self.generator = TrajectoryGenerator(
            n=N_JOINTS,
            Vn=DEFAULT_VMAX,
            An=DEFAULT_AMAX,
        )

        # ----------------------------------------------------------------
        # 6.5 — Serviço de geração de trajetória
        # ----------------------------------------------------------------
        self.generate_srv = self.create_service(
            GenerateTrajectory,
            'generate_trajectory',
            self.generate_trajectory_callback
        )

        self.get_logger().info('TrajectoryGeneratorNode iniciado.')

    # ====================================================================
    # 6.5 — Callback do serviço
    # ====================================================================

    def generate_trajectory_callback(self, request, response):
        """
        Request:
            float64[] qi   — configuração inicial [rad]
            float64[] qf   — configuração final   [rad]
            float64   ts   — período de controle  [s]

        Response:
            JointTrajectory trajectory
            bool            success
        """
        try:
            qi = list(request.qi)
            qf = list(request.qf)
            ts = float(request.ts) if request.ts > 0.0 else 0.01

            if len(qi) != N_JOINTS or len(qf) != N_JOINTS:
                self.get_logger().error(
                    f'qi/qf devem ter {N_JOINTS} elementos. '
                    f'Recebido: qi={len(qi)}, qf={len(qf)}.'
                )
                response.success = False
                return response

            self.get_logger().info(
                f'Gerando trajetória: qi={qi}, qf={qf}, ts={ts}'
            )

            # Atualiza o ts do gerador
            self.generator.ts = ts

            # Gera a trajetória sincronizada
            traj, _, _ = self.generator.compute_trajectory(qi, qf)

            # traj shape: (n_steps, N_JOINTS)
            n_steps = traj.shape[0]

            # ----------------------------------------------------------------
            # Monta a mensagem JointTrajectory
            # ----------------------------------------------------------------
            traj_msg = JointTrajectory()
            traj_msg.joint_names = JOINT_NAMES

            for k in range(n_steps):
                point = JointTrajectoryPoint()
                point.positions = list(traj[k, :])

                # time_from_start acumulado
                t_ns = int(k * ts * 1e9)
                point.time_from_start = Duration(
                    sec=t_ns // 1_000_000_000,
                    nanosec=t_ns % 1_000_000_000,
                )

                traj_msg.points.append(point)

            response.trajectory = traj_msg
            response.success = True

            self.get_logger().info(
                f'Trajetória gerada com {n_steps} pontos.'
            )

        except Exception as e:
            self.get_logger().error(f'Erro na geração de trajetória: {e}')
            response.success = False

        return response


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
