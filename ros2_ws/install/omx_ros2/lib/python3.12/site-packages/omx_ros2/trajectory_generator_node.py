
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
