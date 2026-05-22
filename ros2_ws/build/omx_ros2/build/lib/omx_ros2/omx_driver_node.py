import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from omx_ros2.omx_driver import OmxDriver

class OmxDriverNode(Node):

    def __init__(self):
        super().__init__('omx_driver_node')

        # Instância do driver
        self.driver = OmxDriver()

        # Publisher de joint states
        self.publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer: publica a cada 0.1s (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('OmxDriverNode iniciado.')

    def timer_callback(self):
        positions = self.driver.get_joint_positions()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        msg.position = list(positions[:5])

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OmxDriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()