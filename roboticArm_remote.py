import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from argparse import ArgumentParser
from frankx import *
from time import sleep

class DisplacementSubscriber(Node):
    def __init__(self, robot):
        super().__init__('displacement_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/displacement',
            self.listener_callback,
            10)
        self.subscription  # 防止垃圾回收

        self.robot = robot  # 連接 Franka 機器人
        self.delta_range = 0.05  # 每次最大移動範圍
        self.initial_position = [ 0.3, 0, 0.5]
        
        # 先移動到初始姿勢
        joint_motion = JointMotion([0, -0.796, 0, -2.329, 0, 1.53, 0.785])
        robot.move(joint_motion)

        # 啟動 ImpedanceMotion 並保持控制
        self.impedance_motion = ImpedanceMotion(200.0, 20.0)  # 設定阻抗控制
        self.robot_thread = self.robot.move_async(self.impedance_motion)
        sleep(0.1)

        self.initial_target = self.impedance_motion.target
        self.get_logger().info(f'Initial target: {self.initial_target}')

    def listener_callback(self, msg):
        displacement = msg.data
        if len(displacement) == 3:
            x, y, z = displacement
            self.apply_relative_motion(x, y, z)
        else:
            self.get_logger().warn('Received displacement does not have exactly 3 values.')

    def apply_relative_motion(self, delta_x, delta_y, delta_z):
        """ 限制相對變化範圍並執行機器人運動 """

        # 確保 impedance_motion 物件存在
        if hasattr(self, 'impedance_motion'):
            self.impedance_motion.target = Affine(
                self.initial_position[0] + delta_x,
                self.initial_position[1] + delta_y,
                self.initial_position[2] + delta_z
            )
            sleep(1/60)
            self.get_logger().info(f'delta_x={delta_x:.4f}')
            self.get_logger().info(f'Robot moved: dx={self.initial_position[0] + delta_x:.4f}, dy={self.initial_position[1] + delta_y:.4f}, dz={self.initial_position[2] + delta_z:.4f}')
        else:
            self.get_logger().error('Impedance motion is not initialized!')

    def stop_motion(self):
        """ 結束 ImpedanceMotion """
        if hasattr(self, 'impedance_motion'):
            self.impedance_motion.finish()
            self.robot_thread.join()
            self.get_logger().info("Impedance motion finished.")

def main(args=None):
    rclpy.init(args=args)

    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.0.2', help='FCI IP of the robot')
    args = parser.parse_args()

    robot = Robot(args.host)
    robot.set_default_behavior()
    robot.recover_from_errors()
    robot.set_dynamic_rel(0.15)  # 設定機器人動態

    subscriber = DisplacementSubscriber(robot)
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop_motion()  # 終止阻抗控制
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
