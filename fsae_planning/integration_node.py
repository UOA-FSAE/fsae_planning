import rclpy
from rclpy.node import Node
import math

# 1. IMPORT THE QOS UTILITIES FROM RCLPY
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from fs_msgs.msg import Track, ControlCommand
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class FSDSMockPipeline(Node):
    def __init__(self):
        super().__init__('fsds_mock_pipeline')
        self.get_logger().info("!!! NODE INITIALIZED AND RUNNING SUCCESSFULLY !!!")
        
        # 2. DEFINE A COMPATIBLE BEST-EFFORT PROFILE FOR SIMULATION DATA
        sim_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.current_speed = 0.0
        
        # 3. APPLY THE sim_qos TO YOUR SENSOR SUBSCRIPTIONS
        self.track_sub = self.create_subscription(
            Track, 
            '/fsds/testing_only/track', 
            self.track_callback, 
            sim_qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/fsds/testing_only/odom', 
            self.odom_callback, 
            sim_qos
        )
        
        # Keep your publisher standard (Reliable/Default is fine here)
        self.control_pub = self.create_publisher(ControlCommand, '/fsds/control_command', 10)
        self.path_pub = self.create_publisher(Path, '/fsds/planned_path', 10)

    def odom_callback(self, msg):
        # Read vehicle current speed
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(vx**2 + vy**2)

    def track_callback(self, msg):
        # 1. ALWAYS print the total raw counts coming out of the simulator first
        self.get_logger().info(
            f"--- [RAW SIM DATA RECEIVED] --- Left Cones: {len(msg.cones_left)} | Right Cones: {len(msg.cones_right)}"
        )

        # 2. Extract positions without applying any spatial bounding box restrictions yet
        left_cones = [[c.position.x, c.position.y] for c in msg.cones_left]
        right_cones = [[c.position.x, c.position.y] for c in msg.cones_right]

        # 3. Simple lookahead point calculation (Fallback to 4 meters straight ahead if empty)
        target_x, target_y = 4.0, 0.0
        if left_cones and right_cones:
            target_x = (left_cones[0][0] + right_cones[0][0]) / 2.0
            target_y = (left_cones[0][1] + right_cones[0][1]) / 2.0

        # Calculate tracking steering error angles
        steering_error = math.atan2(target_y, target_x)
        
        # Build the command payload
        cmd = ControlCommand()
        cmd.steering = max(-1.0, min(1.0, steering_error / 0.45))
        cmd.throttle = 0.30  # Low baseline power input
        cmd.brake = 0.0
            
        # 4. ALWAYS print what we are attempting to pass back to the physics engine
        self.get_logger().info(
            f"Publishing Command -> Steering: {cmd.steering:.2f} | Throttle: {cmd.throttle:.2f}"
        )
        
        self.control_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = FSDSMockPipeline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()