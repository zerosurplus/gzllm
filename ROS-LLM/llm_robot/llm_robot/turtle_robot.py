import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import Twist, Vector3
from llm_interfaces.srv import ChatGPT
import json
import time
from llm_config.user_config import UserConfig
from collections import deque
import math  # 添加math库用于计算角度
# 导入 Nav2 客户端库
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped

# Global Initialization
config = UserConfig()

class TurtleRobot(Node):
    def __init__(self):
        super().__init__("turtle_robot")
        # Publisher for cmd_vel
        self.publisher = self.create_publisher(Twist, "/tb1/cmd_vel", 10)
        # Server for function call
        self.function_call_server = self.create_service(
            ChatGPT, "/ChatGPT_function_call_service", self.function_call_callback
        )
        # Node initialization log
        self.get_logger().info("TurtleRobot node has been initialized")

        # 初始化发布者管理器
        self.cmd_vel_publishers = {}
        # 初始化运动控制参数
        self.active_motions = {}  # 新增：跟踪正在进行的运动
        self.motion_queue = deque()  # 用于存储运动任务的队列
        self.current_motion = None  # 当前正在执行的运动任务
        self.current_motion_start_time = None  # 当前运动任务的开始时间
        self.current_motion_duration = 0.0  # 当前运动任务的持续时间
        self.current_motion_twist = Twist()  # 当前运动任务的速度指令
        self.current_motion_publisher = None  # 当前运动任务的发布者

        # 创建定时器
        self.create_timer(0.02, self.motion_control_loop)

        # 初始化 Nav2 客户端
        self.navigator = BasicNavigator()

    def function_call_callback(self, request, response):
        try:
            # 原始请求日志记录
            self.get_logger().debug(f"原始请求: {request.request_text}")
            
            req = json.loads(request.request_text)
            function_name = req["name"]
            
            self.get_logger().info(f"Received function call: {function_name}")
            
            if function_name == "publish_cmd_vel":
                # 参数安全提取
                function_args = self._validate_arguments(req.get("arguments", {}))
                
                if not function_args:
                    raise ValueError("无效的运动参数")
                
                # 参数类型转换
                robot_name = str(function_args["robot_name"])
                duration = max(0.0, float(function_args["duration"]))
                linear_x = min(max(-1.0, float(function_args["linear_x"])), 1.0)
                angular_z = min(max(-1.0, float(function_args["angular_z"])), 1.0)
                
                # 将运动任务添加到队列
                twist = Twist()
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                publisher = self.get_publisher(robot_name)
                self.add_motion_to_queue(robot_name, duration, twist, publisher)
                response.response_text = f"Motion command added to queue for {robot_name}"
            elif function_name == "move_in_circle":
                function_args = self._validate_arguments(req.get("arguments", {}))
                if not function_args:
                    raise ValueError("Invalid motion parameters")
                robot_name = str(function_args["robot_name"])
                radius = float(function_args["radius"])
                duration = float(function_args["duration"])
                self.move_in_circle(robot_name, radius, duration)
                response.response_text = f"Robot {robot_name} moving in a circle with radius {radius} for {duration} seconds"
            elif function_name == "move_in_rectangle":
                function_args = self._validate_arguments(req.get("arguments", {}))
                if not function_args:
                    raise ValueError("Invalid motion parameters")
                robot_name = str(function_args["robot_name"])
                length = float(function_args["length"])
                width = float(function_args["width"])
                duration = float(function_args["duration"])
                self.move_in_rectangle(robot_name, length, width, duration)
                response.response_text = f"Robot {robot_name} moving in a rectangle with length {length} and width {width} for {duration} seconds"
            elif function_name == "navigate_to_position":
                function_args = self._validate_arguments(req.get("arguments", {}))
                if not function_args:
                    raise ValueError("Invalid motion parameters")
                robot_name = str(function_args["robot_name"])
                x = float(function_args["x"])
                y = float(function_args["y"])
                self.navigate_to_position(robot_name, x, y)
                response.response_text = f"Robot {robot_name} navigating to position ({x}, {y})"
            elif function_name == "navigate_to_robot":
                function_args = self._validate_arguments(req.get("arguments", {}))
                if not function_args:
                    raise ValueError("Invalid motion parameters")
                robot_name = str(function_args["robot_name"])
                target_robot_name = str(function_args["target_robot_name"])
                self.navigate_to_robot(robot_name, target_robot_name)
                response.response_text = f"Robot {robot_name} navigating to robot {target_robot_name}"
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {str(e)}"
            self.get_logger().error(error_msg)
            response.response_text = error_msg
        except KeyError as e:
            error_msg = f"Missing required parameter: {str(e)}"
            self.get_logger().error(error_msg)
            response.response_text = error_msg
        except Exception as error:
            self.get_logger().error(f"Service failed: {str(error)}")
            response.response_text = f"Error: {str(error)}"
        return response

    def _validate_arguments(self, raw_args):
        """参数验证与标准化"""
        # 处理字符串类型的参数
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                self.get_logger().error("参数格式错误")
                return None
            return parsed_args
        
        # 处理字典类型的参数
        return {
            "robot_name": str(raw_args.get("robot_name", "tb1")),
            "duration": float(raw_args.get("duration", 0.0)),
            "linear_x": float(raw_args.get("linear_x", 0.0)),
            "angular_z": float(raw_args.get("angular_z", 0.0)),
            "radius": float(raw_args.get("radius", 0.0)),  # 添加 radius 参数
            "length": float(raw_args.get("length", 0.0)),  # 添加 length 参数
            "width": float(raw_args.get("width", 0.0)),   # 添加 width 参数
            "x": float(raw_args.get("x", 0.0)),           # 添加 x 参数
            "y": float(raw_args.get("y", 0.0)),           # 添加 y 参数
            "target_robot_name": str(raw_args.get("target_robot_name", ""))  # 添加 target_robot_name 参数
        }

    def motion_control_loop(self):
        current_time = self.get_clock().now()
        motions_to_remove = []  # 用于存储需要删除的运动任务

        for robot_name, motion in self.active_motions.items():
            elapsed_time = (current_time - motion["start_time"]).nanoseconds / 1e9
            if elapsed_time < motion["duration"]:
                motion["publisher"].publish(motion["twist"])
            else:
                stop_twist = Twist()
                motion["publisher"].publish(stop_twist)
                self.get_logger().info(f"Motion for {robot_name} completed")
                motions_to_remove.append(robot_name)  # 标记需要删除的运动任务

        for motion in list(self.motion_queue):
            if motion["robot_name"] not in self.active_motions:
                self.active_motions[motion["robot_name"]] = motion
                motion["start_time"] = current_time
                self.motion_queue.remove(motion)
                self.get_logger().info(f"Started motion for {motion['robot_name']} with duration {motion['duration']} seconds")

        # 统一删除已完成的运动任务
        for robot_name in motions_to_remove:
            del self.active_motions[robot_name]

    def get_publisher(self, robot_name):
        # 获取或创建发布者
        topic_name = f"/{robot_name}/cmd_vel"
        if robot_name not in self.cmd_vel_publishers:
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10
            )
            self.cmd_vel_publishers[robot_name] = self.create_publisher(
                Twist, 
                topic_name, 
                qos_profile=qos_profile
            )
        return self.cmd_vel_publishers[robot_name]

    def move_in_circle(self, robot_name, radius, duration):
        # Calculate angular velocity for circular motion
        angular_velocity = 1.0  # Example angular velocity
        linear_velocity = angular_velocity * radius
        
        # Create Twist message
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        
        # Get publisher for the robot
        publisher = self.get_publisher(robot_name)
        
        # Add motion to queue
        self.add_motion_to_queue(robot_name, duration, twist, publisher)
        self.get_logger().info(f"Robot {robot_name} moving in a circle with radius {radius} for {duration} seconds")

    def move_in_rectangle(self, robot_name, length, width, duration=None):
        # 参数验证
        if length <= 0 or width <= 0:
            self.get_logger().error("Invalid parameters for move_in_rectangle: length and width must be positive")
            return

        # 计算每边的时间
        if duration is not None and duration > 0:
            time_per_side = duration / 4
        else:
            # 默认时间，可以根据需要调整
            time_per_side = 5.0
            self.get_logger().info(f"No duration specified for move_in_rectangle. Using default time per side: {time_per_side} seconds")

        # 创建 Twist 消息
        twist_length = Twist()
        twist_length.linear.x = length / time_per_side

        twist_width = Twist()
        twist_width.linear.x = width / time_per_side

        twist_rotation = Twist()
        twist_rotation.angular.z = math.pi / 2 / time_per_side  # 90 degree rotation

        # 获取发布者
        publisher = self.get_publisher(robot_name)

        # 添加运动到队列
        self.add_motion_to_queue(robot_name, time_per_side, twist_length, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_rotation, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_width, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_rotation, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_length, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_rotation, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_width, publisher)
        self.add_motion_to_queue(robot_name, time_per_side, twist_rotation, publisher)
        self.get_logger().info(f"Robot {robot_name} moving in a rectangle with length {length} and width {width} for {duration} seconds if specified, otherwise default time per side {time_per_side} seconds")

    def navigate_to_position(self, robot_name, x, y):
        # 使用 Nav2 进行导航
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0  # 默认朝向

        # 发送导航目标
        self.navigator.goToPose(goal_pose)

        # 等待导航完成
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback and feedback.navigation_time > 600.0:  # 超时时间设置为600秒
                self.navigator.cancelTask()
                self.get_logger().info(f"Navigation to position ({x}, {y}) timed out")
                return

        # 检查导航结果
        result = self.navigator.getResult()
        if result == BasicNavigator.TaskResult.SUCCEEDED:
            self.get_logger().info(f"Robot {robot_name} successfully navigated to position ({x}, {y})")
        else:
            self.get_logger().info(f"Robot {robot_name} failed to navigate to position ({x}, {y})")

    def navigate_to_robot(self, robot_name, target_robot_name):
        # Placeholder for navigation to robot logic
        # Example: Get target robot position (placeholder)
        target_x = 1.0  # Replace with actual position
        target_y = 1.0  # Replace with actual position
        
        # Navigate to target position
        self.navigate_to_position(robot_name, target_x, target_y)
        self.get_logger().info(f"Robot {robot_name} navigating to robot {target_robot_name}")

    def add_motion_to_queue(self, robot_name, duration, twist, publisher):
        motion_id = f"{robot_name}_{time.time()}"
        self.motion_queue.append({
            "motion_id": motion_id,
            "robot_name": robot_name,
            "duration": duration,
            "twist": twist,
            "publisher": publisher,
            "start_time": None,
            "end_time": None
        })
        self.get_logger().info(f"Added motion {motion_id} to queue with duration {duration} seconds")

def main():
    rclpy.init()
    turtle_robot = TurtleRobot()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(turtle_robot)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        turtle_robot.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
