#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
#
# Copyright 2023 Herman Ye @Auromix
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# This example demonstrates simulating function calls for any robot,
# such as controlling velocity and other service commands.
# By modifying the content of this file,
# A calling interface can be created for the function calls of any robot.
# The Python script creates a ROS 2 Node
# that controls the movement of the TurtleSim
# by creating a publisher for cmd_vel messages and a client for the reset service.
# It also includes a ChatGPT function call server
# that can call various functions to control the TurtleSim
# and return the result of the function call as a string.
#
# Author: Herman Ye @Auromix

# ROS related
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,Vector3
from std_msgs.msg import String
from llm_interfaces.srv import ChatGPT
import subprocess
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

# LLM related
import json
from llm_config.user_config import UserConfig

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

        self.status_publisher = self.create_publisher(String, "/turtle_robot/status", 10)
        self.timer = self.create_timer(1.0, self.publish_status)

    def publish_status(self):
        status_msg = String()
        if self.active_motions:
            status_msg.data = "BUSY"
        else:
            status_msg.data = "IDLE"
        self.status_publisher.publish(status_msg)
        
    def function_call_callback(self, request, response):
        try:
            req = json.loads(request.request_text)
            function_name = req["name"]
            
            if function_name == "publish_cmd_vel":
                if any(motion["publisher"].topic_name == f"/{req['arguments']['robot_name']}/cmd_vel" for motion in self.active_motions.values()):
                    response.response_text = "Robot is busy, please try again later"
                    return response
                
                function_args = self._validate_arguments(req.get("arguments", {}))
                if not function_args:
                    raise ValueError("Invalid motion parameters")
                
                self.publish_cmd_vel(
                    function_args["robot_name"],
                    function_args["duration"],
                    function_args["linear_x"],
                    function_args["angular_z"]
                )
                response.response_text = f"Motion command sent to {function_args['robot_name']}"
        except Exception as error:
            self.get_logger().error(f"Service failed: {str(error)}")
            response.response_text = f"Error: {str(error)}"
        
        return response

    def _validate_arguments(self, raw_args):
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON format, using default values")
            parsed_args = {}
        
        # 设置默认值和范围检查
        return {
            "robot_name": parsed_args.get("robot_name", "tb1"),
            "duration": max(0.0, float(parsed_args.get("duration", 0.0))),  # 确保 duration >= 0
            "linear_x": min(max(-1.0, float(parsed_args.get("linear_x", 0.0))), 1.0),
            "angular_z": min(max(-1.0, float(parsed_args.get("angular_z", 0.0))), 1.0)
        }

    def publish_cmd_vel(self, robot_name, duration, linear_x=0.0, angular_z=0.0):
        if any(motion["publisher"].topic_name == f"/{robot_name}/cmd_vel" for motion in self.active_motions.values()):
            self.get_logger().warn(f"Robot {robot_name} is already moving, ignoring new command")
            return
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        topic_name = f"/{robot_name}/cmd_vel"
        if robot_name not in self.cmd_vel_publishers:
            self.cmd_vel_publishers[robot_name] = self.create_publisher(
                Twist, topic_name, qos_profile=qos_profile
            )
        
        motion_id = f"{robot_name}_{time.time()}"
        self.active_motions[motion_id] = {
            "publisher": self.cmd_vel_publishers[robot_name],
            "start_time": self.get_clock().now(),
            "duration": Duration(seconds=duration),
            "twist": Twist(linear=Vector3(x=linear_x), angular=Vector3(z=angular_z))
        }
        
        self.create_timer(0.02, lambda: self.motion_control_loop(motion_id))
        self.get_logger().info(f"Started motion {motion_id}")

    def motion_control_loop(self, motion_id):
        if motion_id not in self.active_motions:
            return
        
        motion = self.active_motions[motion_id]
        elapsed_time = self.get_clock().now() - motion["start_time"]
        
        if elapsed_time < motion["duration"]:
            motion["publisher"].publish(motion["twist"])
        else:
            stop_twist = Twist()
            motion["publisher"].publish(stop_twist)
            
            # 销毁定时器
            timer = motion.pop("timer", None)
            if timer:
                timer.cancel()
            
            if motion_id in self.active_motions:
                del self.active_motions[motion_id]
                self.get_logger().info(f"Motion {motion_id} completed")
def main():
    rclpy.init()
    turtle_robot = TurtleRobot()
    rclpy.spin(turtle_robot)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
