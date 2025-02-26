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
                
                # 执行运动控制
                self.publish_cmd_vel(robot_name, duration, linear_x, angular_z)
                response.response_text = f"Motion command sent to {robot_name}"
                
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析失败: {str(e)}"
            self.get_logger().error(error_msg)
            response.response_text = error_msg
        except KeyError as e:
            error_msg = f"缺少必要参数: {str(e)}"
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
            "angular_z": float(raw_args.get("angular_z", 0.0))
        }

    def publish_cmd_vel(self, robot_name, duration, linear_x=0.0, angular_z=0.0):
        # 定义标准QoS配置
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 获取或创建发布者
        topic_name = f"/{robot_name}/cmd_vel"
        if robot_name not in self.cmd_vel_publishers:
            self.cmd_vel_publishers[robot_name] = self.create_publisher(
                Twist, 
                topic_name, 
                qos_profile=qos_profile  # 修正QoS配置
            )
            
        # 非阻塞实现：使用定时器
        motion_id = f"{robot_name}_{time.time()}"
        self.active_motions[motion_id] = {
            "publisher": self.cmd_vel_publishers[robot_name],
            "end_time": self.get_clock().now() + Duration(seconds=duration),
            "twist": Twist(
                linear=Vector3(x=linear_x),
                angular=Vector3(z=angular_z)
            )
        }
        
        # 创建高频定时器
        self.create_timer(0.1, lambda: self.motion_control_loop(motion_id))
        self.get_logger().info(f"Started motion {motion_id}")

    def motion_control_loop(self, motion_id):
        """非阻塞运动控制核心逻辑"""
        if motion_id not in self.active_motions:
            return
            
        motion = self.active_motions[motion_id]
        current_time = self.get_clock().now()
        
        if current_time < motion["end_time"]:
            # 持续发布速度指令
            motion["publisher"].publish(motion["twist"])
        else:
            # 发布停止指令
            stop_twist = Twist()
            motion["publisher"].publish(stop_twist)
            
            # 清理资源
            del self.active_motions[motion_id]
            self.get_logger().info(f"Motion {motion_id} completed")
def main():
    rclpy.init()
    turtle_robot = TurtleRobot()
    rclpy.spin(turtle_robot)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
