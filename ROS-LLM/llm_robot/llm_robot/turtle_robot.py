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
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry  # 新增Odometry导入
from std_msgs.msg import String
from llm_interfaces.srv import ChatGPT
import subprocess
import time
import math  # 新增数学库
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from threading import Lock  # 导入线程锁
# LLM related
import json
from llm_config.user_config import UserConfig

# Global Initialization
config = UserConfig()

class TurtleRobot(Node):
    def __init__(self):
        super().__init__("turtle_robot")
        
        # 初始化里程计订阅（新增）
        self.odom_sub = self.create_subscription(
            Odometry,
            "/tb1/odom",  # 根据实际话题名调整
            self.odom_callback,
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        )
        self.current_pose = None  # 存储当前位姿
        self.current_twist = None  # 存储当前速度
        
        # 初始化控制参数（新增补偿系数）
        self.linear_compensation = 1.05  # 直线运动补偿
        self.angular_compensation = 0.97  # 旋转运动补偿
        self.position_threshold = 0.05  # 位置误差阈值(m)
        self.angle_threshold = math.radians(5)  # 角度误差阈值(rad)

        # 原有初始化保持不变...
        self.publisher = self.create_publisher(Twist, "/tb1/cmd_vel", 10)
        self.function_call_server = self.create_service(
            ChatGPT, "/ChatGPT_function_call_service", self.function_call_callback
        )
        self.get_logger().info("TurtleRobot node has been initialized")
        self.cmd_vel_publishers = {}
        self.active_motions = {}
        self.status_publisher = self.create_publisher(String, "/turtle_robot/status", 10)
        self.timer = self.create_timer(1.0, self.publish_status)

        # 新增校准服务（关键修改）
        self.calibration_srv = self.create_service(
            ChatGPT, "/calibration_service", self.calibration_callback
        )
        self._status_lock = Lock()  # 创建互斥锁对象
        self._current_status = "IDLE"  # 私有变量避免直接访问

    @property
    def status(self):
        """线程安全的状态读取方法"""
        with self._status_lock:  # 进入with语句自动获取锁
            return self._current_status  # 返回当前状态的副本

    @status.setter
    def status(self, value):
        """线程安全的状态写入方法"""
        with self._status_lock:  # 写操作也加锁
            old = self._current_status
            self._current_status = value
            self.get_logger().info(f"Status changed from {old} to {value}")
            
    def odom_callback(self, msg):
        """里程计回调（新增）"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    # 原有publish_status保持不变...

    def _quaternion_to_yaw(self, q):
        """四元数转偏航角（新增）"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def motion_control_loop(self, motion_id):
        """改进后的运动控制循环（关键修改）"""
        if motion_id not in self.active_motions:
            return

        motion = self.active_motions[motion_id]
        elapsed_time = self.get_clock().now() - motion["start_time"]
        
        # 动态调整持续时间（新增补偿）
        adjusted_duration = self._adjust_duration(
            motion["twist"].linear.x,
            motion["twist"].angular.z,
            motion["duration"].nanoseconds / 1e9
        )
        
        # 双重停止条件（新增位置检查）
        time_condition = elapsed_time >= adjusted_duration
        position_condition = self._check_position_target(motion)
        
        if not (time_condition or position_condition):
            motion["publisher"].publish(motion["twist"])
        else:
            self._smooth_stop(motion)  # 修改为平滑停止
            if motion_id in self.active_motions:
                del self.active_motions[motion_id]
                self.get_logger().info(f"Motion {motion_id} completed")

    def _adjust_duration(self, linear_x, angular_z, original_duration):
        """动态补偿持续时间（新增）"""
        if abs(linear_x) > 0.1:
            return Duration(seconds=original_duration * self.linear_compensation)
        elif abs(angular_z) > 0.1:
            return Duration(seconds=original_duration * self.angular_compensation)
        return Duration(seconds=original_duration)

    def _check_position_target(self, motion):
        """基于里程计的位置检查（新增）"""
        if self.current_pose is None:
            return False

        # 计算理论位移
        target_linear = motion["twist"].linear.x * motion["duration"].nanoseconds / 1e9
        target_angular = motion["twist"].angular.z * motion["duration"].nanoseconds / 1e9
        
        # 计算实际位移（简化处理）
        current_x = self.current_pose.position.x
        current_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        
        # 判断误差（新增阈值检查）
        linear_error = abs(target_linear - current_x)
        angular_error = abs(target_angular - current_yaw)
        
        return linear_error < self.position_threshold or angular_error < self.angle_threshold

    def _smooth_stop(self, motion):
        """平滑停止机制（新增）"""
        for _ in range(5):
            motion["twist"].linear.x *= 0.7
            motion["twist"].angular.z *= 0.7
            motion["publisher"].publish(motion["twist"])
            time.sleep(0.02)
        motion["publisher"].publish(Twist())
        
    def calibration_callback(self, request, response):
        """校准服务回调（新增）"""
        try:
            self._perform_calibration()
            response.response_text = json.dumps({
                "linear_compensation": self.linear_compensation,
                "angular_compensation": self.angular_compensation
            })
        except Exception as e:
            response.response_text = f"Calibration failed: {str(e)}"
        return response

    def _perform_calibration(self):
        """执行三步校准（新增）"""
        # 直线校准
        self._execute_calibration_movement(linear_x=0.5, duration=2.0)
        actual_distance = self.current_pose.position.x
        self.linear_compensation = 2.0 / actual_distance  # 理论值2m
        
        # 旋转校准
        self._execute_calibration_movement(angular_z=0.5, duration=6.283)  # 理论转360度
        final_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        self.angular_compensation = 6.283 / abs(final_yaw)
        
    def _execute_calibration_movement(self, **kwargs):
        """执行校准运动（新增）"""
        self.publish_cmd_vel("tb1", **kwargs)
        while any("tb1" in mid for mid in self.active_motions):
            time.sleep(0.1)

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
            # 强制转换为字典类型处理
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args)
            else:
                parsed_args = raw_args
            
            # 新增类型强制转换和范围检查
            validated = {
                "robot_name": str(parsed_args["robot_name"]),
                "duration": max(0.0, float(parsed_args["duration"])),  # 确保浮点型
                "linear_x": max(-1.0, min(1.0, float(parsed_args["linear_x"]))),
                "angular_z": max(-1.0, min(1.0, float(parsed_args.get("angular_z", 0.0))))  # 处理可能缺失的字段
            }
            return validated
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.get_logger().error(f"参数校验失败: {str(e)}")
            return None

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
        
        self.create_timer(1/30, lambda: self.motion_control_loop(motion_id))
        self.get_logger().info(f"Started motion {motion_id}")


def main():
    rclpy.init()
    turtle_robot = TurtleRobot()
    rclpy.spin(turtle_robot)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
