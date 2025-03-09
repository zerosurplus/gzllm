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
# This file contains the behavior of the robot.
# It includes a list of functions for the robot to perform,
# such as publishing a cmd_vel message to control the movement of the robot.
# To customize the robot's behavior,
# modify the functions in this file to customize the behavior of your robot
# and don't forget to modify the corresponding real functions in llm_robot/turtle_robot.py
#
# Author: Herman Ye @Auromix

# Example robot functions list for the TurtleSim
# The user can add, remove, or modify the functions in this list
robot_functions_list_1 = [
    {
        "name": "publish_cmd_vel",
        "description": "Control robot movement and rotation.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration (seconds).",
                },
                "linear_x": {
                    "type": "number",
                    "description": "Linear velocity along x-axis.",
                },
                "angular_z": {
                    "type": "number",
                    "description": "Angular velocity around z-axis.",
                },
            },
            "required": [
                "robot_name",
                "duration",
                "linear_x",
                "angular_z",
            ],
        },
    },
    {
        "name": "move_in_circle",
        "description": "Move robot in a circle.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "radius": {
                    "type": "number",
                    "description": "Circle radius.",
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration (seconds).",
                },
            },
            "required": [
                "robot_name",
                "radius",
                "duration",
            ],
        },
    },
    {
        "name": "move_in_rectangle",
        "description": "Move robot in a rectangle.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "length": {
                    "type": "number",
                    "description": "Rectangle length.",
                },
                "width": {
                    "type": "number",
                    "description": "Rectangle width.",
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration (seconds).",
                },
            },
            "required": [
                "robot_name",
                "length",
                "width",
                "duration",
            ],
        },
    },
    {
        "name": "navigate_to_position",
        "description": "Navigate robot to a position.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "x": {
                    "type": "number",
                    "description": "Target x-coordinate.",
                },
                "y": {
                    "type": "number",
                    "description": "Target y-coordinate.",
                },
            },
            "required": [
                "robot_name",
                "x",
                "y",
            ],
        },
    },
    {
        "name": "navigate_to_robot",
        "description": "Navigate robot to another robot.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "target_robot_name": {
                    "type": "string",
                    "description": "Target robot name.",
                },
            },
            "required": [
                "robot_name",
                "target_robot_name",
            ],
        },
    },
]

robot_functions_list_multi_robot = [
    {
        "name": "publish_cmd_vel",
        "description": "Control robot movement and rotation.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration (seconds).",
                },
                "linear_x": {
                    "type": "number",
                    "description": "Linear velocity along x-axis.",
                },
                "angular_z": {
                    "type": "number",
                    "description": "Angular velocity around z-axis.",
                },
            },
            "required": [
                "robot_name",
                "duration",
                "linear_x",
                "angular_z",
            ],
        },
    },
    {
        "name": "move_in_circle",
        "description": "Move robot in a circle.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "radius": {
                    "type": "number",
                    "description": "Circle radius.",
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration (seconds).",
                },
            },
            "required": [
                "robot_name",
                "radius",
                "duration",
            ],
        },
    },
    {
        "name": "move_in_rectangle",
        "description": "Move robot in a rectangle.",
        "parameters": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Robot name ('tb1', 'tb2', 'tb3', 'tb4').",
                },
                "length": {
                    "type": "number",
                    "description": "Rectangle length.",
                },
                "width": {
                    "type": "number",
                    "description": "Rectangle width.",
                },
                "duration": {
                    "type": "number",
                    "description": "Movement duration (seconds).",
                },
            },
            "required": [
                "robot_name",
                "length",
                "width",
                "duration",
            ],
        },
    },
]

class RobotBehavior:
    """
    This class contains the behavior of the robot.
    It is used in llm_config/user_config.py to customize the behavior of the robot.
    """

    def __init__(self):
        self.robot_functions_list = robot_functions_list_multi_robot


if __name__ == "__main__":
    pass
