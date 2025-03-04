# ROS related
import rclpy
from rclpy.node import Node
from llm_interfaces.srv import ChatGPT
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from threading import Lock  # 导入线程锁
from collections import deque
from typing import Deque, Optional
import copy
# LLM related
import json
import os
import time
import openai
from llm_config.user_config import UserConfig
from openai import OpenAI
import threading

# Global Initialization
config = UserConfig()
openai.api_key = config.openai_api_key
openai.api_base = config.openai_api_base


class ChatGPTNode(Node):
    def __init__(self):
        super().__init__("ChatGPT_node")
        # 初始化发布者、订阅者和客户端
        self.initialization_publisher = self.create_publisher(String, "/llm_initialization_state", 0)
        self.llm_state_publisher = self.create_publisher(String, "/llm_state", 0)
        self.llm_state_subscriber = self.create_subscription(String, "/llm_state", self.state_listener_callback, 0)
        self.llm_input_subscriber = self.create_subscription(String, "/llm_input_audio_to_text", self.llm_callback, 0)
        self.llm_response_type_publisher = self.create_publisher(String, "/llm_response_type", 0)
        self.llm_feedback_publisher = self.create_publisher(String, "/llm_feedback_to_user", 0)
        self.function_call_client = self.create_client(ChatGPT, "/ChatGPT_function_call_service")
        self.function_call_request = ChatGPT.Request()

        # 初始化多小车发布者
        self.cmd_vel_publishers = {}

        self.get_logger().info("ChatGPT Function Call Server is ready")
        self.output_publisher = self.create_publisher(String, "ChatGPT_text_output", 10)
        self.start_timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.chat_history_file = os.path.join(config.chat_history_path, f"chat_history_{self.start_timestamp}.json")
        self.write_chat_history_to_json()
        self.get_logger().info(f"Chat history saved to {self.chat_history_file}")
        self.publish_string("llm_model_processing", self.initialization_publisher)
        self.client = OpenAI(base_url=config.openai_api_base, api_key=config.openai_api_key)

        # 新增任务队列和状态变量
        self.turtle_status = "IDLE"
        self.task_fail_counts = {}  # 新增失败计数器
        self.motion_queue = deque(maxlen=10)  # 替换list为deque（优化性能）
        self.current_request = None  # 修改点：初始化 current_request 属性
        self.status_subscriber = self.create_subscription(
            String, '/turtle_robot/status', self.status_callback, 10)
        self.motion_queue_lock = threading.Lock()  # 新增任务队列锁

    def status_callback(self, msg):
        """机器人状态回调函数（线程安全+deque优化版）"""
        with self.motion_queue_lock:
            # 原子操作：状态更新与队列访问
            old_status = self.turtle_status
            self.turtle_status = msg.data  # 更新当前状态

            # 状态变更日志（非关键操作，不占用锁）
            if old_status != self.turtle_status:
                self.get_logger().info(f"状态变更: {old_status} -> {self.turtle_status}")

            # 仅在IDLE状态处理队列
            if self.turtle_status == "IDLE" and self.motion_queue:
                # 关键修正点：使用popleft()
                next_task = self.motion_queue.popleft()  # 🚩 deque的正确取任务方式

                # 防御性编程：数据校验
                if not isinstance(next_task, str):
                    self.get_logger().error(f"非法任务类型: {type(next_task)}")
                    return

                # 失败次数检查
                if self.task_fail_counts.get(next_task, 0) >= 3:
                    self.get_logger().warn(f"任务重试超限: {next_task[:20]}...")
                    del self.task_fail_counts[next_task]
                    return

                # 发送任务（异步执行）
                try:
                    self.send_motion_request(next_task)
                except Exception as e:
                    self.get_logger().error(f"任务发送失败: {str(e)}")
                    # 重试逻辑
                    self.motion_queue.appendleft(next_task)  # 🚩 使用deque专有方法
                    self.task_fail_counts[next_task] = self.task_fail_counts.get(next_task, 0) + 1

    def send_motion_request(self, request_text):
        """发送运动请求（修复future问题版）"""
        if not isinstance(request_text, str):
            request_text = json.dumps(request_text)

        # 保留原有客户端初始化逻辑
        client = self.create_client(ChatGPT, '/ChatGPT_function_call_service')
        max_retries = 3
        retry_count = 0
        delay_time = 0.5
        while not client.wait_for_service(timeout_sec=delay_time) and retry_count < max_retries:
            self.get_logger().info(f"Service not available, retrying... ({retry_count + 1}/{max_retries})")
            retry_count += 1
            delay_time *= 2

        if retry_count == max_retries:
            self.get_logger().error("Max retries reached. Service call failed.")
            return

        request = ChatGPT.Request()
        request.request_text = request_text
        
        # 关键修改点：绑定数据到future对象
        future = client.call_async(request)
        future.task_data = {  # 新增此行
            "request_text": request_text,  # 直接存储字符串
            "timestamp": time.time()
        }
        
        future.add_done_callback(self.function_call_response_callback)

    def state_listener_callback(self, msg):
        self.get_logger().debug(f"model node get current State:{msg}")

    def publish_string(self, string_to_send, publisher_to_use):
        msg = String()
        msg.data = string_to_send
        publisher_to_use.publish(msg)
        self.get_logger().info(f"Topic: {publisher_to_use.topic_name}\nMessage published: {msg.data}")

    def add_message_to_history(self, role, content="null", function_call=None, name=None):
        message_element_object = {
            "role": role,
            "content": content,
        }
        if name is not None:
            message_element_object["name"] = name
        if function_call is not None:
            message_element_object["function_call"] = function_call
        config.chat_history.append(message_element_object)
        if len(config.chat_history) > config.chat_history_max_length:
            self.get_logger().info(f"Chat history is too long, popping the oldest message: {config.chat_history[0]}")
            config.chat_history.pop(0)
        return config.chat_history

    def generate_chatgpt_response(self, messages_input):
        response = self.client.chat.completions.create(
            model=config.openai_model,
            messages=messages_input,
            functions=config.robot_functions_list,
            function_call="auto",
        )
        return response

    def process_chatgpt_response(self, messages_input):
        chatgpt_response = self.generate_chatgpt_response(messages_input)
        choice, message, content, function_call, function_flag = self.get_response_information(chatgpt_response)
        self.add_message_to_history(role="assistant", content=content, function_call=function_call)
        self.write_chat_history_to_json()
        if function_flag == 1:  # 如果响应类型是函数调用
            self.publish_string("function_call", self.llm_response_type_publisher)
            self.get_logger().info("STATE: function_execution")
            self.function_call(function_call)  # 调用机器人函数
        else:  # 如果响应类型不是函数调用
            self.publish_string("feedback_for_user", self.llm_response_type_publisher)
            self.get_logger().info("STATE: feedback_for_user")
            self.publish_string(content, self.llm_feedback_publisher)

    def llm_callback(self, msg):
        self.get_logger().info("STATE: model_processing")
        self.get_logger().info(f"Input message received: {msg.data}")
        user_prompt = msg.data
        self.add_message_to_history("user", user_prompt)
        self.process_chatgpt_response(config.chat_history)

    def get_response_information(self, chatgpt_response):
        if isinstance(chatgpt_response, dict):  # 如果是旧版 SDK 返回的字典
            choice = chatgpt_response["choices"][0]["message"]
            content = choice.get("content")
            function_call = choice.get("function_call", None)
        else:  # 如果是新版 SDK 返回的类实例
            choice = chatgpt_response.choices[0]
            message = choice.message
            content = getattr(message, "content", None)
            function_call = getattr(message, "function_call", None)
        if function_call:
            function_call = {
                "name": function_call.name,
                "arguments": function_call.arguments
            }
        function_flag = 0 if content is not None else 1
        self.get_logger().info(f"Get message from OpenAI: {choice}, type: {type(choice)}")
        self.get_logger().info(f"Get content from OpenAI: {content}, type: {type(content)}")
        self.get_logger().info(f"Get function call from OpenAI: {function_call}, type: {type(function_call)}")
        return choice, message, content, function_call, function_flag

    def write_chat_history_to_json(self):
        try:
            json_data = json.dumps(config.chat_history)
            with open(self.chat_history_file, "w", encoding="utf-8") as file:
                file.write(json_data)
            self.get_logger().info("Chat history has been written to JSON")
            return True
        except IOError as error:
            self.get_logger().error(f"Error writing chat history to JSON: {error}")
            return False

    def function_call(self, function_call_input):
        if not self.function_call_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("ChatGPT_function_call_service 未就绪，调用失败")
            return
        
        try:
            parsed_arguments = self._parse_function_arguments(function_call_input)
            validated_request = {
                "name": function_call_input["name"],
                "arguments": parsed_arguments
            }
            function_call_input_str = json.dumps(
                validated_request,
                ensure_ascii=False,
                separators=(",", ":")
            )
            self.function_name = validated_request["name"]
            self.function_call_request.request_text = function_call_input_str
            self.get_logger().info(
                f"Request for ChatGPT_function_call_service: {self.function_call_request.request_text}",
                throttle_duration_sec=1
            )
            future = self.function_call_client.call_async(self.function_call_request)
            future.task_data = {  # 新增此行
                "request_text": function_call_input_str,  # 直接存储字符串
                "timestamp": time.time()
            }
            future.add_done_callback(self.function_call_response_callback)
        
        except Exception as e:
            self.get_logger().error(f"函数调用预处理失败: {str(e)}")

    def _parse_function_arguments(self, function_call_input):
        raw_arguments = function_call_input.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                parsed_args = json.loads(raw_arguments)
            except json.JSONDecodeError:
                self.get_logger().error("无法解析的arguments格式")
                return {}
        else:
            parsed_args = raw_arguments

        required_fields = ["robot_name", "duration", "linear_x", "angular_z"]
        for field in required_fields:
            if field not in parsed_args:
                self.get_logger().error(f"Missing required field: {field}")
                return {}

        try:
            robot_name = str(parsed_args["robot_name"])
            duration = float(parsed_args["duration"])
            linear_x = float(parsed_args["linear_x"])
            angular_z = float(parsed_args["angular_z"])

            # 参数范围校验
            if duration <= 0:
                raise ValueError("Duration must be positive.")
            if abs(linear_x) > 1.0 or abs(angular_z) > 1.0:
                raise ValueError("Linear and angular values must be within [-1.0, 1.0].")

            return {
                "robot_name": robot_name,
                "duration": duration,
                "linear_x": linear_x,
                "angular_z": angular_z
            }
        except (KeyError, ValueError) as e:
            self.get_logger().error(f"Invalid arguments: {str(e)}")
            return {}

    def function_call_response_callback(self, future):
        """服务响应回调（线程安全修复版）"""
        # 新增future有效性校验
        if not hasattr(future, 'task_data'):
            self.get_logger().error("Invalid future object")
            self.get_logger().error(f"Future object: {future}")
            return

        # 从future直接获取数据（关键修改点）
        task_data = future.task_data
        try:
            request_text = task_data["request_text"]
        except KeyError:
            self.get_logger().error("Missing request_text in task_data")
            return

        # 保持原有响应处理逻辑
        try:
            response = future.result()
            if not hasattr(response, 'response_text'):
                raise ValueError("无效的服务响应格式")
        except Exception as e:
            self.get_logger().error(f"服务调用失败: {str(e)}")
            return

        # 优化队列处理逻辑
        if "Robot is busy" in response.response_text:
            with self.motion_queue_lock:
                if request_text in self.motion_queue:
                    self.get_logger().debug("重复任务已存在")
                    return
                if len(self.motion_queue) >= 10:
                    self.get_logger().warn("任务队列已满，移除最早的任务")
                    self.motion_queue.popleft()
                self.motion_queue.append(request_text)  # 使用绑定数据
            return

        # 保持原有正常处理流程
        try:
            self.add_message_to_history(
                role="function", 
                name=self.function_name,
                content=response.response_text
            )
            self.process_chatgpt_response(config.chat_history)
        except json.JSONDecodeError:
            self.get_logger().error("响应JSON解析失败")

def main(args=None):
    rclpy.init(args=args)
    chatgpt = ChatGPTNode()
    rclpy.spin(chatgpt)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
