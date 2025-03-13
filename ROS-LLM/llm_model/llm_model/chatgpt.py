# ROS related
import rclpy
from rclpy.node import Node
from llm_interfaces.srv import ChatGPT
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# LLM related
import json
import os
import time
import openai
from llm_config.user_config import UserConfig
from openai import OpenAI

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
        self.client = OpenAI(base_url=config.openai_api_base,api_key=config.openai_api_key)
    def state_listener_callback(self, msg):
        self.get_logger().debug(f"model node get current State:{msg}")

    def publish_string(self, string_to_send, publisher_to_use):
        msg = String()
        msg.data = string_to_send
        publisher_to_use.publish(msg)
        self.get_logger().info(f"Topic: {publisher_to_use.topic_name}\nMessage published: {msg.data}")

    def add_message_to_history(self, role, content="null", function_call=None, name=None):
        # 修复点：严格校验函数调用结构
        message = {"role": role, "content": content}
        
        # 添加函数调用字段（当且仅当有效时）
        if function_call and isinstance(function_call, dict):
            if "name" in function_call and "arguments" in function_call:
                # 确保 arguments 是字符串格式
                message["function_call"] = {
                    "name": str(function_call["name"]),
                    "arguments": json.dumps(function_call["arguments"])
                }
            else:
                self.get_logger().warning("无效的函数调用结构，已过滤")
        
        # 添加name字段（如需要）
        if name:
            message["name"] = name
        
        # 维护历史长度
        config.chat_history.append(message)
        if len(config.chat_history) > config.chat_history_max_length:
            removed = config.chat_history.pop(0)
            self.get_logger().debug(f"移除历史消息: {removed}")
        
        return config.chat_history

    def generate_chatgpt_response(self, messages_input):
        # 修复点：完全适配新版SDK，正确处理FunctionCall对象
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=messages_input,
                functions=config.robot_functions_list,
                function_call="auto",
            )
            
            # 新版SDK响应解析
            message = response.choices[0].message
            function_call = None
            
            # 提取函数调用信息
            if hasattr(message, "function_call") and message.function_call:
                fc = message.function_call
                function_call = {
                    "name": fc.name,
                    "arguments": self._parse_arguments(fc.arguments)
                }
            
            return response, function_call
        
        except Exception as e:
            self.get_logger().error(f"生成响应时发生错误: {str(e)}")
            return None, None

    def _parse_arguments(self, arguments_str):
        """安全解析arguments字符串为字典"""
        try:
            return json.loads(arguments_str) if isinstance(arguments_str, str) else {}
        except json.JSONDecodeError:
            self.get_logger().error("无法解析的arguments格式")
            return {}

    def process_chatgpt_response(self, messages_input):
        # 确保 system_prompt 存在
        if not config.chat_history or config.chat_history[0]["role"] != "system":
            config.chat_history.insert(0, {"role": "system", "content": config.system_prompt})
        # 修复点：避免添加空函数调用
        try:
            chatgpt_response, function_call = self.generate_chatgpt_response(messages_input)
            if chatgpt_response is None:
                self.get_logger().error("无法获取有效响应")
                return
            
            # 添加消息到历史（区分函数调用和普通响应）
            if function_call:
                self.add_message_to_history(
                    role="assistant", 
                    function_call=function_call
                )
                self.publish_string("function_call", self.llm_response_type_publisher)
                self.function_call(function_call)
            else:
                content = chatgpt_response.choices[0].message.content
                self.add_message_to_history(role="assistant", content=content)
                self.publish_string("feedback_for_user", self.llm_response_type_publisher)
                self.publish_string(content, self.llm_feedback_publisher)
            
            self.write_chat_history_to_json()
        
        except Exception as e:
            self.get_logger().error(f"响应处理失败: {str(e)}")

    def llm_callback(self, msg):
        self.get_logger().info("STATE: model_processing")
        self.get_logger().info(f"Input message received: {msg.data}")
        user_prompt = msg.data
        self.add_message_to_history("user", user_prompt)
        self.process_chatgpt_response(config.chat_history)
        
    def get_response_information(self, chatgpt_response):
        # 修复点：严格区分内容响应和函数调用
        try:
            message = chatgpt_response.choices[0].message
            content = message.content
            
            # 提取函数调用信息
            function_call = None
            if hasattr(message, "function_call") and message.function_call:
                fc = message.function_call
                function_call = {
                    "name": fc.name,
                    "arguments": self._parse_arguments(fc.arguments)
                }
            
            # 判定响应类型
            function_flag = 1 if function_call else 0
            
            # 记录调试信息
            self.get_logger().info(f"响应内容: {content}")
            self.get_logger().info(f"函数调用: {json.dumps(function_call, ensure_ascii=False)}")
            
            return (
                chatgpt_response.choices[0],
                message,
                content,
                function_call,
                function_flag
            )
        
        except AttributeError as e:
            self.get_logger().error(f"响应解析失败: {str(e)}")
            return None, None, None, None, 0

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
        # 添加服务可用性检查
        if not self.function_call_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("ChatGPT_function_call_service 未就绪，调用失败")
            return
        
        try:
            # 参数深度解析
            parsed_arguments = self._parse_function_arguments(function_call_input)
            
            # 构造安全参数结构
            validated_request = {
                "name": function_call_input["name"],
                "arguments": parsed_arguments  # 确保是字典类型
            }
            
            # 安全序列化
            function_call_input_str = json.dumps(
                validated_request,
                ensure_ascii=False,
                separators=(",", ":")
            )
            
            self.function_name = validated_request["name"]
            self.function_call_request.request_text = function_call_input_str
            
            self.get_logger().info(
                f"Request for ChatGPT_function_call_service: {self.function_call_request.request_text}",
                throttle_duration_sec=1  # 添加节流防止日志轰炸
            )
            
            future = self.function_call_client.call_async(self.function_call_request)
            future.add_done_callback(self.function_call_response_callback)
            
        except Exception as e:
            self.get_logger().error(f"函数调用预处理失败: {str(e)}")

    def _parse_function_arguments(self, function_call_input):
        """解析函数参数"""
        try:
            arguments = function_call_input.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            elif not isinstance(arguments, dict):
                self.get_logger().warning("arguments 不是字典类型，尝试解析为字典")
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    self.get_logger().error("无法解析的arguments格式")
                    arguments = {}
            return arguments
        except json.JSONDecodeError:
            self.get_logger().error("无法解析的arguments格式")
            return {}

    def function_call_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Response from ChatGPT_function_call_service: {response}")
            response_text = response.response_text  # 假设响应内容在 `response_text` 字段中
            self.add_message_to_history(role="function", name=self.function_name, content=response_text)
            
            # 重新生成ChatGPT响应，处理可能的新函数调用
            self.process_chatgpt_response(config.chat_history)
        except Exception as e:
            self.get_logger().info(f"ChatGPT function call service failed {e}")

def main(args=None):
    rclpy.init(args=args)
    chatgpt = ChatGPTNode()
    rclpy.spin(chatgpt)
    rclpy.shutdown()

if __name__ == "__main__":
    main()