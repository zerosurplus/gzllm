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

# Global Initialization
config = UserConfig()
openai.api_key = config.openai_api_key

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
        response = openai.ChatCompletion.create(
            model=config.openai_model,
            messages=messages_input,
            functions=config.robot_functions_list,
            function_call="auto",
        )
        return response

    def get_response_information(self, chatgpt_response):
        message = chatgpt_response["choices"][0]["message"]
        content = message.get("content")
        function_call = message.get("function_call", None)
        function_flag = 0 if content is not None else 1
        self.get_logger().info(f"Get message from OpenAI: {message}, type: {type(message)}")
        self.get_logger().info(f"Get content from OpenAI: {content}, type: {type(content)}")
        self.get_logger().info(f"Get function call from OpenAI: {function_call}, type: {type(function_call)}")
        return message, content, function_call, function_flag

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
        """处理多级编码参数"""
        raw_arguments = function_call_input.get("arguments", {})
        
        # 处理字符串类型的arguments
        if isinstance(raw_arguments, str):
            try:
                return json.loads(raw_arguments)
            except json.JSONDecodeError:
                self.get_logger().error("无法解析的arguments格式")
                return {}
        
        # 处理字典类型的arguments
        return {
            "robot_name": str(raw_arguments.get("robot_name", "tb1")),
            "duration": float(raw_arguments.get("duration", 5)),
            "linear_x": float(raw_arguments.get("linear_x", 0.0)),
            "angular_z": float(raw_arguments.get("angular_z", 0.0))
        }

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

    def process_chatgpt_response(self, messages_input):
        chatgpt_response = self.generate_chatgpt_response(messages_input)
        message, content, function_call, function_flag = self.get_response_information(chatgpt_response)
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

def main(args=None):
    rclpy.init(args=args)
    chatgpt = ChatGPTNode()
    rclpy.spin(chatgpt)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
