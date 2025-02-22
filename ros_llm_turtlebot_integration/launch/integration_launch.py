from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution  # 添加了LaunchConfiguration和PathJoinSubstitution的导入

def generate_launch_description():
    ld = LaunchDescription()

    # 声明一个仿真时间的参数（如果需要）
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time', default_value=use_sim_time, description='Use simulator time'
    )
    ld.add_action(declare_use_sim_time)

    # 包含ROS-LLM项目的启动文件
    llm_launch_file_path = PathJoinSubstitution(
        [FindPackageShare('llm_bringup'), 'launch', 'chatgpt_with_turtle_robot.launch.py']  # 更改路径以匹配实际的大模型项目启动文件
    )
    llm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(llm_launch_file_path)
    )
    ld.add_action(llm_launch)

    # 包含turtlebot3_multi_robot项目的启动文件
    turtlebot_launch_file_path = PathJoinSubstitution(
        [FindPackageShare('turtlebot3_multi_robot'), 'launch', 'gazebo_multi_nav2_world.launch.py']  # 更新了启动文件名称以匹配提供的路径
    )
    turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(turtlebot_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time}.items()  # 使用LaunchConfiguration传递参数
    )
    ld.add_action(turtlebot_launch)

    return ld