cmake_minimum_required(VERSION 3.8)
project(jar_reloj)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)

add_executable(reloj_lento src/reloj_lento.cpp)
ament_target_dependencies(reloj_lento rclcpp rosgraph_msgs)
install(TARGETS reloj_lento DESTINATION lib/${PROJECT_NAME})

ament_package()
