cmake_minimum_required(VERSION 3.5)
project(slam_toolbox)

set(CMAKE_BUILD_TYPE Release) #None, Debug, Release, RelWithDebInfo, MinSizeRel
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
cmake_policy(SET CMP0077 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/CMake/")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/karto_sdk/cmake)

find_package(ament_cmake REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
# tf2_sensor_msgs 0.25.20 (Humble) is header-only and exports no imported
# target.  Create a stub INTERFACE target so the upstream target_link_libraries
# calls that reference tf2_sensor_msgs::tf2_sensor_msgs resolve without error.
if(NOT TARGET tf2_sensor_msgs::tf2_sensor_msgs)
  add_library(tf2_sensor_msgs::tf2_sensor_msgs INTERFACE IMPORTED)
  target_include_directories(tf2_sensor_msgs::tf2_sensor_msgs INTERFACE
    ${tf2_sensor_msgs_INCLUDE_DIRS}
  )
endif()
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(bondcpp REQUIRED)
find_package(bond REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(rviz_ogre_vendor REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(interactive_markers REQUIRED)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Test Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Test Concurrent)

find_package(rclcpp_components REQUIRED)

#karto_sdk lib
set(BUILD_SHARED_LIBS ON)
add_subdirectory(lib/karto_sdk)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(dependencies
  rclcpp
  message_filters
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  visualization_msgs
  pluginlib
  tf2_geometry_msgs
  tf2_sensor_msgs
  std_msgs
  std_srvs
  builtin_interfaces
  rclcpp_lifecycle
  bondcpp
  bond
  lifecycle_msgs
  rviz_common
  rviz_default_plugins
  rviz_ogre_vendor
  rviz_rendering
  interactive_markers
  Qt${QT_VERSION_MAJOR}
  rclcpp_components
)

set(libraries
    toolbox_common
    SlamToolboxPlugin
    ceres_solver_plugin
    decentralized_multirobot_slam_toolbox
    async_slam_toolbox
    sync_slam_toolbox
    localization_slam_toolbox
    lifelong_slam_toolbox
    map_and_localization_slam_toolbox
)

find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CHOLMOD REQUIRED)
find_package(CSparse REQUIRED)
find_package(G2O REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_definitions(-DQT_NO_KEYWORDS)
find_package(Boost REQUIRED system serialization filesystem thread)

# ---------------------------------------------------------------------------
# Optional CUDA support
# ---------------------------------------------------------------------------
# We use check_language() rather than find_package(CUDA) so that the entire
# GPU code path compiles out silently when no CUDA Toolkit is installed.
# The variable SLAM_TOOLBOX_CUDA_AVAILABLE is written into gpu_config.h by
# configure_file(); it is the single source of truth used by every header
# and source file in this project.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)  # must follow enable_language(CUDA); provides CUDA::cudart, CUDA::cuda_driver
  message(STATUS "slam_toolbox: CUDA Toolkit found — GPU scan matching ENABLED")
  message(STATUS "slam_toolbox: CUDA compiler: ${CMAKE_CUDA_COMPILER}")

  # -------------------------------------------------------------------------
  # Minimum GPU architecture: sm_60 (Pascal).
  # atomicAdd(double*) is UNDEFINED BEHAVIOR below sm_60 — it silently
  # compiles but produces incorrect results on Kepler/Maxwell hardware
  # (sm_35, sm_50, sm_52, sm_53).  Reject any user-supplied list that
  # includes a sub-60 arch to prevent hard-to-diagnose numeric corruption.
  # -------------------------------------------------------------------------
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default: Turing through Ampere for nvcc 11.5 compatibility.
    # sm_89 requires newer toolchains and breaks builds on CUDA 11.5.
    set(CMAKE_CUDA_ARCHITECTURES 75 86)
  else()
    # User supplied a list — validate every entry.
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
      if(_arch LESS 60)
        message(FATAL_ERROR
          "slam_toolbox: CMAKE_CUDA_ARCHITECTURES contains sm_${_arch}, "
          "which is below the required minimum of sm_60.  "
          "atomicAdd(double*) is undefined behaviour on sm_${_arch} and "
          "will produce silently wrong scan-match responses.  "
          "Remove sm_${_arch} from CMAKE_CUDA_ARCHITECTURES and re-run cmake.")
      endif()
    endforeach()
  endif()

  # Propagate C++17 to device code.
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Treat CUDA warnings the same way as CXX warnings.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

  set(SLAM_TOOLBOX_CUDA_AVAILABLE TRUE)
else()
  message(STATUS "slam_toolbox: CUDA Toolkit NOT found — GPU scan matching DISABLED")
  set(SLAM_TOOLBOX_CUDA_AVAILABLE FALSE)
endif()

# Generate include/slam_toolbox/gpu_config.h from the .in template.
# The file is written into the build tree and added to the include path below.
configure_file(
  cmake/gpu_config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/slam_toolbox/gpu_config.h
  @ONLY
)
# ---------------------------------------------------------------------------

include_directories(include lib/karto_sdk/include
                            ${CMAKE_CURRENT_BINARY_DIR}/include
                            ${EIGEN3_INCLUDE_DIRS}
                            ${CHOLMOD_INCLUDE_DIR}
                            ${Boost_INCLUDE_DIRS}
                            ${CERES_INCLUDES}
)

add_definitions(${EIGEN3_DEFINITIONS})

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/LocalizedLaserScan.msg
  srv/Pause.srv
  srv/Reset.srv
  srv/ClearQueue.srv
  srv/ToggleInteractive.srv
  srv/Clear.srv
  srv/SaveMap.srv
  srv/LoopClosure.srv
  srv/MergeMaps.srv
  srv/AddSubmap.srv
  srv/DeserializePoseGraph.srv
  srv/SerializePoseGraph.srv
  msg/NewNodeEvent.msg
  msg/LoopClosureEvent.msg
  msg/GraphNode.msg
  msg/GraphEdge.msg
  msg/PoseGraph.msg
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs nav_msgs visualization_msgs sensor_msgs
)

#### rviz Plugin
qt_wrap_cpp(MOC_FILES rviz_plugin/slam_toolbox_rviz_plugin.hpp)
add_library(SlamToolboxPlugin SHARED
  rviz_plugin/slam_toolbox_rviz_plugin.cpp
  ${MOC_FILES})
target_link_libraries(SlamToolboxPlugin PUBLIC
  ${bond_TARGETS}
  ${builtin_interfaces_TARGETS}
  ${lifecycle_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${visualization_msgs_TARGETS}
  bondcpp::bondcpp
  interactive_markers::interactive_markers
  message_filters::message_filters
  pluginlib::pluginlib
  Qt${QT_VERSION_MAJOR}::Widgets
  rclcpp::rclcpp
  rclcpp_components::component
  rclcpp_components::component_manager
  rclcpp_lifecycle::rclcpp_lifecycle
  rviz_common::rviz_common
  rviz_default_plugins::rviz_default_plugins
  rviz_rendering::rviz_rendering
  sensor_msgs::sensor_msgs_library
  tf2::tf2
  tf2_geometry_msgs::tf2_geometry_msgs
  tf2_ros::static_transform_broadcaster_node
  tf2_ros::tf2_ros
  tf2_sensor_msgs::tf2_sensor_msgs
)
target_include_directories(SlamToolboxPlugin PUBLIC
  ${OGRE_INCLUDE_DIRS}
)
target_link_libraries(SlamToolboxPlugin PUBLIC ${QT_LIBRARIES} rviz_common::rviz_common)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(SlamToolboxPlugin PUBLIC "${cpp_typesupport_target}")
target_compile_definitions(SlamToolboxPlugin PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
target_compile_definitions(SlamToolboxPlugin PRIVATE "RVIZ_DEFAULT_PLUGINS_BUILDING_LIBRARY")
pluginlib_export_plugin_description_file(rviz_common rviz_plugins.xml)

#### Ceres Plugin
add_library(ceres_solver_plugin solvers/ceres_solver.cpp)
target_link_libraries(ceres_solver_plugin PUBLIC
  ${bond_TARGETS}
  ${builtin_interfaces_TARGETS}
  ${lifecycle_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${visualization_msgs_TARGETS}
  bondcpp::bondcpp
  interactive_markers::interactive_markers
  message_filters::message_filters
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_components::component
  rclcpp_components::component_manager
  rclcpp_lifecycle::rclcpp_lifecycle
  rviz_common::rviz_common
  rviz_default_plugins::rviz_default_plugins
  rviz_rendering::rviz_rendering
  sensor_msgs::sensor_msgs_library
  tf2::tf2
  tf2_geometry_msgs::tf2_geometry_msgs
  tf2_ros::static_transform_broadcaster_node
  tf2_ros::tf2_ros
  tf2_sensor_msgs::tf2_sensor_msgs
)
target_link_libraries(ceres_solver_plugin PUBLIC ${CERES_LIBRARIES}
                                          ${Boost_LIBRARIES}
                                          kartoSlamToolbox
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ceres_solver_plugin PUBLIC "${cpp_typesupport_target}")
pluginlib_export_plugin_description_file(slam_toolbox solver_plugins.xml)

#### GPU scan-matching backend (compiled only when CUDA is available)
# The library contains gpu_scan_correlation_backend.cu and nothing else.
# Non-CUDA builds skip this block entirely; toolbox_common does NOT link
# to gpu_correlation_backend in that case, so zero CUDA symbols are present.
if(SLAM_TOOLBOX_CUDA_AVAILABLE)
  add_library(gpu_correlation_backend
    src/gpu_scan_correlation_backend.cu
  )
  # Explicit property-form CUDA standard (required for nvcc on CMake < 3.18)
  set_property(TARGET gpu_correlation_backend PROPERTY CUDA_STANDARD 17)
  set_target_properties(gpu_correlation_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF      # only one TU; no device linking needed
    POSITION_INDEPENDENT_CODE ON
  )
  # Workaround for nvcc 11.5 + GCC 11 parameter-pack errors from
  # /usr/include/c++/11/bits/std_function.h when compiling in C++17 mode.
  # --expt-relaxed-constexpr  : silences constexpr-in-device-context errors
  #                             that GCC 11 headers trigger in nvcc 11.5
  # -Xcompiler -std=c++17     : explicitly pass the C++17 dialect to the host
  #                             compiler (nvcc 11.5 does not always forward it)
  # --generate-code ...       : explicit virtual arch/code for sm_61 to avoid
  #                             nvcc 11.5 defaulting to an incompatible arch
  target_compile_options(gpu_correlation_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr
      -Xcompiler=-std=c++17
      --generate-code=arch=compute_61,code=sm_61
    >
  )
  target_include_directories(gpu_correlation_backend PUBLIC
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # for generated gpu_config.h
    lib/karto_sdk/include
    ${EIGEN3_INCLUDE_DIRS}
  )
  target_link_libraries(gpu_correlation_backend PUBLIC
    kartoSlamToolbox
    CUDA::cudart          # -lcudart, device runtime
    CUDA::cuda_driver     # -lcuda, driver API (for capability queries)
  )
  # Let kartoSlamToolbox know GPU is present so Mapper.cpp compiles the branch
  target_compile_definitions(kartoSlamToolbox PUBLIC KARTO_GPU_AVAILABLE)
  # Export so downstream packages can link against the GPU backend.
  list(APPEND libraries gpu_correlation_backend)
endif()

#### Tool lib for mapping
add_library(toolbox_common src/slam_toolbox_common.cpp src/map_saver.cpp src/loop_closure_assistant.cpp src/loop_closure_listener.cpp src/laser_utils.cpp src/slam_mapper.cpp)
target_link_libraries(toolbox_common PUBLIC
  ${bond_TARGETS}
  ${builtin_interfaces_TARGETS}
  ${lifecycle_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${visualization_msgs_TARGETS}
  bondcpp::bondcpp
  interactive_markers::interactive_markers
  message_filters::message_filters
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_components::component
  rclcpp_components::component_manager
  rclcpp_lifecycle::rclcpp_lifecycle
  rviz_common::rviz_common
  rviz_default_plugins::rviz_default_plugins
  rviz_rendering::rviz_rendering
  sensor_msgs::sensor_msgs_library
  tf2::tf2
  tf2_geometry_msgs::tf2_geometry_msgs
  tf2_ros::static_transform_broadcaster_node
  tf2_ros::tf2_ros
  tf2_sensor_msgs::tf2_sensor_msgs
)
target_link_libraries(toolbox_common PUBLIC kartoSlamToolbox ${Boost_LIBRARIES})
# Link the GPU backend into toolbox_common when available.
# unit in toolbox_common that includes scan_correlation_backend_factory.hpp
# will then automatically find the GPU symbols at link time.
if(SLAM_TOOLBOX_CUDA_AVAILABLE)
  target_link_libraries(toolbox_common PUBLIC gpu_correlation_backend)
endif()
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(toolbox_common PUBLIC "${cpp_typesupport_target}")

#### Mapping executibles
add_library(decentralized_multirobot_slam_toolbox src/slam_toolbox_decentralized_multirobot.cpp)
target_link_libraries(decentralized_multirobot_slam_toolbox toolbox_common kartoSlamToolbox ${Boost_LIBRARIES})
add_executable(decentralized_multirobot_slam_toolbox_node src/slam_toolbox_decentralized_multirobot_node.cpp)
target_link_libraries(decentralized_multirobot_slam_toolbox_node decentralized_multirobot_slam_toolbox)

add_library(async_slam_toolbox src/slam_toolbox_async.cpp)
target_link_libraries(async_slam_toolbox toolbox_common kartoSlamToolbox ${Boost_LIBRARIES})
add_executable(async_slam_toolbox_node src/slam_toolbox_async_node.cpp)
target_link_libraries(async_slam_toolbox_node async_slam_toolbox)

add_library(sync_slam_toolbox src/slam_toolbox_sync.cpp)
target_link_libraries(sync_slam_toolbox toolbox_common kartoSlamToolbox ${Boost_LIBRARIES})
add_executable(sync_slam_toolbox_node src/slam_toolbox_sync_node.cpp)
target_link_libraries(sync_slam_toolbox_node sync_slam_toolbox)

add_library(localization_slam_toolbox src/slam_toolbox_localization.cpp)
target_link_libraries(localization_slam_toolbox toolbox_common kartoSlamToolbox ${Boost_LIBRARIES})
add_executable(localization_slam_toolbox_node src/slam_toolbox_localization_node.cpp)
target_link_libraries(localization_slam_toolbox_node localization_slam_toolbox)

add_library(lifelong_slam_toolbox src/experimental/slam_toolbox_lifelong.cpp)
target_link_libraries(lifelong_slam_toolbox toolbox_common kartoSlamToolbox ${Boost_LIBRARIES})
add_executable(lifelong_slam_toolbox_node src/experimental/slam_toolbox_lifelong_node.cpp)
target_link_libraries(lifelong_slam_toolbox_node lifelong_slam_toolbox)

add_library(map_and_localization_slam_toolbox src/experimental/slam_toolbox_map_and_localization.cpp)
target_link_libraries(map_and_localization_slam_toolbox localization_slam_toolbox toolbox_common kartoSlamToolbox ${Boost_LIBRARIES})
add_executable(map_and_localization_slam_toolbox_node src/experimental/slam_toolbox_map_and_localization_node.cpp)
target_link_libraries(map_and_localization_slam_toolbox_node map_and_localization_slam_toolbox)

#### Merging maps tool
add_executable(merge_maps_kinematic src/merge_maps_kinematic.cpp)
target_link_libraries(merge_maps_kinematic kartoSlamToolbox toolbox_common)

#### testing
#if(BUILD_TESTING)
#  find_package(ament_cmake_gtest REQUIRED)
#  find_package(ament_lint_auto REQUIRED)
#  ament_lint_auto_find_test_dependencies()
#  include_directories(test)
#  ament_add_gtest(lifelong_metrics_test test/lifelong_metrics_test.cpp)
#  target_link_libraries(lifelong_metrics_test lifelong_slam_toolbox)
#endif()

rclcpp_components_register_nodes(async_slam_toolbox "slam_toolbox::AsynchronousSlamToolbox")
rclcpp_components_register_nodes(sync_slam_toolbox "slam_toolbox::SynchronousSlamToolbox")
rclcpp_components_register_nodes(localization_slam_toolbox "slam_toolbox::LocalizationSlamToolbox")
rclcpp_components_register_nodes(lifelong_slam_toolbox "slam_toolbox::LifelongSlamToolbox")
rclcpp_components_register_nodes(map_and_localization_slam_toolbox "slam_toolbox::MapAndLocalizationSlamToolbox")

#### Install
install(TARGETS decentralized_multirobot_slam_toolbox_node
                async_slam_toolbox_node
                sync_slam_toolbox_node
                localization_slam_toolbox_node
                map_and_localization_slam_toolbox_node
                lifelong_slam_toolbox_node
                merge_maps_kinematic
                ${libraries}
                kartoSlamToolbox
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS SlamToolboxPlugin
  EXPORT SlamToolboxPlugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install the generated gpu_config.h alongside the rest of the headers.
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/slam_toolbox/gpu_config.h
  DESTINATION include/slam_toolbox
)
install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "lifelong_launch.py" EXCLUDE
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES solver_plugins.xml rviz_plugins.xml
  DESTINATION share
)

ament_export_include_directories(include)
ament_export_libraries(${libraries} kartoSlamToolbox)
ament_export_dependencies(${dependencies})
ament_export_dependencies(${dependencies} rosidl_default_runtime)
ament_export_targets(SlamToolboxPlugin HAS_LIBRARY_TARGET)
ament_package()
