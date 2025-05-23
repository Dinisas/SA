execute_process(COMMAND "/home/dinisas/SA/catkin_ws/build/pioneer_fast_slam/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/dinisas/SA/catkin_ws/build/pioneer_fast_slam/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
