# 1. 使用 GLOB 获取所有子目录
file(GLOB SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)

# 2. 遍历每个子目录
foreach(SUBDIR ${SUBDIRS})
    # 检查该子目录是否包含 CMakeLists.txt
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
        add_subdirectory(${SUBDIR})
        message(STATUS "Added subdirectory: ${SUBDIR}")
    endif()
endforeach()