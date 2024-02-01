cmake_minimum_required(VERSION 3.1.5)

# Make sure submodules are downloaded
find_package(git QUIET)
if (GIT_FOUND AND EXISTS "${PARENT_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${PARENT_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )

        if (NOT GITSUBMOD_RESULT EQUAL "0")
            message (FATAL_ERROR
                "git submodule --init failed with ${GIT_SUBMOD_RESULT}"
            )
        endif()
    endif()
endif()

#####
# Directories
#####
set (GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set (SCRIPTS_DIR   "${CMAKE_CURRENT_BINARY_DIR}/scripts")
set (OBJ_DIR       "${CMAKE_CURRENT_BINARY_DIR}/obj")
set (LOG_DIR       "${CMAKE_CURRENT_BINARY_DIR}/log")
set (REPORTS_DIR   "${CMAKE_CURRENT_BINARY_DIR}/reports")
set (BIN_DIR       "${CMAKE_CURRENT_BINARY_DIR}/bin")
set (TEMP_DIR      "${CMAKE_CURRENT_BINARY_DIR}/tmp")
file (MAKE_DIRECTORY
    ${GENERATED_DIR}
    ${SCRIPTS_DIR}
    ${OBJ_DIR}
    ${LOG_DIR}
    ${REPORTS_DIR}
    ${BIN_DIR}
    ${TEMP_DIR}
)

set (SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

#####
# Load hlslib
#####
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HLSLIB_DIR}/cmake")
find_package(Vitis REQUIRED)
include_directories("${HLSLIB_DIR}/include" ${Vitis_INCLUDE_DIRS})

#####
# Load rtllib
#####
include ("${RTLLIB_DIR}/cmake/rtl_target.cmake")
include ("${RTLLIB_DIR}/cmake/cpp_target.cmake")
set (RTLLIB_MODULES "${RTLLIB_DIR}/rtl")
set (TEMPLATES_DIR  "${RTLLIB_DIR}/templates")

#####
# Device section
#####

# RTL kernels
foreach (RTL_KERNEL ${RTL_KERNELS})
    rtllib_rtl_generate_from_cfg(${RTL_KERNEL} ${GENERATED_DIR} ${SRC_DIR} ${SCRIPTS_DIR} ${TEMPLATES_DIR})

    set (XO "${OBJ_DIR}/${RTL_KERNEL}.xo")
    rtllib_rtl_target(${RTL_KERNEL} ${SRC_DIR} ${SCRIPTS_DIR} ${GENERATED_DIR} ${RTLLIB_MODULES} ${XO})
    set (XOS ${XOS} ${XO})
endforeach()

# General Vitis directories and flags
set (VPP_TEMP_DIR "${TEMP_DIR}/vitis")
set (CONFIGS_DIR "${SRC_DIR}/configs")
if (NOT EXISTS VPP_PLATFORM)
    set (VPP_PLATFORM xilinx_u250_xdma_201830_2)
endif()
set (VPP_FLAGS
    --log_dir ${LOG_DIR}
    -t ${VPP_TARGET}
    -f ${VPP_PLATFORM}
    -s
    --report_dir ${REPORTS_DIR}
    --temp_dir ${VPP_TEMP_DIR}
)

# HLS kernels
foreach (CPP_KERNEL ${CPP_KERNELS})
    set (XO "${OBJ_DIR}/${CPP_KERNEL}.xo")
    rtllib_cpp_target(${CPP_KERNEL} ${SRC_DIR} ${XO} ${VPP_TARGET} ${VPP_PLATFORM})
    set (XOS ${XOS} ${XO})
endforeach()

# Link the .xo files
set (VPP_LINKED "${OBJ_DIR}/${PROJECT_NAME}.link.xclbin")
if (EXISTS "${CONFIGS_DIR}/link.ini")
    set (CONFIG_LINK --config "${CONFIGS_DIR}/link.ini")
endif()
set (VPP_LINK_FLAGS
    ${CONFIG_LINK}
    -o ${VPP_LINKED}
    -l ${XOS}
)
add_custom_command(
    OUTPUT ${VPP_LINKED}
    COMMAND ${Vitis_COMPILER} ${VPP_FLAGS} ${VPP_LINK_FLAGS}
    DEPENDS ${XOS}
)

# Package the .xclbin file
set (DEVICE_BINARY "${BIN_DIR}/${PROJECT_NAME}.xclbin")
if (EXISTS "${CONFIGS_DIR}/package.ini")
    set (CONFIG_PKG --config "${CONFIGS_DIR}/package.ini")
endif()
set (VPP_PKG_FLAGS
    ${CONFIG_PKG}
    -o ${DEVICE_BINARY}
    -p ${VPP_LINKED}
)
add_custom_command(
    OUTPUT ${DEVICE_BINARY}
    COMMAND ${Vitis_COMPILER} ${VPP_FLAGS} ${VPP_PKG_FLAGS}
    DEPENDS ${VPP_LINKED}
)

#####
# Xilinx Runtime emulation configuration
#####
set (EMCONFIG "${BIN_DIR}/emconfig.json")
add_custom_command(
    OUTPUT ${EMCONFIG}
    COMMAND emconfigutil --platform ${VPP_PLATFORM} --od bin
)

#####
# Host section
#####
set (HOST_SRC "${SRC_DIR}/host.cpp")
set (HOST_BINARY "${BIN_DIR}/${PROJECT_NAME}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DHLSLIB_LEGACY_SDX=0")
add_executable(${PROJECT_NAME} ${HOST_SRC})
target_link_libraries(${PROJECT_NAME} ${Vitis_LIBRARIES})
set_target_properties(${PROJECT_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)

#####
# Targets
#####
add_custom_target(device_${PROJECT_NAME}
    ALL
    DEPENDS ${DEVICE_BINARY} ${EMCONFIG}
)

add_custom_target(build_${PROJECT_NAME}
    DEPENDS ${HOST_BINARY} ${DEVICE_BINARY} ${EMCONFIG}
)

add_custom_target(run_${PROJECT_NAME}
    COMMAND XCL_EMULATION_MODE=${VPP_TARGET} ${HOST_BINARY} ${DEVICE_BINARY}
    DEPENDS ${PROJECT_NAME} ${DEVICE_BINARY} ${EMCONFIG}
)
