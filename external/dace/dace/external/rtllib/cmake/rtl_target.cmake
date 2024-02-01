function (rtllib_rtl_generate_from_cfg RTLLIB_KERNEL RTLLIB_GEN_DIR RTLLIB_SRC_DIR RTLLIB_SCRIPTS_DIR RTLLIB_TEMPLATES_DIR)
    set (RTLLIB_CONFIG           "${RTLLIB_SRC_DIR}/hdl/${RTLLIB_KERNEL}/kernel.json")
    set (RTLLIB_CONTROL_TEMPLATE "${RTLLIB_TEMPLATES_DIR}/control.py")
    set (RTLLIB_TOP_TEMPLATE     "${RTLLIB_TEMPLATES_DIR}/top.py")
    set (RTLLIB_PACKAGE_TEMPLATE "${RTLLIB_TEMPLATES_DIR}/package.py")
    set (RTLLIB_SYNTH_TEMPLATE   "${RTLLIB_TEMPLATES_DIR}/synth.py")

    # Generate the controller
    set (RTLLIB_CONTROL "${RTLLIB_GEN_DIR}/${RTLLIB_KERNEL}_Control.v")
    add_custom_command(
        OUTPUT  ${RTLLIB_CONTROL}
        COMMAND ${RTLLIB_CONTROL_TEMPLATE} ${RTLLIB_CONFIG} -o ${RTLLIB_CONTROL} -f
        DEPENDS ${RTLLIB_CONTROL_TEMPLATE} ${RTLLIB_CONFIG}
    )

    # Generate the top file
    set (RTLLIB_TOP "${RTLLIB_GEN_DIR}/${RTLLIB_KERNEL}_top.v")
    add_custom_command(
        OUTPUT  ${RTLLIB_TOP}
        COMMAND ${RTLLIB_TOP_TEMPLATE} ${RTLLIB_CONFIG} -o ${RTLLIB_TOP} -f
        DEPENDS ${RTLLIB_TOP_TEMPLATE} ${RTLLIB_CONFIG}
    )

    # Generate the package script
    set (RTLLIB_PKG "${RTLLIB_SCRIPTS_DIR}/${RTLLIB_KERNEL}_package.tcl")
    add_custom_command(
        OUTPUT  ${RTLLIB_PKG}
        COMMAND ${RTLLIB_PACKAGE_TEMPLATE} ${RTLLIB_CONFIG} -o ${RTLLIB_PKG} -f
        DEPENDS ${RTLLIB_PACKAGE_TEMPLATE} ${RTLLIB_CONFIG}
    )

    # Make elaborate and synth script
    set (RTLLIB_SYNTH "${RTLLIB_SCRIPTS_DIR}/${RTLLIB_KERNEL}_synth.tcl")
    add_custom_command(
        OUTPUT  ${RTLLIB_SYNTH}
        COMMAND ${RTLLIB_SYNTH_TEMPLATE} ${RTLLIB_CONFIG} -o ${RTLLIB_SYNTH} -f
        DEPENDS ${RTLLIB_SYNTH_TEMPLATE} ${RTLLIB_CONFIG}
    )
endfunction()

function (rtllib_rtl_target RTLLIB_KERNEL RTLLIB_SRC_DIR RTLLIB_TCL_DIR RTLLIB_GEN_DIR RTLLIB_LOG_DIR RTLLIB_TMP_DIR RTLLIB_MODULES RTLLIB_XO RTLLIB_PART RTLLIB_DEPS RTLLIB_USER_IP_REPO)
    # Files and directories for the kernel
    set (RTLLIB_HDL_DIR        "${RTLLIB_SRC_DIR}/hdl")
    #set (RTLLIB_SRC_DIR     "${RTLLIB_HDL_DIR}/${RTLLIB_KERNEL}")
    file(GLOB RTLLIB_SRCS      "${RTLLIB_SRC_DIR}/*.*v")
    set (RTLLIB_CTRL           "${RTLLIB_SRC_DIR}/${RTLLIB_KERNEL}_control.v")
    if (NOT EXISTS "${RTLLIB_CTRL}")
        set (RTLLIB_SRCS ${RTLLIB_SRCS} "${RTLLIB_GEN_DIR}/${RTLLIB_KERNEL}_control.v")
    endif()
    set (RTLLIB_TOP            "${RTLLIB_SRC_DIR}/${RTLLIB_KERNEL}_top.v")
    if (NOT EXISTS "${RTLLIB_TOP}")
        set (RTLLIB_SRCS ${RTLLIB_SRCS} "${RTLLIB_GEN_DIR}/${RTLLIB_KERNEL}_top.v")
    endif()
    set (RTLLIB_PKG            "${RTLLIB_TCL_DIR}/${RTLLIB_KERNEL}_package.tcl")
    set (RTLLIB_SYNTH          "${RTLLIB_TCL_DIR}/${RTLLIB_KERNEL}_synth.tcl")
    #set (RTLLIB_TMP_DIR        "${CMAKE_CURRENT_BINARY_DIR}/tmp")
    #set (RTLLIB_LOG_DIR        "${CMAKE_CURRENT_BINARY_DIR}/log")
    set (RTLLIB_VIVADO_TMP_DIR "${RTLLIB_TMP_DIR}/vivado")

    # Package the kernel
    set (RTLLIB_VIVADO_PKG_FLAGS
        -mode batch
        -log "${RTLLIB_LOG_DIR}/vivado_${RTLLIB_KERNEL}.log"
        -journal "${RTLLIB_LOG_DIR}/vivado_${RTLLIB_KERNEL}.jou"
        -source ${RTLLIB_PKG}
        -tclargs
            ${RTLLIB_XO}
            ${RTLLIB_KERNEL}_top
            ${RTLLIB_VIVADO_TMP_DIR}/${RTLLIB_KERNEL}
            ${RTLLIB_SRC_DIR}
            ${RTLLIB_MODULES}
            ${RTLLIB_GEN_DIR}
            ${RTLLIB_USER_IP_REPO}
            ${RTLLIB_PART}
    )
    add_custom_command(
        OUTPUT  ${RTLLIB_XO}
        COMMAND ${Vitis_VIVADO} ${RTLLIB_VIVADO_PKG_FLAGS}
        DEPENDS ${RTLLIB_PKG} ${RTLLIB_SRCS} ${RTLLIB_DEPS}
    )

    # Make targets for elaborate and synth, for verifying the RTL code
    set (RTLLIB_VIVADO_SYNTH_FLAGS
        -mode batch
        -log "${RTLLIB_LOG_DIR}/vivado_synth_${RTLLIB_KERNEL}.log"
        -journal "${RTLLIB_LOG_DIR}/vivado_synth_${RTLLIB_KERNEL}.jou"
        -source ${RTLLIB_SYNTH}
        -tclargs
            ${RTLLIB_SRC_DIR}
            ${RTLLIB_KERNEL}_top
            "${RTLLIB_VIVADO_TMP_DIR}/${RTLLIB_KERNEL}_synth"
            ${RTLLIB_MODULES}
            ${RTLLIB_GEN_DIR}
            ${RTLLIB_PART}
            ${RTLLIB_USER_IP_REPO}
    )
    add_custom_target(rtllib_elaborate_${PROJECT_NAME}_${RTLLIB_KERNEL}
        COMMAND ${Vitis_VIVADO} ${RTLLIB_VIVADO_SYNTH_FLAGS} -rtl
        DEPENDS ${RTLLIB_SYNTH} ${RTLLIB_SRCS} ${RTLLIB_DEPS}
    )
    add_custom_target(rtllib_synth_${PROJECT_NAME}_${RTLLIB_KERNEL}
        COMMAND ${Vitis_VIVADO} ${RTLLIB_VIVADO_SYNTH_FLAGS}
        DEPENDS ${RTLLIB_SYNTH} ${RTLLIB_SRCS} ${RTLLIB_DEPS}
    )
endfunction()
