function (rtllib_cpp_target RTLLIB_KERNEL RTLLIB_SRC_DIR RTLLIB_XO RTLLIB_TARGET RTLLIB_PLATFORM)
    set (RTLLIB_HLS_DIR     "${RTLLIB_SRC_DIR}/hls")
    set (RTLLIB_CONFIG      "${RTLLIB_SRC_DIR}/configs/build.ini")
    set (RTLLIB_TMP_DIR     "${CMAKE_CURRENT_BINARY_DIR}/tmp")
    set (RTLLIB_LOG_DIR     "${CMAKE_CURRENT_BINARY_DIR}/log")
    set (RTLLIB_REPORTS_DIR "${CMAKE_CURRENT_BINARY_DIR}/reports")
    set (RTLLIB_VPP_TMP_DIR "${RTLLIB_TMP_DIR}/vpp")
    set (RTLLIB_KERNEL_SRC  "${RTLLIB_HLS_DIR}/${RTLLIB_KERNEL}.cpp")
    if (EXISTS ${RTLLIB_CONFIG})
        set   (RTLLIB_CONFIG_BUILD --config ${RTLLIB_CONFIG})
    else()
        unset (RTLLIB_CONFIG_BUILD)
    endif()
    set (RTLLIB_VPP_BUILD_FLAGS
        --log_dir ${RTLLIB_LOG_DIR}
        -t ${RTLLIB_TARGET}
        -f ${RTLLIB_PLATFORM}
        -s
        --report_dir ${RTLLIB_REPORTS_DIR}
        --temp_dir ${RTLLIB_VPP_TMP_DIR}
        ${RTLLIB_CONFIG_BUILD}
        -o ${RTLLIB_XO}
        -k ${RTLLIB_KERNEL}
        -c
        ${RTLLIB_KERNEL_SRC}
    )
    add_custom_command(
        OUTPUT ${RTLLIB_XO}
        COMMAND ${Vitis_COMPILER} ${RTLLIB_VPP_BUILD_FLAGS}
        DEPENDS ${RTLLIB_KERNEL_SRC}
    )
endfunction()
