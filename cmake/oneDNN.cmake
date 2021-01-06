include(ExternalProject)

set(DOWNLOAD_URL https://github.com/oneapi-src/oneDNN/archive/v1.7.zip)
set(ROOT ${CMAKE_CURRENT_LIST_DIR}/../3rd_party/)
set(ONEDNN_DIR ${ROOT}/oneDNN/)
set(MNN_BUILD_DIR ${CMAKE_CURRENT_LIST_DIR}/../build/)

set(CONFIGURE_CMD cd ${ONEDNN_DIR} && cmake -DCMAKE_INSTALL_PREFIX=${MNN_BUILD_DIR} -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=SEQ)
set(BUILD_CMD cd ${ONEDNN_DIR} && make -j8)
set(INSTALL_CMD cd ${ONEDNN_DIR} && make install)

ExternalProject_Add(oneDNN
    PREFIX              oneDNN
    URL                 ${DOWNLOAD_URL}
    DOWNLOAD_DIR        ${ROOT}
    SOURCE_DIR          ${ONEDNN_DIR}
    CONFIGURE_COMMAND   ${CONFIGURE_CMD}
    BUILD_COMMAND       ${BUILD_CMD}
    INSTALL_COMMAND     ${INSTALL_CMD}
)

