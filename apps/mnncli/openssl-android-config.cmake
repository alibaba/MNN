# OpenSSL Android configuration using prebuilt libraries
set(OPENSSL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/prebuilt/Prebuilt/arm64-shared/include")
set(OPENSSL_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/prebuilt/Prebuilt/arm64-shared/lib/libssl.a;${CMAKE_CURRENT_LIST_DIR}/prebuilt/Prebuilt/arm64-shared/lib/libcrypto.a")
set(OPENSSL_FOUND TRUE)
set(OPENSSL_VERSION "1.0.2s")

message(STATUS "Using prebuilt OpenSSL Android libraries: ${OPENSSL_LIBRARIES}")
message(STATUS "OpenSSL include directory: ${OPENSSL_INCLUDE_DIR}")
