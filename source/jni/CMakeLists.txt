cmake_minimum_required(VERSION 3.4.1)

file(GLOB_RECURSE CPP_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp )
add_library( mnncore SHARED ${CPP_SRCS} )

find_library( log-lib log )
find_library( jnigraphics-lib jnigraphics )

add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries( mnncore MNN ${log-lib} ${jnigraphics-lib})
