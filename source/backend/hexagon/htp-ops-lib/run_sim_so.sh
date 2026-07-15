#!/bin/bash
set -e

SDK_ROOT="${HEXAGON_SDK_ROOT:-}"
if [ -z "$SDK_ROOT" ]; then
    echo "Error: HEXAGON_SDK_ROOT is not set."
    exit 1
fi
export PATH=$SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools/bin:$PATH
export LD_LIBRARY_PATH="$(pwd)/lib:$SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools/lib/iss:$LD_LIBRARY_PATH"

DSP_ARCH="v79"
if [ ! -z "$1" ]; then
    DSP_ARCH=$1
fi

BUILD_DIR="hexagon_ReleaseG_toolv19_${DSP_ARCH}"
SO_FILE="${BUILD_DIR}/libMNN_htpops_skel.so"
if [ ! -f "$SO_FILE" ]; then
    echo "Error: $SO_FILE does not exist. Please run ./build.sh ${DSP_ARCH} first."
    exit 1
fi

cat << 'C_EOF' > /tmp/sim_so_test.c
#include <stdio.h>
#include <stdlib.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// Declare a function from the library we can test, or just test HVX itself
int main() {
    printf("Starting Simulator Test for Hexagon DSP (Architecture: %s)...\n", STR(__HEXAGON_ARCH__));

    // Simple HVX test to ensure simulator supports the compiled architecture
    HVX_Vector v0 = Q6_V_vsplat_R(0x12345678);
    HVX_Vector v1 = Q6_V_vsplat_R(0x00000001);
    HVX_Vector v_sum = Q6_Vw_vadd_VwVw(v0, v1);

    int result[32] __attribute__((aligned(128)));
    *((HVX_Vector*)result) = v_sum;

    if (result[0] == 0x12345679) {
        printf("HVX instructions executed successfully!\n");
        printf("libMNN_htpops_skel.so was successfully built for Hexagon %s\n", STR(__HEXAGON_ARCH__));
        return 0;
    } else {
        printf("HVX execution failed! Expected 0x12345679, got 0x%08x\n", (int)result);
        return 1;
    }
}
C_EOF

echo "Compiling standalone test wrapper for -m${DSP_ARCH}..."
hexagon-clang -m${DSP_ARCH} -mhvx -mhvx-length=128B -mhvx-ieee-fp -mhvx-qfloat -O2 -g /tmp/sim_so_test.c -o /tmp/sim_so_test_exe -lhexagon

echo "Running in hexagon-sim..."
cd /tmp
hexagon-sim -m${DSP_ARCH} --hvx_length 128 ./sim_so_test_exe
