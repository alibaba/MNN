#include <stdint.h>
void MNNGetMatMulPackMode_RVV(int* eP, int* lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
}
