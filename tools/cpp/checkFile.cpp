//
//  checkFile.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: ./checkFile.out XXX.txt YYY.txt tolerance\n");
        return 0;
    }

    // read args
    const char* file1 = argv[1];
    const char* file2 = argv[2];
    float tolerance   = 0.001;
    if (argc > 3) {
        std::istringstream ss(argv[3]);
        ss >> tolerance;
    }

    // open file
    std::ifstream input1(file1);
    assert(!input1.fail());
    std::ifstream input2(file2);
    assert(!input2.fail());

    // compare
    float v1, v2;
    int pos = 0;
    while (input1 >> v1) {
        auto& valid = input2 >> v2;
        if (::fabsf(v1 - v2) > tolerance) {
            printf("Error for %d, v1=%.6f, v2=%.6f\n", pos, v1, v2);
        }
        pos++;
        if (!valid) {
            break;
        }
    }
    return 0;
}
