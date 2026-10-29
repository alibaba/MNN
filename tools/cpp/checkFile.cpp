//
//  checkFile.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <assert.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

// Formatted extraction can reject finite subnormals, so parse the complete token.
static bool readFiniteFloat(std::istream& input, float& value) {
    std::string token;
    if (!(input >> token)) {
        return false;
    }
    char* end = nullptr;
    value = std::strtof(token.c_str(), &end);
    return end == token.c_str() + token.size() && std::isfinite(value);
}

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
    while (true) {
        input1 >> std::ws;
        input2 >> std::ws;
        if (input1.eof() && input2.eof() && !input1.bad() && !input2.bad()) {
            break;
        }
        if (!readFiniteFloat(input1, v1) || !readFiniteFloat(input2, v2)) {
            printf("Error for %d: invalid or missing value\n", pos);
            break;
        }
        if (::fabsf(v1 - v2) > tolerance) {
            printf("Error for %d, v1=%.6f, v2=%.6f\n", pos, v1, v2);
        }
        pos++;
    }
    return 0;
}
