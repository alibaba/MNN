//
//  checkDir.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

#define NONE "\e[0m"
#define RED "\e[0;31m"
#define GREEN "\e[0;32m"
#define L_GREEN "\e[1;32m"
#define BLUE "\e[0;34m"
#define L_BLUE "\e[1;34m"
#define BOLD "\e[1m"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: ./checkDir.out outputDir1 outputDir2 thredhold\n");
        return 0;
    }

    // read args
    printf("Compare:\n");
    printf("%s\n", argv[1]);
    printf("%s\n", argv[2]);
    float tolerance = 0.001;
    if (argc > 3) {
        tolerance = stof(string(argv[3]));
    }
    printf("tolerance=%f\n", tolerance);

    // open dir
    DIR* root = opendir(argv[1]);
    if (NULL == root) {
        printf("Error to open %s\n", argv[1]);
        return 0;
    }
    std::vector<std::string> compareFiles;
    struct dirent* ent;
    while ((ent = readdir(root)) != NULL) {
        compareFiles.push_back(ent->d_name);
    }
    closedir(root);

    // compare files
    for (auto s : compareFiles) {
        if (s.size() <= 2) {
            continue;
        }
        std::string empty   = "";
        std::string oldFile = empty + argv[1] + "/" + s;
        std::string newFile = empty + argv[2] + "/" + s;
        std::ifstream oldOs(oldFile.c_str());
        if (oldOs.fail()) {
            // printf("Can's open %s\n", oldFile.c_str());
            continue;
        }
        std::ifstream newOs(newFile.c_str());
        if (newOs.fail()) {
            // printf("Can's open %s\n", newFile.c_str());
            continue;
        }

        int pos      = 0;
        bool correct = true;
        float v1, v2;
        while (oldOs >> v1) {
            auto& valid   = newOs >> v2;
            auto absError = fabsf(v1 - v2);
            if (absError <= tolerance && !isnan(absError)) {
                pos++;
                continue;
            }
            printf(RED "Error for %s, %d, v1=%.6f, v2=%.6f\n" NONE, s.c_str(), pos, v1, v2);
            correct = false;
            break;
        }
        if (correct) {
            printf(GREEN "Correct : %s\n" NONE, s.c_str());
        }
    }

    return 0;
}
