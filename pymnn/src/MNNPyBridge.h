//
//  MNNPyBridge.h
//  MNNPyBridge
//
//  Created by hebin on 2020/12/17.
//  Copyright © 2020 hebin. All rights reserved.
//
#pragma once

#ifdef WIN32
#ifdef BUILDING_PYMNN_DLL
#define PYMNN_PUBLIC __declspec(dllexport)
#else
#define PYMNN_PUBLIC __declspec(dllimport)
#endif // BUILDING_PYMNN_DLL
#else
#define PYMNN_PUBLIC
#endif // WIN32

// memoryToVar's type define
#define TypeFloat 1
#define TypeDouble 2
#define TypeInt 3
#define TypeUint8 4
#define TypeInt8 6
#define TypeInt64 9
extern "C" PYMNN_PUBLIC void loadMNN();
void* memoryToVar(void* ptr, int h, int w, int c, int type);