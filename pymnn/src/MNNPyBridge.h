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

extern "C" PYMNN_PUBLIC void loadMNN();