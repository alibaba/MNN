//
//  TestUtils.h
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TestUtils_h
#define TestUtils_h

#include <assert.h>
#include <stdio.h>
#include <functional>
#include <string>
#include "MNNForwardType.h"
#include "Session.hpp"
#include "Tensor.hpp"

/**
 * @brief create session with net and backend
 * @param net       given net
 * @param backend   given backend
 * @return created session
 */
MNN::Session* createSession(MNN::Interpreter* net, MNNForwardType backend);

/**
 * @brief dispatch payload on all available backends
 * @param payload   test to perform
 */
void dispatch(std::function<void(MNNForwardType)> payload);
/**
 * @brief dispatch payload on given backend
 * @param payload   test to perform
 * @param backend   given backend
 */
void dispatch(std::function<void(MNNForwardType)> payload, MNNForwardType backend);

#endif /* TestUtils_h */
