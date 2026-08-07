//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <queue>
#include "PriorityQueueBenchmarks.h"
#include "Utils.h"

typedef std::priority_queue<intptr_t> pqueue;

void *
cpp_priority_queue_create(const intptr_t *start, size_t count)
{
  auto pq = new pqueue(start, start + count);
  return pq;
}

void
cpp_priority_queue_destroy(void *ptr)
{
  delete static_cast<pqueue *>(ptr);
}

void
cpp_priority_queue_push(void *ptr, intptr_t value)
{
  auto pq = static_cast<pqueue *>(ptr);
  pq->push(value);
}

void
cpp_priority_queue_push_loop(void *ptr, const intptr_t *start, size_t count)
{
  auto pq = static_cast<pqueue *>(ptr);
  for (auto p = start; p < start + count; ++p) {
    pq->push(*p);
  }
}

intptr_t cpp_priority_queue_pop(void *ptr)
{
  auto pq = static_cast<pqueue *>(ptr);
  auto result = pq->top();
  pq->pop();
  return result;
}

void
cpp_priority_queue_pop_all(void *ptr)
{
  auto pq = static_cast<pqueue *>(ptr);
  while (!pq->empty()) {
    black_hole(pq->top());
    pq->pop();
  }
}
