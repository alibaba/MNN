/*
 * Copyright (C) 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOOPER_H
#define LOOPER_H

#include <pthread.h>
#include <semaphore.h>

struct loopermessage;

class looper {
 public:
  looper();
  looper& operator=(const looper&) = delete;
  looper(looper&) = delete;
  virtual ~looper();

  void post(int what, void* data, bool flush = false);
  void quit();

  virtual void handle(int what, void* data);

 private:
  void addmsg(loopermessage* msg, bool flush);
  static void* trampoline(void* p);
  void loop();
  loopermessage* head;
  pthread_t worker;
  sem_t headwriteprotect;
  sem_t headdataavailable;
  bool running;
};

#endif // LOOPER_H
