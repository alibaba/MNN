/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FFRT_LINKED_LIST_H
#define FFRT_LINKED_LIST_H

#include <cstddef>
#include <cstdint>

namespace ffrt {
class LinkedList {
public:
    LinkedList() : prev(this), next(this)
    {
    }

    LinkedList(LinkedList* prev, LinkedList* next) : prev(prev), next(next)
    {
    }

    template <typename T>
    static ptrdiff_t OffsetOf(LinkedList T::*member) noexcept
    {
        return reinterpret_cast<ptrdiff_t>(&(reinterpret_cast<T*>(0)->*member));
    }

    template <typename T>
    static T* ContainerOf(LinkedList* node, LinkedList T::*member) noexcept
    {
        return reinterpret_cast<T*>(reinterpret_cast<intptr_t>(node) - OffsetOf<T>(member));
    }

    template <typename T>
    T* ContainerOf(LinkedList T::*member) noexcept
    {
        return ContainerOf(this, member);
    }

    static void InsertAfter(LinkedList* cur, LinkedList* node) noexcept
    {
        node->next = cur->next;
        node->prev = cur;
        cur->next->prev = node;
        cur->next = node;
    }

    static void InsertBefore(LinkedList* cur, LinkedList* node) noexcept
    {
        node->next = cur;
        node->prev = cur->prev;
        cur->prev->next = node;
        cur->prev = node;
    }

    static void Delete(LinkedList& node) noexcept
    {
        node.prev->next = node.next;
        node.next->prev = node.prev;
        node.next = &node;
        node.prev = &node;
    }

    static void Delete(LinkedList* node) noexcept
    {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        node->next = node;
        node->prev = node;
    }

    static void RemoveCur(LinkedList& node) noexcept
    {
        if (node.Null()) {
            return;
        }
        Delete(node);
    }

    static void RemoveCur(LinkedList* node) noexcept
    {
        if (node->Null()) {
            return;
        }
        Delete(node);
    }

    static LinkedList* Next(LinkedList* cur) noexcept
    {
        if (cur->Empty()) {
            return nullptr;
        }

        LinkedList* next = cur->next;
        return next;
    }

    template <typename T>
    static T* Next(LinkedList* cur, LinkedList T::*member) noexcept
    {
        if (cur->Empty()) {
            return nullptr;
        }

        LinkedList* next = cur->next;
        return ContainerOf<T>(next, member);
    }

    static LinkedList* RemoveNext(LinkedList* cur) noexcept
    {
        if (cur->Empty()) {
            return nullptr;
        }

        LinkedList* next = cur->next;
        Delete(next);
        return next;
    }

    template <typename T>
    static T* RemoveNext(LinkedList* cur, LinkedList T::*member) noexcept
    {
        if (cur->Empty()) {
            return nullptr;
        }

        LinkedList* next = cur->next;
        Delete(next);
        return ContainerOf<T>(next, member);
    }

    static LinkedList* RemovePrev(LinkedList* cur) noexcept
    {
        if (cur->Empty()) {
            return nullptr;
        }

        LinkedList* prev = cur->prev;
        Delete(prev);
        return prev;
    }

    template <typename T>
    static T* RemovePrev(LinkedList* cur, LinkedList T::*member) noexcept
    {
        if (cur->Empty()) {
            return nullptr;
        }

        LinkedList* prev = cur->prev;
        Delete(prev);
        return ContainerOf<T>(prev, member);
    }

    void InsertAfter(LinkedList& node) noexcept
    {
        InsertAfter(this, &node);
    }

    void InsertAfter(LinkedList* node) noexcept
    {
        InsertAfter(this, node);
    }

    void InsertBefore(LinkedList& node) noexcept
    {
        InsertBefore(this, &node);
    }

    void InsertBefore(LinkedList* node) noexcept
    {
        InsertBefore(this, node);
    }

    LinkedList* Next() noexcept
    {
        return Next(this);
    }

    template <typename T>
    T* Next(LinkedList T::*member) noexcept
    {
        return Next(this, member);
    }

    LinkedList* RemoveNext() noexcept
    {
        return RemoveNext(this);
    }

    template <typename T>
    T* RemoveNext(LinkedList T::*member) noexcept
    {
        return RemoveNext(this, member);
    }

    LinkedList* RemovePrev() noexcept
    {
        return RemovePrev(this);
    }

    template <typename T>
    T* RemovePrev(LinkedList T::*member) noexcept
    {
        return RemovePrev(this, member);
    }

    void PushFront(LinkedList& node) noexcept
    {
        InsertAfter(&node);
    }

    void PushFront(LinkedList* node) noexcept
    {
        InsertAfter(node);
    }

    void PushBack(LinkedList& node) noexcept
    {
        InsertBefore(&node);
    }

    void PushBack(LinkedList* node) noexcept
    {
        InsertBefore(node);
    }

    LinkedList* Front() noexcept
    {
        return Next();
    }

    template <typename T>
    T* Front(LinkedList T::*member) noexcept
    {
        return Next(member);
    }

    LinkedList* PopFront() noexcept
    {
        return RemoveNext();
    }

    template <typename T>
    T* PopFront(LinkedList T::*member) noexcept
    {
        return RemoveNext(member);
    }

    LinkedList* PopBack() noexcept
    {
        return RemovePrev();
    }

    template <typename T>
    T* PopBack(LinkedList T::*member) noexcept
    {
        return RemovePrev(member);
    }

    bool Empty() const noexcept
    {
        return next == this;
    }

    bool Null() const noexcept
    {
        return prev == nullptr && next == nullptr;
    }

    bool InList() const noexcept
    {
        return (next != nullptr && next != this);
    }

    void PushBack(LinkedList& first, LinkedList& last) noexcept
    {
        // push back multiple linked nodes to list
        last.next = this;
        first.prev = this->prev;
        this->prev->next = &first;
        this->prev = &last;
    }

private:
    LinkedList* prev;
    LinkedList* next;
};
} // namespace ffrt
#endif
