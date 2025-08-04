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

#ifndef _INTRUSIVELIST_H_
#define _INTRUSIVELIST_H_
#include <cassert>
#include <utility>
#include <cstddef>
namespace ffrt {
///////////////////////////////////////////////////////////////////////////////
template <typename Tag>
struct SListNode {
    // if next point to self, isn't in list
    bool IsLinked() const noexcept
    {
        return next != this;
    }
    explicit SListNode(SListNode* p) noexcept : next {p}
    {
    }
    SListNode() noexcept = default;

private:
    void Swap(SListNode& rhs) noexcept
    {
        std::swap(next, rhs.next);
    }

private:
    template <typename, typename>
    friend struct SList;
    SListNode* next {this};
};

template <typename T, typename NodeType>
struct SList {
    bool empty() const noexcept
    {
        return m_head.next == nullptr;
    }

    SList() noexcept = default;

    SList(SList&& rhs) noexcept
    {
        Swap(rhs);
    }
    SList& operator=(SList&& rhs) noexcept
    {
        if (this != &rhs) {
            SList tmp {std::move(rhs)};
            Swap(tmp);
        }
        return *this;
    }

    void PushFront(T& node) noexcept
    {
        auto& nd = static_cast<NodeType&>(node);
        nd.next = std::exchange(m_head.next, &nd);
    }

    T* PopFront() noexcept
    {
        if (empty()) {
            return nullptr;
        } else {
            auto node = m_head.next;
            m_head.next = std::exchange(node->next, node);
            return static_cast<T*>(node);
        }
    }

private:
    void Swap(SList& rhs) noexcept
    {
        m_head.Swap(rhs.m_head);
    }

private:
    NodeType m_head {nullptr};
};

///////////////////////////////////////////////////////////////////////////////
struct ListNode {
    bool IsLinked() const noexcept
    {
        return next != this && prev != this;
    }

    ListNode(ListNode* p, ListNode* n) noexcept : prev {p}, next {n}
    {
    }
    ListNode() noexcept = default;

private:
    template <typename, typename>
    friend struct List;
    ListNode* prev {this};
    ListNode* next {this};
};

template <typename T, typename NodeType>
struct List {
    List() noexcept : m_head {&m_tail, &m_tail}, m_tail {&m_head, &m_head}
    {
    }

    List(List&& rhs) noexcept : List()
    {
        if (!rhs.empty()) {
            NodeType* x = rhs.m_head.next;
            m_head.next = x;
            x->prev = &m_head;

            NodeType* y = rhs.m_tail.prev;
            y->next = &m_tail;
            m_tail.prev = y;

            m_size = rhs.m_size;
            // reset rhs to empty
            rhs.Reset();
        }
    }

    List& operator=(List&& rhs) noexcept
    {
        if (this != &rhs && !rhs.empty()) {
            Reset();
            NodeType* x = rhs.m_head.next;
            m_head.next = x;
            x->prev = &m_head;

            NodeType* y = rhs.m_tail.prev;
            y->next = &m_tail;
            m_tail.prev = y;

            m_size = rhs.m_size;
            // reset rhs to empty
            rhs.Reset();
        }
        return *this;
    }

    bool empty() const noexcept
    {
        return Size() == 0;
    }

    size_t Size() const noexcept
    {
        return m_size;
    }

    void PushFront(T& node) noexcept
    {
        auto& nd = static_cast<NodeType&>(node);

        m_head.next->prev = &nd;
        nd.prev = &m_head;
        nd.next = m_head.next;
        m_head.next = &nd;
        ++m_size;
    }

    void PushBack(T& node) noexcept
    {
        auto& nd = static_cast<NodeType&>(node);

        m_tail.prev->next = &nd;
        nd.prev = m_tail.prev;
        nd.next = &m_tail;
        m_tail.prev = &nd;
        ++m_size;
    }

    T* PopFront() noexcept
    {
        if (empty()) {
            return nullptr;
        } else {
            auto node = static_cast<T*>(m_head.next);
            Unlink(*node);
            return node;
        }
    }

    T* Front() noexcept
    {
        return empty() ? nullptr : static_cast<T*>(m_head.next);
    }

    T* PopBack() noexcept
    {
        if (empty()) {
            return nullptr;
        } else {
            auto node = static_cast<T*>(m_tail.prev);
            Unlink(*node);
            return node;
        }
    }

    void Erase(T& node) noexcept
    {
        Unlink(node);
    }

private:
    void Unlink(NodeType& node) noexcept
    {
        assert(node.IsLinked());
        node.prev->next = node.next;
        node.next->prev = node.prev;
        node.next = node.prev = &node;
        --m_size;
    }

private:
    void Reset() noexcept
    {
        m_tail.prev = &m_head;
        m_tail.next = &m_head;
        m_head.prev = &m_tail;
        m_head.next = &m_tail;
        m_size = 0;
    }

private:
    NodeType m_head;
    NodeType m_tail;
    size_t m_size {0};
};
} // namespace ffrt
#endif // HICORO_INTRUSIVELIST_H
