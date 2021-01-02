## 链表相关

### 1. 链表基础操作

1. **链表定义和初始化**

```c++
struct ListNode {
  int val;
  ListNode* next;
  ListNode(int x): val(x), next(nullptr) {}
};
ListNode* head = new ListNode(0);
```

2. **链表添加节点**

```c++
ListNode* nxt = cur->next;
cur->next = newNode;
newNode->next = nxt;
```

3. **链表删除节点**

```
cur->next = cur->next->next;
```

4. **添加哑节点的小技巧**:  在头节点之前添加一个dummyHead，可以有效处理一些边界条件



### 2. 常见问题和常见思路

1. **利用多指针、快慢指针直接处理的问题**：这类问题通常比较简单，如两个链表归并、链表找环（两次双指针）、链表找倒数k个节点、链表找中点等。
2. **反转链表**：重点问题，可以使用迭代（多个指针）或者递归。这类反转链表的模版可以用到很多题目上。下面给出迭代实现反转的关键代码：

```c++
ListNode* pre = nullptr, *cur = head, *nxt = nullptr;
while (cur) {
  nxt = cur->next;
  cur->next = pre;
  pre = cur;
  cur = nxt;
}
return pre;
```

3. **链表中的排序**：常见的排序在链表这种数据结构下的实现（链表无法随机访问，因此最大区别在这），如归并排序，交换结点的快速排序，插入排序等

### 3. 链表问题中的递归

1. **反转链表的递归写法**
2. **删除链表中重复元素的递归写法**

