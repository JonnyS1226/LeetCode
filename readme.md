## Leetcode

#### 位运算	

[解法与总结]()

| No.         | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 136.        | [只出现一次的数字](https://leetcode-cn.com/problems/single-number/) | 直接异或                                                     |
| 137.        | [只出现一次的数字Ⅱ](https://leetcode-cn.com/problems/single-number-ii/) | 1. 二进制表示中的每一位相加并分析。 2. 直接使用位运算（理解有难度，结合自动机） |
| 260.        | [只出现一次的数字 Ⅲ](https://leetcode-cn.com/problems/single-number-iii/) | 分组异或                                                     |
| 645         | [645. 错误的集合](https://leetcode-cn.com/problems/set-mismatch/) | 同260，先补充数组，再分组异或                                |
| 318.        | [最大单词长度乘积](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/) | 把字符串映射成26位二进制位，若两个单词无相同字符，则与操作结果为0 |
| 1318.       | [或运算的最小翻转次数](https://leetcode-cn.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/) | 从二进制每一位来模拟或运算即可                               |
| 389.        | [找不同](https://leetcode-cn.com/problems/find-the-difference/) | 使用异或求出出现奇数次的那个字符/数字                        |
| 面试题05.06 | [整数转换](https://leetcode-cn.com/problems/convert-integer-lcci/) | 直接按位比较                                                 |
| 693         | [交替位二进制数](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/) | 按位比较，但是根据题意不能用for循环32位，而是“有效位数”，因此要不断右移 |
| 231         | [231. 2的幂](https://leetcode-cn.com/problems/power-of-two/) | 2的幂在二进制中只有一位是1，因此可以判断 n& (n-1)  **（可以将最右边的1变为0）** 是否等于0，或者判断 n&(-n)  （**可以使得最右边1保留，其它1变为0**）  是否等于n |
| 342         | [342. 4的幂](https://leetcode-cn.com/problems/power-of-four/) | 先按231判断2的幂，然后4的幂满足1处于奇数位上，因此还要满足与0xaaaaaaaa做与运算为0 |
| 268         | [268. 缺失数字](https://leetcode-cn.com/problems/missing-number/) | 异或                                                         |
|             |                                                              |                                                              |

#### 二分查找

[解法与总结]()

| No.   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1095  | [山脉数组中查找目标值](https://leetcode-cn.com/problems/find-in-mountain-array/) | 三次二分查找（找山顶，再在前后两个有序数组中二分找target）   |
| 162   | [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/) | 就是1095的第一步（找山顶）                                   |
| 33.   | [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) | 整个旋转数组是两段有序数组，因此可以进行二分                 |
| 34.   | [在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | 直接找左边界和有边界                                         |
| 35.   | [搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/) | 直接二分查找即可，若找不到，返回值就是插入位置               |
| 1011. | [在D天内送达包裹的能力](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/) | 本质上是枚举遍历，只不过使用二分进行了优化                   |
| 875.  | [爱吃香蕉的珂珂](https://leetcode-cn.com/problems/koko-eating-bananas/) | 类似1011，本质上是遍历优化                                   |
| 69    | [x的平方根](https://leetcode-cn.com/problems/sqrtx/)         | 二分查找(0,x)                                                |
| 287   | [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/) | 二分查找(0,n)，判定指针移动的标准是小于等于x的数的个数是否大于x（抽屉原理） |
| 209   | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 可以滑窗，也可以用前缀和构造有序数组，然后二分查找           |
|       |                                                              |                                                              |
| 4     | [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)（hard） |                                                              |

#### 链表类型题

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 21.  | [合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/) | 简单的双指针依次比较归并                                     |
| 23.  | [合并k个有序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/) | 在No.21的基础上加上分治策略                                  |
| 61.  | [旋转链表](https://leetcode-cn.com/problems/rotate-list/)    | 先连成环（这步非必须），再添加断点。本质就是找到旋转后的头节点 |
| 206. | [反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) | 迭代法：三指针逐步分析     递归法：输入一个节点 `head`，将「以 `head` 为起点」的链表反转，并返回反转之后的头结点 |
| 25   | [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/) |                                                              |
| 24   | [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/) | 可设置一个dummyhead，然后链表依次操作                        |
| 92   | [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/) |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |



#### 动态规划

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 55   | [跳跃游戏](https://leetcode-cn.com/problems/jump-game/)      | dp[i]表示是否能跳到下标为i的元素，可以使用动态规划解决       |
| 983  | [最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/) | dp[i]表示前i天买票旅行的最低消费，dp[i]由dp[i-1],dp[i-7],dp[i-30]决定 |
| 70   | [爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)  | dp[i]表示爬i层楼的方法数，dp[i] = dp[i-1] + dp[i-2];         |
| 322  | [零钱兑换](https://leetcode-cn.com/problems/coin-change/)    | dp[i]表示凑总金额i所需最少硬币数，dp[i] = min(dp[i], dp[i-coin]+1); |
| 518  | [零钱兑换Ⅱ](https://leetcode-cn.com/problems/coin-change-2/) | dp[i]表示凑总金额i的方法数，dp[i] = $ \sum$ dp[i-coin];      |
| 72   | [编辑距离](https://leetcode-cn.com/problems/edit-distance/)  | 用 dp[i] [j] 表示 `A` 的前 `i` 个字母和 `B` 的前 `j` 个字母之间的编辑距离. |
| 1340 | [跳跃游戏Ⅴ](https://leetcode-cn.com/problems/jump-game-v/)   | dp[i]表示某一点i可以到达的最大点个数，dp[i] = 1 + max(max(dp[i-d]...dp[i-1]), max(dp[i+1, i+d]))，其中要排除位置高度大于i位置的部分 |
| 221  | [最大正方形](https://leetcode-cn.com/problems/maximal-square/) | dp(*i*,*j*) 表示以 (i, j)为右下角，且只包含1的正方形的边长最大值，dp(i, j) = min(dp(i-1, j), dp(i, j-1), dp(i-1, j-1)) + 1 |
| 1277 | [统计全为1的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/) | 同221                                                        |
| 53   | [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/) | *f*(*i*) = max{*f*(*i*−1)+*ai* , *ai*}                       |
| 152  | [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) | 类似最大连续和，但是要同时维护最大乘积dp和最小乘积dp（应对负数乘积为正的情况） |
| 198  | [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/) | dp[i]表示前i+1个房屋最大偷窃金额，dp[i] = max(dp[i-1], dp[i-2] + nums[i]) |
| 213  | [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/) | 类似198，可以将环形划分解成[0:n-2]和[1:n-1]两个线性的dp，使用同198的递推式分段解决，求最大值 |
| 516  | [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/) | 子序列问题的动态规划，dp(i)(j)表示s[i]-s[j]区间内最长回文子序列长度，注意区间dp要斜着打表或者反着打表。 <br>也可以转换为求逆字符串，再求最长公共子序列 |
| 300  | [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) | dp[i]=max{dp[j]}+1，if num[i] > num[j], 其中i>j，其中dp[i]表示前i个得最长递增序列长度，且nums[i]必须选择 |
| 1143 | [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) |                                                              |
|      |                                                              |                                                              |
| 718  | [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/) |                                                              |
|      |                                                              |                                                              |
| 5419 | [5419. 两个子序列的最大点积](https://leetcode-cn.com/problems/max-dot-product-of-two-subsequences/) | 类似72，1143，  两个数组的dp， $O(n^2)$                      |
|      |                                                              |                                                              |





#### 贪心算法

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 45   | [跳跃游戏Ⅱ](https://leetcode-cn.com/problems/jump-game-ii/submissions/) | 贪心思想：每次选择能到达的最远路径，并且方法数+1，在此基础上再跳下一次 |
| 55   | [跳跃游戏](https://leetcode-cn.com/problems/jump-game/)      | 除了使用dp，也可以使用贪心思想：也是尽可能选择最远的跳，维护一个当前可跳最远距离 |
| 1029 | [两地调度](https://leetcode-cn.com/problems/two-city-scheduling/) | 首先将这 2N 个人全都安排飞往 BB 市，再选出 N 个人改变它们的行程，让他们飞往 AA 市。如果选择改变一个人的行程，那么公司将会额外付出 price_A - price_B 的费用，所以只要这部分最小即可 |
|      |                                                              |                                                              |
|      |                                                              |                                                              |



#### 二叉树及树专题

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 98   | [验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/) | 中序遍历满足递增才是二叉搜索树                               |
| 94   | [二叉树中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/) | 在中序遍历中，每个节点也会访问两次，第一次是入栈且不输出，第二次出栈 输出 |
| 144  | [二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/) | 在先序遍历中，每个节点会被访问两次，第一次是入栈，此时就输出，第二次是出栈 |
| 145  | [二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/) | 在后序遍历中，每个节点要访问三次<br/>第一次：第一次访问，第一次入栈，不输出<br/>第二次：第二次访问，第一次出栈，此时也不输出，而是进行第二次入栈，然后访问该节点的右节点<br/>第三次：第三次访问，是当访问完了某个节点的右子树，再次回到该节点时，即第二次出栈，此时输出 |
| 102  | [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/) | 结合队列，遍历一个节点后将其子节点入队（BFS）                |
| 1022 | [1022. 从根到叶的二进制数之和](https://leetcode-cn.com/problems/sum-of-root-to-leaf-binary-numbers/) | 先序遍历（DFS），移位求解，达到叶子节点则加入ans             |
| 501  | [二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/) | 先中序遍历，再对数组求众数                                   |
| 572  | [另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/) | 当前两个树的根节点值相等； 并且，s 的左子树和 t 的左子树相等； 并且，s 的右子树和 t 的右子树相等。 |
| 538  | [把二叉搜索树转换成累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/) | 反中序遍历                                                   |
| 236  | [二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) | 两个节点的最近公共祖先满足这两个节点分列左右子树，因此递归全盘搜索 |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
| 437  | [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/) | 两次dfs（先序遍历）                                          |
| 116  | [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/) |                                                              |
|      |                                                              |                                                              |
| 105  | [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | 前序遍历第一个元素就是根，根据该元素在中序遍历中找到树的左右子树，然后递归或者迭代 连接 |
| 106  | [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) | 后序遍历最后一个元素就是根，根据该元素在中序遍历中找到树的左右子树，然后递归或者迭代 连接 |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |



#### DFS/BFS

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 130  | [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) | 从边界开始dfs(或BFS)，找到不被包围的，其他就是被包围的       |
| 417  | [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/) | 从两个边界开始 两次dfs（或BFS），都遍历到的地方就是结果集一部分 |
|      |                                                              |                                                              |



#### 回溯剪枝

[解法与总结]()
| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 46   | [46. 全排列](https://leetcode-cn.com/problems/permutations/) | 回溯剪枝，可以用交换法，也可以纯回溯                         |
| 47   | [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/) | 回溯剪枝，判断好去重条件                                     |
| 784  | [784. 字母大小写全排列](https://leetcode-cn.com/problems/letter-case-permutation/) | 回溯剪枝，但结果集不需要等到搜索到叶子才添加，而是每一个节点更改都要添加 |
|      |                                                              |                                                              |




#### 栈/单调栈问题

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |      |
| :--- | ------------------------------------------------------------ | :----------------------------------------------------------- | ---- |
| 42   | [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/) |                                                              |      |
| 84   | [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/) | 求以每个矩形高度为框出来高度的最大矩形面积（这一步通过单调栈，对于某个矩形高度，找向左延伸第一个小于它的，向右延伸第一个小于它的，然后求面积），再在里面取最大的。 |      |
| 496  | [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/) | 构造一个单调递减栈，找到一个比栈顶大的就出栈，这个元素就是出栈元素后面第一个比它大的 |      |
| 503  | [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/) | 构造一个单调递减栈，与496区别在于循环判断，如[4321]相当于用496的方法计算[43214321] |      |
| 739  | [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/) | 同496，维护一个单调递减栈                                    |      |
| 239  | [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/) |                                                              |      |
| 901  | [901. 股票价格跨度](https://leetcode-cn.com/problems/online-stock-span/) | 与496，739一样的思路                                         |      |
| 402  | [402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/) | 贪心的想法+单调递增栈                                        |      |
| 33   | [面试题33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/) | 递归分治O($n^2$)，*可以用单调栈实现O(n)                      |      |
|      |                                                              |                                                              |      |
| 394  | [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/) | 维护数字栈和字符栈，碰到[入栈，碰到]出栈                     |      |
|      |                                                              |                                                              |      |
|      |                                                              |                                                              |      |

#### 滑动窗口

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1052 | [1052. 爱生气的书店老板](https://leetcode-cn.com/problems/grumpy-bookstore-owner/) | 可以维护一个大小为X的滑动窗口，O(xn)   **                    |
| 209  | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 典型滑窗，也可以用前缀和+二分法                              |
| 76   | [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) | 滑窗，判断窗口内是否满足要求，若不满足则扩大窗口，满足则尝试缩小 |
| 424  | [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/) | 滑动窗口，窗口内条件right - left + 1 - tmpMaxCnt <= k        |
|      |                                                              |                                                              |



#### 快慢指针

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 19   | [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) | 快慢指针，快指针先走n步                                      |
| 141  | [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/) | 快慢指针                                                     |
| 142  | [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) | 快慢指针，相遇时快指针走的步数=n*慢指针的，以此来找入口      |
| 287  | [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/) | 可以用二分法，也可以快慢指针思想，p=num[p]                   |
| 234  | [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/) | 空间O(1)：找中点，翻转后半段, pre指向翻转后头节点，比较两段相同长度部分 |
| 202  | [202. 快乐数](https://leetcode-cn.com/problems/happy-number/) | 快慢指针思路，用这种方法判断是不是无限循环                   |
| 876  | [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) | 快慢指针                                                     |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |





#### 拓扑排序

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                          |
| ---- | ------------------------------------------------------------ | ------------------------------- |
| 207  | [207. 课程表](https://leetcode-cn.com/problems/course-schedule/) | 拓扑排序标准模板，判断是否是DAG |
| 210  | [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/) | 带结果集的拓扑排序              |
| 329  | [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/) |                                 |
| 1203 | [1203. 项目管理](https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/) |                                 |
|      |                                                              |                                 |
|      |                                                              |                                 |



#### 回文系列

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                         |
| ---- | ------------------------------------------------------------ | ---------------------------------------------- |
| 125  | [125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/) | 首尾双指针                                     |
| 680  | [680. 验证回文字符串 Ⅱ](https://leetcode-cn.com/problems/valid-palindrome-ii/) | 首尾双指针，结合递归判断                       |
| 1328 | [1328. 破坏回文串](https://leetcode-cn.com/problems/break-a-palindrome/) |                                                |
| 9    | [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/) |                                                |
| 409  | [409. 最长回文串](https://leetcode-cn.com/problems/longest-palindrome/) |                                                |
| 5    | [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/) | 中心扩展暴力，或者manacher算法                 |
| 516  | [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/) | 注意子串和子序列的区别，子序列问题常用动态规划 |
|      |                                                              |                                                |
|      |                                                              |                                                |







#### 经典模板

| No.  | <span style="white-space:nowrap;">Template</span>            | description                                                  | case                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | 快速幂 [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/), | 从位运算或者从幂数二分的角度，将幂运算从 $O(n)$ 提升到 $O(log_2n$) | [372. 超级次方](https://leetcode-cn.com/problems/super-pow/) |
| 2    | 前缀和（积）                                                 | 即前n项的累加和（也可能是其它计数），对于无序数组计算区间和有效果。常与hash表或者桶合用 | [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/), <br>[974. 和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/)(状态压缩+前缀和)<br/>[1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/)<br> [554. 砖墙](https://leetcode-cn.com/problems/brick-wall/)<br>[238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)(前缀和后缀积的技巧)<br>[1423. 可获得的最大点数](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)<br>[1371. 每个元音包含偶数次的最长子字符串](https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/)(状态压缩+前缀和)<br>[209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)<br> |
| 3    | Manacher                                                     | 用于解决最长回文子串问题，本质是暴力中心扩展的优化           | [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)<br> |
| 4    | KMP                                                          | 经典串匹配算法，也是对暴力法的优化加速                       |                                                              |
|      |                                                              |                                                              |                                                              |
|      |                                                              |                                                              |                                                              |

