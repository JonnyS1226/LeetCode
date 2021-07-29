# 目录

<!-- TOC -->

- [目录](#目录)
- [Leetcode & codeforces](#leetcode--codeforces)
    - [位运算](#位运算)
    - [二分查找/分治减治思想](#二分查找分治减治思想)
    - [链表类型题](#链表类型题)
    - [动态规划](#动态规划)
        - [一般dp](#一般dp)
        - [树形dp](#树形dp)
        - [区间dp](#区间dp)
    - [贪心思想](#贪心思想)
    - [二叉树及树专题](#二叉树及树专题)
    - [图论题](#图论题)
        - [DFS/BFS/四种最短路算法(dijkstra,bellman-ford,spfa,floyd)](#dfsbfs四种最短路算法dijkstrabellman-fordspfafloyd)
        - [拓扑排序](#拓扑排序)
        - [并查集](#并查集)
    - [回溯剪枝](#回溯剪枝)
    - [栈/单调栈问题](#栈单调栈问题)
    - [滑动窗口/单调队列/双端队列](#滑动窗口单调队列双端队列)
    - [快慢指针/双指针](#快慢指针双指针)
    - [线段树/树状数组](#线段树树状数组)
    - [容器使用](#容器使用)
        - [set和multiset](#set和multiset)
        - [堆/优先队列实现的堆](#堆优先队列实现的堆)
        - [hashmap和hash](#hashmap和hash)
        - [其它](#其它)
    - [字符串独立专题（前缀、后缀、字典树及其它技巧）](#字符串独立专题前缀后缀字典树及其它技巧)
    - [数学题/杂项题讨论题](#数学题杂项题讨论题)
    - [回文系列](#回文系列)
    - [前缀和思想/差分](#前缀和思想差分)
    - [经典模板，思想和技巧](#经典模板思想和技巧)
    - [补充：模拟退火/梯度下降/爬山算法](#补充模拟退火梯度下降爬山算法)
- [说明](#说明)

<!-- /TOC -->


# Leetcode & codeforces

## 位运算	

[小结](bit_operation.md)

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
| 717         | [717. 1比特与2比特字符](https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/) | 总是以0结尾，所以只要看最后一个0和倒数第二个0之间1的个数(即连续的1个数)，为奇数就返回false，偶数是true，可用异或计数 |
| 201         | [201. 数字范围按位与](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/) | 使用位移，找m和n的二进制数的公共前缀                         |
| 1356        | [1356. 根据数字二进制下 1 的数目排序](https://leetcode-cn.com/problems/sort-integers-by-the-number-of-1-bits/) | 两种方式统计1的个数（不断右移或者x&(x-1)）                   |
| 393         | [393. UTF-8 编码验证](https://leetcode-cn.com/problems/utf-8-validation/) | 移位判断即可                                                 |
| 16.01       | [面试题 16.01. 交换数字](https://leetcode-cn.com/problems/swap-numbers-lcci/) | a xor a xor b = b; b xor b xor a = a                         |
| 338         | [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) | 位运算结合动态规划                                           |
| 461         | [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/) | 位运算简单题                                                 |
| 1018        | [1018. 可被 5 整除的二进制前缀](https://leetcode-cn.com/problems/binary-prefix-divisible-by-5/) | 运用取模的基本性质+位运算                                    |
| 78          | [78. 子集](https://leetcode-cn.com/problems/subsets/)        | 常规的位运算迭代枚举子集                                     |
| 1774        | [1774. 最接近目标价格的甜点成本](https://leetcode-cn.com/problems/closest-dessert-cost/) | 也可以迭代枚举子集，两次枚举二进制子集，或者直接枚举三进制子集 |
| 1178        | [1178. 猜字谜](https://leetcode-cn.com/problems/number-of-valid-words-for-each-puzzle/)(hard) | 状态压缩 + 二进制枚举子集                                    |
| 338         | [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) | 运用 (n-1)&n 和动态规划 的思想                               |
| 190         | [190. 颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits/) | 逐位颠倒，或者**位运算分治（妙啊）**                         |
| 1835        | [1835. 所有数对按位与结果的异或和](https://leetcode-cn.com/problems/find-xor-sum-of-all-pairs-bitwise-and/)（hard） | 用到`(a&b)^(a&c)=a&(b^c)`                                    |
| 1829        | [1829. 每个查询的最大异或值](https://leetcode-cn.com/problems/maximum-xor-for-each-query/) | 前缀和 + 异或运算 找到0的地方k取1，1的地方k取0               |

## 二分查找/分治减治思想

[小结](binary_search.md)

| No.   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1095  | [山脉数组中查找目标值](https://leetcode-cn.com/problems/find-in-mountain-array/) | 三次二分查找（找山顶，再在前后两个有序数组中二分找target）   |
| 162   | [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/) | 就是1095的第一步（找山顶）                                   |
| 33.   | [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) | 整个旋转数组是两段有序数组，因此可以进行二分                 |
| 81    | [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/) | 相比33 数组有重复元素，所以遇到重复元素，直接lo++，因此最坏情况复杂度On |
| 34.   | [在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | 直接找左边界和有边界                                         |
| 35.   | [搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/) | 直接二分查找即可，若找不到，返回值就是插入位置               |
| 1011. | [在D天内送达包裹的能力](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/) | 本质上是枚举遍历，只不过使用二分进行了优化                   |
| 875.  | [爱吃香蕉的珂珂](https://leetcode-cn.com/problems/koko-eating-bananas/) | 类似1011，本质上是遍历优化                                   |
| 69    | [x的平方根](https://leetcode-cn.com/problems/sqrtx/)         | 二分查找(0,x/2)， 或者牛顿迭代法                             |
| 287   | [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/) | 二分查找(0,n)，判定指针移动的标准是小于等于x的数的个数是否大于x（抽屉原理） |
| 209   | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 可以滑窗，也可以用前缀和构造有序数组，然后二分查找           |
| 1300  | [1300. Sum of Mutated Array Closest to Target](https://leetcode-cn.com/problems/sum-of-mutated-array-closest-to-target/) | 两次二分查找（一次是二分枚举，一次是二分查找大于等于数组中元素），结合前缀和 |
| 5438  | [5438. Minimum Number of Days to Make m Bouquets](https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/) | 二分查找枚举法                                               |
| 4     | [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)（hard） |                                                              |
| 74    | [74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/) | 二维->一维，二分法，或者减治缩域法，从左下或者右上开始缩     |
| 240   | [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/) | 减治缩域法，从左下或者右上开始缩                             |
| 378   | [378. 有序矩阵中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/) | 值域二分+减治缩域法                                          |
| 153   | [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/) | 左闭右闭，二分法                                             |
| 154   | [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/) | 同153，新增情况：对于nums[mid] == nums[j]， 我们可以不考虑j，因为mid总可以替代掉 |
| 410   | [410. 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/) | 同875，1011，二分枚举                                        |
| 441   | [441. 排列硬币](https://leetcode-cn.com/problems/arranging-coins/) | 二分查找模版题                                               |
| 436   | [436. 寻找右区间](https://leetcode-cn.com/problems/find-right-interval/) | 排序+二分查找                                                |
| 5563  | [5563. 销售价值减少的颜色球](https://leetcode-cn.com/problems/sell-diminishing-valued-colored-balls/) | 二分找购买后的球剩余数的下界，然后不够的再补                 |
| 778   | [778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/)（hard） | 二分答案+dfs/并查集                                          |
| 475   | [475. 供暖器](https://leetcode-cn.com/problems/heaters/)     | 二分查找, 对每个houses 二分找一个最近的heaters               |
| 395   | [395. 至少有K个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/) | 分治思想，以不满足条件的索引为分界点进行分治                 |
| 480   | [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/) | 用deque(便于增删)维护窗口+二分查找来增删元素                 |
| 1802  | [1802. 有界数组中指定下标处的最大值](https://leetcode-cn.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/) | 二分答案                                                     |

## 链表类型题

[小结](linkedlist.md)

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 21.  | [合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/) | 简单的双指针依次比较归并                                     |
| 23.  | [合并k个有序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/) | 在No.21的基础上加上分治策略（类似链表归并排序）。或者使用k指针+堆 |
| 61.  | [旋转链表](https://leetcode-cn.com/problems/rotate-list/)    | 先连成环（这步非必须），再添加断点（从前往后第n-k个，但要注意取余） |
| 147  | [147. 对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/) | 插入排序                                                     |
| 148  | [148. 排序链表](https://leetcode-cn.com/problems/sort-list/) | 链表归并排序和链表快速排序                                   |
| 24   | [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/) | 可设置一个dummyhead，然后链表依次操作                        |
| 206. | [反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) | 迭代法：三指针逐步分析     递归法：输入一个节点 `head`，将「以 `head` 为起点」的链表反转，并返回反转之后的头结点 |
| 92   | [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/) |                                                              |
| 25   | [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/) |                                                              |
| 328  | [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/) | 四个指针分别指向奇数偶数的链表头尾                           |
| 725  | [725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/) | 计算分割后每一块的长度，然后遍历链表即可                     |
| 83   | [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/) | 相同的跳过，不同的连接，迭代/递归                            |
| 82   | [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/) | 类似83，但是递归的写法要注意                                 |
| 138  | [138. 复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/) | 1. 复制结点添加在当前结点后面，2.复制random指针情况，3.将整个链表一分为二 |



## 动态规划

[小结]()

### 一般dp

| No.   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 55    | [跳跃游戏](https://leetcode-cn.com/problems/jump-game/)      | `dp[i]`表示是否能跳到下标为i的元素，可以使用动态规划解决     |
| 983   | [最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/) | `dp[i]`表示前i天买票旅行的最低消费，`dp[i]`由`dp[i-1],dp[i-7],dp[i-30]`决定 |
| 70    | [爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)  | `dp[i]`表示爬i层楼的方法数，`dp[i] = dp[i-1] + dp[i-2];`     |
| 377   | [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/) | 70的进阶版，`dp[i]+=dp[i-num]`                               |
| 46    | [面试题46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/) | 类似70                                                       |
| 322   | [零钱兑换](https://leetcode-cn.com/problems/coin-change/)    | `dp[i]`表示凑总金额i所需最少硬币数，`dp[i] = min(dp[i], dp[i-coin]+1);` |
| 279   | [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/) | 类似322，dp                                                  |
| 518   | [零钱兑换Ⅱ](https://leetcode-cn.com/problems/coin-change-2/) | `dp[i]`表示凑总金额i的方法数，`dp[i] `= $ \sum$ `dp[i-coin]`; |
| 72    | [编辑距离](https://leetcode-cn.com/problems/edit-distance/)(hard) | 用 `dp[i] [j] `表示 `A` 的前 `i` 个字母和 `B` 的前 `j` 个字母之间的编辑距离. |
| 115   | [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)(hard) | 两个字符串，子序列问题dp，类似72，1143等                     |
| 1340  | [跳跃游戏Ⅴ](https://leetcode-cn.com/problems/jump-game-v/)   | dp[i]表示某一点i可以到达的最大点个数，dp[i] = 1 + max(max(dp[i-d]...dp[i-1]), max(dp[i+1, i+d]))，其中要排除位置高度大于i位置的部分 |
| 221   | [最大正方形](https://leetcode-cn.com/problems/maximal-square/) | dp(*i*,*j*) 表示以 (i, j)为右下角，且只包含1的正方形的边长最大值，dp(i, j) = min(dp(i-1, j), dp(i, j-1), dp(i-1, j-1)) + 1 |
| 1277  | [统计全为1的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/) | 同221                                                        |
| 53    | [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/) | `f(i) = max{f(i−1)+ai , ai}`                                 |
| 152   | [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) | 类似最大连续和，但是要同时维护最大乘积dp和最小乘积dp（应对负数乘积为正的情况） |
| 1567  | [1567. 乘积为正数的最长子数组长度](https://leetcode-cn.com/problems/maximum-length-of-subarray-with-positive-product/) | 同152思路，需要维护以i结尾乘积为正和乘积为负的两个dp         |
| 5521  | [5521. 矩阵的最大非负积](https://leetcode-cn.com/problems/maximum-non-negative-product-in-a-matrix/) | 同152，只不过是二维dp，维护最大值和最小值，因为最小值可能翻身成为最大值 |
| 343   | [343. Integer Break](https://leetcode-cn.com/problems/integer-break/) | `dp[i]`表示n的题设下，分割整数后的乘积最大值                 |
| 746   | [746. Min Cost Climbing Stairs](https://leetcode-cn.com/problems/min-cost-climbing-stairs/) | `dp[i]`表示选择了i所需要的最小cost                           |
| 96    | [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/) | 讨论时分析每个数为根节点的情况，可以递推出：`dp[i] = dp[0]*dp[i-1] + dp[1]*dp[i-2] + ... + dp[i-1]*dp[0]` |
| 198   | [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/) | `dp[i]`表示前i+1个房屋最大偷窃金额，`dp[i] = max(dp[i-1], dp[i-2] + nums[i])` |
| 213   | [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/) | 类似198，可以将环形划分解成[0:n-2]和[1:n-1]两个线性的dp，使用同198的递推式分段解决，求最大值 |
| 740   | [740. 删除并获得点数](https://leetcode-cn.com/problems/delete-and-earn/) | 转换为值域，然后就可以转换为打家劫舍问题                     |
| 410   | [410. 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/)（hard） |                                                              |
| 300   | [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) | `dp[i]=max{dp[j]}+1，if num[i] > num[j]`, 其中i>j，其中dp[i]表示前i个得最长递增序列长度，且nums[i]必须选择 |
| 1143  | [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) | `dp[i] [j]`代表text1前i个字符和text2前j个字符的最长公共子序列 |
| 712   | [712. 两个字符串的最小ASCII删除和](https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings/) | 和1143一样思路，只不过目标是找到ascii码最大的子串            |
| 1770  | [1770. 执行乘法运算的最大分数](https://leetcode-cn.com/problems/maximum-score-from-performing-multiplication-operations/) | `dp[i][j]表示前i个数，后j个数乘以mul后最大值,最后遍历所有i+j=m的情况下的最大值` |
| 718   | [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/) | `dp[i] [j]`表示A前i个和B前j个 最长公共子数组长度(且要求取到公共子数组必须以i和j结尾) |
| 97    | [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)（hard） |                                                              |
| 1458  | [1458. 两个子序列的最大点积](https://leetcode-cn.com/problems/max-dot-product-of-two-subsequences/) | 类似72，1143，  两个数组的dp， $O(n^2)$                      |
| 139   | [139. 单词拆分](https://leetcode-cn.com/problems/word-break/) | 动态规划, dp[i]表示前i个字符是否能被分隔                     |
| 140   | [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/) | 动态规划 + 回溯                                              |
| 1139  | [1139. 最大的以 1 为边界的正方形](https://leetcode-cn.com/problems/largest-1-bordered-square/) | 三维dp或者用两个二维dp，分别表示向上能扩展的个数和向左能扩展的个数。更新时，看右上角和左下角分别向左和向上延伸的长度是否符合要求 |
| 392   | [392. 判断子序列](https://leetcode-cn.com/problems/is-subsequence/)（进阶挑战） |                                                              |
| 514   | [514. 自由之路](https://leetcode-cn.com/problems/freedom-trail/)（hard） | `dp[i][j]` 表示在ring的第j处找到key的第i个字符所需要移动的步数。最后返回 `*max_element(dp[m-1][t], t=1~n-1)`。使用pos记录sring中每个字符的位置 |
| 64    | [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) | `dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]​`        |
| 10    | [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)（hard） | 用dp i  j 表示 s的前 i 个字符与 p中的前 j 个字符是否能够匹配。 |
| 5411  | [5411. 摘樱桃 II](https://leetcode-cn.com/problems/cherry-pickup-ii/)（hard） | 三层dp `dp[i][j][k]`表示第i行，机器人1在j列，机器人2在k列的最大樱桃数 |
| 5431  | [5431. 给房子涂色 III](https://leetcode-cn.com/problems/paint-house-iii/)（hard） | 三维dp，`dp[i][j][k]` 表示第i个房子，涂了第j个颜色，且形成了k个社区的最小花费 |
| 837   | [837. 新21点](https://leetcode-cn.com/problems/new-21-game/) | `dp[i]`表示当前和为i（i < K）时获胜的概率， dp[i] = 摸j点的概率(1/w) 乘以 摸完之后成功的概率`(dp[i+j])`，并遍历j求和 |
| 1494  | [1494. 并行课程 II](https://leetcode-cn.com/problems/parallel-courses-ii/)（hard） |                                                              |
| 32    | [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)（hard） | 有多种方法，栈，dp，双向扫描。此处用dp，dp[i]表示以i位置结尾的最长有小括号子串长度。更新时，如果当前位置是(，显然长度为0，如果当前位置是右括号，那么要尝试找到与之对应左括号，需要判断i-`dp[i-1]-1`的位置是否是左括号，如果是：dp[i] = `dp[i-1] + 2 + dp[i-dp[i-1]-2] ` (还需要看匹配位置之前有没有有小括号) |
| 44    | [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)（hard） | `dp[i][j]`表示s前i个和p前j个是否能匹配                       |
| 174   | [174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)（hard） | 二维逆序dp                                                   |
| 121   | [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) |                                                              |
| 122   | [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/) |                                                              |
| 123   | [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/) |                                                              |
| 188   | [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/) |                                                              |
| 714   | [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) |                                                              |
| 309   | [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | dp`[i][0]`表示持有股票；dp`[i][1]`表示不持有股票，处于冷冻期;dp`[i][2]`表示不持有股票，不处于冷冻期。这里的「处于冷冻期」指的是在第 i 天结束之后的状态 |
| 1140  | [1140. 石子游戏 II](https://leetcode-cn.com/problems/stone-game-ii/) | `dp[i][j] 表示 对于 piles[i:] 和给定的 M=j 情况下的最大值`   |
| 1406  | [1406. 石子游戏 III](https://leetcode-cn.com/problems/stone-game-iii/)（hard） | `dp[i]` 表示从i开始拿，后续剩余数组 最多能领先多少           |
| 5447  | [5447. 石子游戏 IV](https://leetcode-cn.com/problems/stone-game-iv/)（hard） | 博弈dp，`dp[i]` 表示对于数i是否能先手赢                      |
| 1025  | [1025. 除数博弈](https://leetcode-cn.com/problems/divisor-game/) | 同石子游戏Ⅳ                                                  |
| LCP13 | [LCP 13. 寻宝](https://leetcode-cn.com/problems/xun-bao/)（hard） |                                                              |
| 546   | [546. 移除盒子](https://leetcode-cn.com/problems/remove-boxes/)（hard） |                                                              |
| 1024  | [1024. 视频拼接](https://leetcode-cn.com/problems/video-stitching/) | 动态规划问题，`dp[i]` 表示将区间`[0,i)`覆盖所需要的最少子区间的数量 |
| 845   | [845. 数组中的最长山脉](https://leetcode-cn.com/problems/longest-mountain-in-array/) | 两遍扫描，类似dp思想，设定left和right数组表示向左向右能扩展的最大距离 |
| LCP09 | [LCP 09. 最小跳跃次数](https://leetcode-cn.com/problems/zui-xiao-tiao-yue-ci-shu/) | 复杂度限定O(n)，所以注意剪枝                                 |
| 1269  | [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/) | `dp[i][j] 表示i次操作，在j处，方案数,dp[i][j] = dp[i-1][j] + dp[i-1][j-1] + dp[i-1][j+1]` |
| 1681  | [1681. 最小不兼容性](https://leetcode-cn.com/problems/minimum-incompatibility/)（hard） | 状态压缩动态规划                                             |
| 1187  | [1187. 使数组严格递增](https://leetcode-cn.com/problems/make-array-strictly-increasing/)（hard） | `dp[i][j] 表示arr1前i个数，经过不多于j次变化，最后一个（即第i个）数的值，最后从小到大遍历dp[n-1][j]找到第一个符合的即可` |
| 354   | [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)（hard） | 按一维排序后，转换为最大上升序列问题。然后可以使用O(n2)或者借助二分的O(nlgn)解决 |
| 368   | [368. 最大整除子集](https://leetcode-cn.com/problems/largest-divisible-subset/) | 类似最大上升子序列，最后得到了长度如何得到具体数组是关键     |
| 376   | [376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/) | dp[i]表示前i个元素摆动序列长度，第二维是状态:0表示最后上升,1表示最后下降 |
| 978   | [978. 最长湍流子数组](https://leetcode-cn.com/problems/longest-turbulent-subarray/) | 376的变形，在于要连续，所以要置1                             |
| 1691  | [1691. 堆叠长方体的最大高度](https://leetcode-cn.com/problems/maximum-height-by-stacking-cuboids/)（hard） | 可以把每个长方体所有情况放入数组一起考虑，三维最长递增子序列 |
| 813   | [813. 最大平均值和的分组](https://leetcode-cn.com/problems/largest-sum-of-averages/) | `dp[i][k] 表示前i个元素，构成k个子数组时的最大平均值`        |
| 5631  | [5631. 跳跃游戏 VI](https://leetcode-cn.com/problems/jump-game-vi/) | 动态规划+单调队列优化                                        |
| 119   | [119. 杨辉三角 II](https://leetcode-cn.com/problems/pascals-triangle-ii/) | 重点是优化空间，类似背包问题中的方式，从尾开始加，滚动数组   |
| 1771  | [1771. 由子序列构造的最长回文串的长度](https://leetcode-cn.com/problems/maximize-palindrome-length-from-subsequences/)（hard） | 类似516，`dp[i][j]表示 s[i]到s[j]范围内最长回文串长度`       |
| 132   | [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/) | 预处理回文串（dp） + LIS dp， O(n^2)复杂度                   |
| 1824  | [1824. 最少侧跳次数](https://leetcode-cn.com/problems/minimum-sideway-jumps/) | dpij表示第i个节点第j跑道 最小侧跳次数                        |
| 403   | [403. 青蛙过河](https://leetcode-cn.com/problems/frog-jump/)(hard) | `dp[i][k]`表示能否跳k步到达第i个石头                         |
|       |                                                              |                                                              |

### 树形dp

| No.    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| lcp 34 | [LCP 34. 二叉树染色](https://leetcode-cn.com/problems/er-cha-shu-ran-se-UGC/) | 树形dp, `dp[node][k] ` 表示以node为根的子树，与node相连结点数还剩k个余额的情况下，最大价值 |
| 337    | [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/) | 树形dp， `dp[node][j] `表示以node为根的子数，最大金额， j=0表示不选根，j=1表示选根 |
| 124    | [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/) | 类似树形dp的递归                                             |
| 687    | [687. 最长同值路径](https://leetcode-cn.com/problems/longest-univalue-path/) | 类似树形dp的递归                                             |
| 543    | [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/) | 类似树形dp的递归                                             |
| 1372   | [1372. 二叉树中的最长交错路径](https://leetcode-cn.com/problems/longest-zigzag-path-in-a-binary-tree/) | `dp[node][j]`表示以node为根子树最长交错路径，j=0表示下一步向左，j=1表示下一步向右 |
|        |                                                              |                                                              |

### 区间dp

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1000 | [1000. 合并石头的最低成本](https://leetcode-cn.com/problems/minimum-cost-to-merge-stones/)(hard) | 区间dp，`dp[i][j][k]为合并第i到第j堆石头为一堆的成本，每次合并k堆` |
| 5486 | [5486. 切棍子的最小成本](https://leetcode-cn.com/problems/minimum-cost-to-cut-a-stick/)（hard） | 倒过来想，区间dp，`dp[i][j] `表示区间cuts[i]到cuts[j]的距离的合并的最小代价 |
| 1690 | [1690. 石子游戏 VII](https://leetcode-cn.com/problems/stone-game-vii/) | 前缀和+区间dp                                                |
| 312  | [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)（hard） | 本质和矩阵链乘法 一样的dp；`dp[i][j] = v[i] * v[k] * [j] + dp[i][k] + dp[k][j];` |
| 877  | [877. 石子游戏](https://leetcode-cn.com/problems/stone-game/) | `dp[i][j]`表示从i到j序列，先手和后手的差值；递推时分析 如果选开头堆如何更新，选末尾堆如何更新即可推出递推式 |
| 516  | [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/) | 子序列问题的动态规划，dp(i)(j)表示s[i]-s[j]区间内最长回文子序列长度，注意区间dp要斜着打表或者反着打表。 <br>也可以转换为求逆字符串，再求最长公共子序列 |
| 87   | [87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/) | 区间dp, `dp[i][j][len]` 表示s1从i开始，s2从j开始，长度为len的字符串是否匹配。用k对len进行分割 |



## 贪心思想

[小结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 45   | [跳跃游戏Ⅱ](https://leetcode-cn.com/problems/jump-game-ii/submissions/) | 贪心思想：每次选择能到达的最远路径，并且方法数+1，在此基础上再跳下一次 |
| 55   | [跳跃游戏](https://leetcode-cn.com/problems/jump-game/)      | 除了使用dp，也可以使用贪心思想：也是尽可能选择最远的跳，维护一个当前可跳最远距离 |
| 1029 | [两地调度](https://leetcode-cn.com/problems/two-city-scheduling/) | 首先将这 2N 个人全都安排飞往 BB 市，再选出 N 个人改变它们的行程，让他们飞往 AA 市。如果选择改变一个人的行程，那么公司将会额外付出 price_A - price_B 的费用，所以只要这部分最小即可 |
| 991  | [991. 坏了的计算器](https://leetcode-cn.com/problems/broken-calculator/) | 逆向思维，y+1或者y/2，贪心让y尽可能/2                        |
| 452  | [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/) | 贪心，按`points[i][1]`排序 + 区间合并。为什么按后一个排序，因为如果按前一个排序，有可能出现A,B,C, A合并B同时又能合并C，但其实B和C不能合并 |
| 435  | [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/) | 和452一样，贪心，排序                                        |
| 1665 | [1665. 完成所有任务的最少初始能量](https://leetcode-cn.com/problems/minimum-initial-energy-to-finish-tasks/)（hard） | 二分答案+贪心                                                |
| 1663 | [1663. 具有给定数值的最小字符串](https://leetcode-cn.com/problems/smallest-string-with-a-given-numeric-value/) | 贪心，先尽量补z，再往前补和修改已有z                         |
| 659  | [659. 分割数组为连续子序列](https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/) | tail[i]存储以数字i结尾的且符合题意的连续子序列个数，nc存出现次数 |
| 1402 | [1402. 做菜顺序](https://leetcode-cn.com/problems/reducing-dishes/)（hard） | 贪心                                                         |
| 861  | [861. 翻转矩阵后的得分](https://leetcode-cn.com/problems/score-after-flipping-matrix/) | **1)**行列变换使得第一列全为1.   **2)** 列变换使得后续每一列上1的个数多于0的个数 |
| 179  | [179. 最大数](https://leetcode-cn.com/problems/largest-number/) | 贪心思想的自定义排序，这么自定义排序后的传递性证明要注意     |
| 1686 | [1686. 石子游戏 VI](https://leetcode-cn.com/problems/stone-game-vi/) | 每个石头的总价值是alice[i]+Bob[i]， 每次alice拿走石头，那么alice多了alice[i]的钱，扣除了bob bob[i]的钱。所以贪心选总价值最大的 |
| 738  | [738. 单调递增的数字](https://leetcode-cn.com/problems/monotone-increasing-digits/) | 贪心，从后往前扫描，碰到后面比前面小，就将前面减1，后面全变9 |
| 135  | [135. 分发糖果](https://leetcode-cn.com/problems/candy/)     | 贪心，两次扫描，后一位比前一位分高，就糖果+1，否则糖果=1。前一位比后一位分高并且分到的糖果不多于后一位，就糖果+1 |
| 330  | [330. 按要求补齐数组](https://leetcode-cn.com/problems/patching-array/) | 很有趣的贪心思想                                             |
| 665  | [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/) | 对于当前数，要看再前面那个数，如果再前面那个数不存在或者小于等于当前数，则修改前面数为当前数。如果大于当前数，则修改当前数为前面数。 |
| 1717 | [1717. 删除子字符串的最大得分](https://leetcode-cn.com/problems/maximum-score-from-removing-substrings/) | 贪心，先删除分值大的。可以优化预处理为总是先删“ab”，再删“ba” |
| 1775 | [1775. 通过最少操作次数使数组的和相等](https://leetcode-cn.com/problems/equal-sum-arrays-with-minimum-number-of-operations/) | 贪心, 每次都最大化变，直到sum1 >= sum2 （这个说明最后一次不用最大化变） |



## 二叉树及树专题

[小结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 98   | [验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/) | 中序遍历满足递增才是二叉搜索树；或者直接用递归：对于二叉搜索树每个结点要满足在区间[左子树，右子树] |
| 337  | [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/) | 记忆化搜索+剪枝                                              |
| 94   | [二叉树中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/) | 在中序遍历中，每个节点也会访问两次，第一次是入栈且不输出，第二次出栈 输出 |
| 144  | [二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/) | 在先序遍历中，每个节点会被访问两次，第一次是入栈，此时就输出，第二次是出栈 |
| 145  | [二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/) | 在后序遍历中，每个节点要访问三次<br/>第一次：第一次访问，第一次入栈，不输出<br/>第二次：第二次访问，第一次出栈，此时也不输出，而是进行第二次入栈，然后访问该节点的右节点<br/>第三次：第三次访问，是当访问完了某个节点的右子树，再次回到该节点时，即第二次出栈，此时输出 |
| 102  | [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/) | 结合队列，遍历一个节点后将其子节点入队（BFS）                |
| 1022 | [1022. 从根到叶的二进制数之和](https://leetcode-cn.com/problems/sum-of-root-to-leaf-binary-numbers/) | 先序遍历（DFS），移位求解，达到叶子节点则加入ans             |
| 501  | [二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/) | 先中序遍历，再对数组求众数                                   |
| 572  | [另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/) | 当前两个树的根节点值相等； 并且，s 的左子树和 t 的左子树相等； 并且，s 的右子树和 t 的右子树相等。 |
| 538  | [把二叉搜索树转换成累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/) | 反中序遍历                                                   |
| 236  | [二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) | 两个节点的最近公共祖先满足这两个节点分列左右子树，因此递归全盘搜索 |
| 297  | [297. Serialize and Deserialize Binary Tree](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)（hard） | 序列化和反序列化，用层序遍历即可                             |
| 112  | [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)  | 可以递归dfs，也可以是存储值的bfs（使用两个队列或者用pair）   |
| 113  | [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) |                                                              |
| 437  | [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/) | 两次dfs（先序遍历）                                          |
| 116  | [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/) | 根据上层指针的next，来进行填充，整体上是bfs的思路            |
| 687  | [687. 最长同值路径](https://leetcode-cn.com/problems/longest-univalue-path/) | 递归，类似124和543                                           |
| 105  | [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | 前序遍历第一个元素就是根，根据该元素在中序遍历中找到树的左右子树，然后递归或者迭代 连接 |
| 106  | [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) | 后序遍历最后一个元素就是根，根据该元素在中序遍历中找到树的左右子树，然后递归或者迭代 连接 |
| 889  | [889. 根据前序和后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/) | 类似思路，但要注意特殊情况                                   |
| 1028 | [1028. 从先序遍历还原二叉树](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/)（hard） | 通过-确定层级关系，控制出入栈                                |
| 101  | [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/) | 递归：相当于两个指针，分别比较左右子树；迭代：一次从队列取出两个 比较值是否相等或者是否只有一个为空 |
| 124  | [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)（hard） | 递归 dfs。   有点类似于树形dp的想法                          |
| 543  | [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/) | 类似124，687，dfs。有点类似树形dp思想                        |
| 99   | [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)（hard） |                                                              |
| 95   | [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/) | 考虑枚举[start,end]中的值 i 为当前二叉搜索树的根，再对划分出的两部分递归求解，最后左子树右子树各选择一颗接上去即可 |
| 110  | [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/) |                                                              |
|      | 完全二叉树                                                   |                                                              |
| 404  | [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/) | 先序遍历，递归或迭代，找到**满足左叶子条件**就记录结果       |
| 114  | [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/) | 按右左根的遍历，通过右指针组成链表                           |
| 897  | [897. 递增顺序搜索树](https://leetcode-cn.com/problems/increasing-order-search-tree/) | 类似114，按右根左的顺序进行遍历，通过右指针组成链表          |
| 430  | [430. 扁平化多级双向链表](https://leetcode-cn.com/problems/flatten-a-multilevel-doubly-linked-list/) | 同114，把child看成左，next看成右即可                         |
| 235  | [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | 可以用236的普遍方法，也可以直接利用二叉搜索树的性质求LCA     |
| 236  | [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) | 左右搜，如果分列左右，那么分岔点就是。否则就向左/右搜        |
| 834  | [834. 树中距离之和](https://leetcode-cn.com/problems/sum-of-distances-in-tree/)(hard) | 两次dfs，类似换根树形dp，`ans(child) = ans(root) - cnt(child) + N - cnt(child);` |
|      |                                                              |                                                              |
| 1584 | [1584. 连接所有点的最小费用](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/) | 最小生成树模版题，prim或者kruskal                            |
| 1489 | [1489. 找到最小生成树里的关键边和伪关键边](https://leetcode-cn.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/)（hard） | 枚举+最小生成树判定                                          |
| 450  | [450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/) | 左右子树都存在，找左子树最大的或右子树最小的作为根，然后再删除掉这个找到的结点 |
| *    | [二叉查找树中第 K 小的元素 II]((https://leetcode-cn.com/leetbook/read/high-frequency-algorithm-exercise/5hhxs5/)) | 递归，有nodenum_root，复杂度O(h)                             |
| 331  | [331. 验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/) | 运用n0=n2+1，一次遍历                                        |
| 386  | [386. 字典序排数](https://leetcode-cn.com/problems/lexicographical-numbers/) | 可以转换为10叉树先序遍历，或者直接模拟                       |
| 173  | [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/) | 迭代中序遍历即可                                             |
| 865  | [865. 具有所有最深节点的最小子树](https://leetcode-cn.com/problems/smallest-subtree-with-all-the-deepest-nodes/) | 两个递归，一个求depth，一个是不断往深的方向递归              |
| 863  | [863. 二叉树中所有距离为 K 的结点](https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/) | 用map记录父节点，然后dfs                                     |
| 1104 | [1104. 二叉树寻路](https://leetcode-cn.com/problems/path-in-zigzag-labelled-binary-tree/) | 推导公式                                                     |

## 图论题

### DFS/BFS/四种最短路算法(dijkstra,bellman-ford,spfa,floyd)

[小结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 130  | [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) | 从边界开始dfs(或BFS)，找到不被包围的，其他就是被包围的       |
| 417  | [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/) | 从两个边界开始 两次dfs（或BFS），都遍历到的地方就是结果集一部分 |
| 5426 | [5426. 重新规划路线](https://leetcode-cn.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/) | $O(n^2)$超时，可以建两种顺序的图，BFS，也可以并查集          |
| 200  | [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/) | bfs或者dfs 模版题                                            |
| 785  | [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/) | 经典染色法，dfs或者bfs                                       |
| 886  | [886. 可能的二分法](https://leetcode-cn.com/problems/possible-bipartition/) | 相当于二着色问题，每个点都要遍历一次dfs，因为可能存在非连通图 |
| 1349 | [1349. 参加考试的最大学生数](https://leetcode-cn.com/problems/maximum-students-taking-exam/)（hard） |                                                              |
| 127  | [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/) | bfs对于无向图找最短路的问题                                  |
| 433  | [433. 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/) | 同127                                                        |
| 126  | [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)（hard） |                                                              |
| 980  | [980. 不同路径 III](https://leetcode-cn.com/problems/unique-paths-iii/)(hard) | dfs， 先统计需要走的空格数，dfs的过程中如果走到终点，进行判断是否走完了所有空格 |
| 5211 | [5211. 概率最大的路径](https://leetcode-cn.com/problems/path-with-maximum-probability/) | 先要证明这种0-1之间的乘法最长路 可以用 dijkstra/bellman-ford/spfa求最短路的方法求 （反证法）。然后就可用堆优化的dijkstra，bellman-ford，或者spfa（这题没卡spfa） |
| 5410 | [5410. 课程安排 IV](https://leetcode-cn.com/problems/course-schedule-iv/) | 可以用floyd算法判断两点是否有通路，或者使用并查集            |
| 733  | [733. 图像渲染](https://leetcode-cn.com/problems/flood-fill/) | 题意就是类似油漆桶工具，存储原始color，然后使用bfs或者dfs    |
| 5490 | [5490. 吃掉 N 个橘子的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-eat-n-oranges/) | 可以用带map缓存的bfs或双向bfs                                |
| 529  | [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/) | dfs                                                          |
| 679  | [679. 24 点游戏](https://leetcode-cn.com/problems/24-game/)（hard） | 纯暴力，4种运算，4个数， dfs回溯                             |
| 5482 | [5482. 二维网格图中探测环](https://leetcode-cn.com/problems/detect-cycles-in-2d-grid/)（hard） | 带前置节点的dfs或者bfs                                       |
| 841  | [841. 钥匙和房间](https://leetcode-cn.com/problems/keys-and-rooms/) | 建图dfs， 或者拓扑排序判环                                   |
| 78   | [78. 子集](https://leetcode-cn.com/problems/subsets/)        | 简单dfs                                                      |
|      |                                                              |                                                              |
| 1034 | [1034. 边框着色](https://leetcode-cn.com/problems/coloring-a-border/) | DFS求连通分量的边界（通过控制属于该分量返回1，不属于返回0来实现） |
| 1293 | [1293. 网格中的最短路径](https://leetcode-cn.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/)（hard） | bfs，增加一维 k （重点题）                                   |
| 1654 | [1654. 到家的最少跳跃次数](https://leetcode-cn.com/problems/minimum-jumps-to-reach-home/) | bfs，和1293很像，增加一维状态                                |
| 847  | [847. 访问所有节点的最短路径](https://leetcode-cn.com/problems/shortest-path-visiting-all-nodes/) |                                                              |
| 854  | [854. 相似度为 K 的字符串q](https://leetcode-cn.com/problems/k-similar-strings/) |                                                              |
| 1345 | [1345. 跳跃游戏 IV](https://leetcode-cn.com/problems/jump-game-iv/)(hard) | bfs+倒排索引+同值跳跃只发生一次                              |
| 301  | [301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)（hard） | 对当前串，考虑多删除一个字符后的所有新串，做bfs              |
| 1774 | [1774. 最接近目标价格的甜点成本](https://leetcode-cn.com/problems/closest-dessert-cost/) | 可以用dfs枚举情况                                            |
| 1786 | [1786. 从第一个节点出发到最后一个节点的受限路径数](https://leetcode-cn.com/problems/number-of-restricted-paths-from-first-to-last-node/) | 堆优化的dijkstra + dp（注意剪枝）                            |
| 79   | [79. 单词搜索](https://leetcode-cn.com/problems/word-search/) | 对每个点做dfs                                                |

### 拓扑排序

[小结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 207  | [207. 课程表](https://leetcode-cn.com/problems/course-schedule/) | 拓扑排序标准模板，判断是否是DAG                              |
| 210  | [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/) | 带结果集的拓扑排序                                           |
| 329  | [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/) |                                                              |
| 1203 | [1203. 项目管理](https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/) |                                                              |
| 851  | [851. 喧闹和富有](https://leetcode-cn.com/problems/loud-and-rich/) | 富->穷，构建有向图，拓扑排序，每次遍历邻接点时找所有点中最安静的 |
|      |                                                              |                                                              |



### 并查集

[小结]()

| No    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 5426  | [5426. 重新规划路线](https://leetcode-cn.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/) | 此处使用并查集                                               |
| 5410  | [5410. 课程安排 IV](https://leetcode-cn.com/problems/course-schedule-iv/) | 此处使用并查集                                               |
| 130   | [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) | 并查集模版题，将外围的O与dummyNode作为一个集合，内部是相连的作为一个集合，最终不和dummyNode一个集合的就全部修改掉 |
| 959   | [959. 由斜杠划分区域](https://leetcode-cn.com/problems/regions-cut-by-slashes/) | 将每一个小正方形按对角线分成四块，遇到/和\分别合并其中两块   |
| 17.07 | [面试题 17.07. 婴儿名字](https://leetcode-cn.com/problems/baby-names-lcci/) |                                                              |
| 128   | [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)（hard） | 维护一个map(size)，将num，num-1，num+1这样的merge，并根据连通分量的size不断更新ans |
| 990   | [990. Satisfiability of Equality Equations](https://leetcode-cn.com/problems/satisfiability-of-equality-equations/) | 等号则连通，不等号则判断左右两个是否在一个连通里，如果在，则返回false |
| 785   | [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/) |                                                              |
| 684   | [684. 冗余连接](https://leetcode-cn.com/problems/redundant-connection/) | 并查集模版题                                                 |
| 685   | [685. 冗余连接 II](https://leetcode-cn.com/problems/redundant-connection-ii/) |                                                              |
| 5497  | [5497. 查找大小为 M 的最新分组](https://leetcode-cn.com/problems/find-latest-group-of-size-m/) |                                                              |
| 765   | [765. 情侣牵手](https://leetcode-cn.com/problems/couples-holding-hands/)(hard) | 最后返回 情侣个数-环个数（连通分量个数）。 这道题很难想到用并查集 ！！ |
| 839   | [839. 相似字符串组](https://leetcode-cn.com/problems/similar-string-groups/)(hard) | 并查集+similar函数的编写+对于长单词和短单词分别讨论是用枚举还是遍历（这样控制复杂度O（n^3）） |
| 778   | [778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/) | 二分答案+dfs/并查集                                          |
| 1627  | [1627. 带阈值的图连通性](https://leetcode-cn.com/problems/graph-connectivity-with-threshold/)(hard) | 并查集+数学技巧优化                                          |
| 547   | [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/) | 并查集模版题                                                 |
| 947   | [947. 移除最多的同行或同列石头](https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/) | 并查集模版题，初始化parents时需要注意                        |
| 1202  | [1202. 交换字符串中的元素](https://leetcode-cn.com/problems/smallest-string-with-swaps/) | 索引对作为边，进行并查集合并。最后每一个连通分量中字母排序   |
| 721   | [721. 账户合并](https://leetcode-cn.com/problems/accounts-merge/) | 并查集，注意建立序号和邮箱映射                               |
| 803   | [803. 打砖块](https://leetcode-cn.com/problems/bricks-falling-when-hit/)(hard) | 逆向思维，每次补石块，看看导致新增多少石块和根相连了         |
| 1319  | [1319. 连通网络的操作次数](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/) | 并查集模版题，返回连通分量个数                               |
| 1579  | [1579. 保证图可完全遍历](https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/)（hard） | 逆向思维，补线。需要两个并查集。先补type3的，再补1，2. 补的过程中如果两点已联通，则这条边可删除。 |
| 1631  | [1631. 最小体力消耗路径](https://leetcode-cn.com/problems/path-with-minimum-effort/) | 抽象为图论，边权重定义为高度差绝对值，排序后以此加入，直到左上角和右下角联通。或者用二分+bfs、Dijkstra |


## 回溯剪枝

[小结]()

| No    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 46    | [46. 全排列](https://leetcode-cn.com/problems/permutations/) | 回溯剪枝，可以用交换法，也可以选取法                         |
| 47    | [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/) | 回溯剪枝，排序，判断好去重条件(前后两个数相等，并且前面一个数没有使用) |
| 90    | [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)  | 回溯，去重                                                   |
| 784   | [784. 字母大小写全排列](https://leetcode-cn.com/problems/letter-case-permutation/) | 回溯剪枝，但结果集不需要等到搜索到叶子才添加，而是每一个节点更改都要添加 |
| 93    | [93. 复原IP地址](https://leetcode-cn.com/problems/restore-ip-addresses/) |                                                              |
| 140   | [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/) | 回溯剪枝                                                     |
| 491   | [491. 递增子序列](https://leetcode-cn.com/problems/increasing-subsequences/) | 回溯剪枝                                                     |
| 37    | [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)（hard） | 回溯法                                                       |
| 5520  | [5520. 拆分字符串使唯一子字符串的数目最大](https://leetcode-cn.com/problems/split-a-string-into-the-max-number-of-unique-substrings/) | dfs+回溯                                                     |
| 698   | [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/) | 回溯，构建k个和为target的集合，在每个集合构建时用回溯        |
| 473   | [473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/) | No.698 的k=4的特殊情况                                       |
| 77    | [77. 组合](https://leetcode-cn.com/problems/combinations/)   | 有顺序的回溯                                                 |
| 39    | [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/) | 可以重复选取的回溯                                           |
| 40    | [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/) | 回溯去重                                                     |
| 216   | [216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/) | 可以构造数组进行回溯（回溯函数中写一次递归即可），也可以直接回溯（分别考虑取和不取两种情况，两次递归） |
| 08.12 | [面试题 08.12. 八皇后](https://leetcode-cn.com/problems/eight-queens-lcci/) | 经典回溯                                                     |
| 842   | [842. 将数组拆分成斐波那契序列](https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/) | 回溯，剪枝情况很多                                           |
| 306   | [306. 累加数](https://leetcode-cn.com/problems/additive-number/) | 和842完全相同的回溯，当然更好的做法是用字符串处理溢出情况    |
| 1718  | [1718. 构建字典序最大的可行序列](https://leetcode-cn.com/problems/construct-the-lexicographically-largest-valid-sequence/) | 类似8皇后，回溯填格子                                        |
| 22    | [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/) | 任何时刻左括号数量大于等于右括号数量， 用这个条件做回溯      |
| 89    | [89. 格雷编码](https://leetcode-cn.com/problems/gray-code/)  | 1. 异或处理只改变一位的问题， 2. 回溯的写法！                |
| 131   | [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/) | 回溯+check 模版题                                            |






## 栈/单调栈问题

[小结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| :--- | ------------------------------------------------------------ | :----------------------------------------------------------- |
| 42   | [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)（hard） | 典型单调递减栈题目                                           |
| 84   | [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/) | 求以每个矩形高度为框出来高度的最大矩形面积（这一步通过单调栈，对于某个矩形高度，找向左延伸第一个小于它的，向右延伸第一个小于它的，然后求面积），再在里面取最大的。 |
| 85   | [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/) | 类似84，将y轴置于最左边，将x轴作用于每一行，都调用一次84，最后取最大 |
| 1793 | [1793. 好子数组的最大分数](https://leetcode-cn.com/problems/maximum-score-of-a-good-subarray/)(hard) | 类似84，加一个简单的限制条件                                 |
| 496  | [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/) | 构造一个单调递减栈，找到一个比栈顶大的就出栈，这个元素就是出栈元素后面第一个比它大的 |
| 503  | [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/) | 构造一个单调递减栈，与496区别在于循环判断，如[4321]相当于用496的方法计算[43214321] |
| 739  | [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/) | 同496，维护一个单调递减栈                                    |
|      |                                                              |                                                              |
| 901  | [901. 股票价格跨度](https://leetcode-cn.com/problems/online-stock-span/) | 与496，739一样的思路                                         |
| 402  | [402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/) | 贪心的想法+栈，使靠前的数尽量小，所以用单调递增栈            |
| 316  | [316. 去除重复字母](https://leetcode-cn.com/problems/remove-duplicate-letters/)（hard） | 类似402，使用单调递增栈，同时要满足不能删光也不能超过总数（cnt来控制），不能多加入（vis来控制） |
| 1081 | [1081. 不同字符的最小子序列](https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters/) | 和316一样                                                    |
| 321  | [321. 拼接最大数](https://leetcode-cn.com/problems/create-maximum-number/)(hard) | 先求两个nums中能构成的长为k的最大数（单调栈、402题），然后使用类似归并的思想合并 |
| 33   | [面试题33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/) | 递归分治O($n^2$)，*可以用单调栈实现O(n)                      |
| 5420 | [5420. Final Prices With a Special Discount in a Shop](https://leetcode-cn.com/problems/final-prices-with-a-special-discount-in-a-shop/) | 单调栈的简单应用                                             |
| 394  | [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/) | 维护数字栈和字符栈，碰到[入栈，碰到]出栈                     |
| 5614 | [5614. 找出最具竞争力的子序列](https://leetcode-cn.com/problems/find-the-most-competitive-subsequence/) | 限制栈大小的单调栈，要考虑栈内元素不够的情况                 |
| 1526 | [1526. 形成目标数组的子数组最少增加次数](https://leetcode-cn.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/)(hard） | 可以利用单调栈思想：考虑每个元素左侧相邻元素的贡献值，但不同于常规单调栈，不需要所有出栈都计算 |
| 1544 | [1544. 整理字符串](https://leetcode-cn.com/problems/make-the-string-great/) | 用数组模拟栈，或者直接用栈                                   |
| 862  | [862. 和至少为 K 的最短子数组](https://leetcode-cn.com/problems/shortest-subarray-with-sum-at-least-k/) | 前缀和+双端队列模拟的单调递增栈                              |
| 1776 | [1776. 车队 II](https://leetcode-cn.com/problems/car-fleet-ii/)(hard) | 首先，要追上前面的车，一定速度大于前车；所以 很显然只要考虑车子右边，因此从后往前遍历；所以 很显然只要考虑车子右边，因此从后往前遍历 |
| 224  | [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)(hard) | 栈 + 拆括号                                                  |
| 227  | [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/) | 双栈                                                         |
| 150  | [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/) | 经典问题，一个数字栈即可                                     |
| 1047 | [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/) | 栈的简单使用，关键是要想到用栈                               |
| 456  | [456. 132 模式](https://leetcode-cn.com/problems/132-pattern/) | 记录左侧最小值作为“1”（贪心思想）， 单调递减栈 找小于“3”的右边的最大元素（因此从右往左遍历）。 最后判断找到的“2” 是否大于 “1” |

## 滑动窗口/单调队列/双端队列

[小结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1052 | [1052. 爱生气的书店老板](https://leetcode-cn.com/problems/grumpy-bookstore-owner/) | 可以维护一个大小为X的滑动窗口+逆向思维。或者直接使用前缀和+滑动窗口，正向考虑 |
| 209  | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 典型滑窗，也可以用前缀和+二分法                              |
| 76   | [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)（hard） | 滑窗，判断窗口内是否满足要求，若不满足则扩大窗口，满足则尝试缩小 |
| 632  | [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)（hard） | 可以使用hash+排序+滑动窗口，转换成和76类似的题目             |
| 424  | [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/) | 滑动窗口，窗口内条件right - left + 1 - tmpMaxCnt <= k        |
| 1456 | [1456. 定长子串中元音的最大数目](https://leetcode-cn.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/) | map+滑动窗口                                                 |
| 5423 | [5423. Find Two Non-overlapping Sub-arrays Each With Target Sum](https://leetcode-cn.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/) | 从后往前的滑动窗口+dp                                        |
| 862  | [862. 和至少为 K 的最短子数组](https://leetcode-cn.com/problems/shortest-subarray-with-sum-at-least-k/) | 前缀和+双端队列模拟的单调递增栈                              |
| 239  | [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)（hard） | 双端队列，front存最大，本质也是双端队列模拟单调栈。单调队列  |
| 1499 | [1499. 满足不等式的最大值](https://leetcode-cn.com/problems/max-value-of-equation/) | 即求 `max(yi + yj + xj - xi) = max(xj + yj) + max(yi - xi), i < j`，转换后就变成了239题，单调队列求滑动窗口内（区间内）最大值 |
| 632  | [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)（hard） | hash+排序后滑动窗口                                          |
| 649  | [649. Dota2 参议院](https://leetcode-cn.com/problems/dota2-senate/) | 循环队列+贪心模拟                                            |
| 1208 | [1208. 尽可能使字符串相等](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/) | 滑动窗口模版题                                               |
| 713  | [713. 乘积小于K的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/) | 求最大满足的情况：对于[l,r]区间内符合条件，那么它所有符合条件的且以r为右端点的子数组一共是r-l+1个 |
| 992  | [992. K 个不同整数的子数组](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)（hard） | 求最多K个不同子数组-最多K-1个不同子数组 即可。而求最多的情况类似713。 |
| 567  | [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/) | 滑动窗口模版题                                               |
| 995  | [995. K 连续位的最小翻转次数](https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/)（hard） | 队列模拟的滑动窗口+贪心                                      |
| 1004 | [1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/) | 统计0个数，滑动窗口                                          |
| 697  | [697. 数组的度](https://leetcode-cn.com/problems/degree-of-an-array/) | 可以使用滑动窗口，也可以用多个hashmap记录首尾位置            |
| 1438 | [1438. 绝对差不超过限制的最长连续子数组](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/) | 滑动窗口+单调队列维护区间最大最小值                          |
| 1358 | [1358. 包含所有三种字符的子字符串数目](https://leetcode-cn.com/problems/number-of-substrings-containing-all-three-characters/) | 滑动窗口，注意累加的时候，[i,j]符合条件的话，[i,j+1],[i,j+2]这些都会符合条件 |
| 1838 | [1838. 最高频元素的频数](https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element/) | 上界右移一位 增加(right - left) * (nums[right] - nums[right-1]). presum超过k，则收缩下界 |

## 快慢指针/双指针

[小结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 19   | [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) | 快慢指针，快指针先走n步                                      |
| 141  | [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/) | 快慢指针                                                     |
| 142  | [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) | 快慢指针，相遇时快指针走的步数=n*慢指针的，以此来找入口      |
| 287  | [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/) | 可以用二分法，也可以快慢指针思想，p=num[p]                   |
| 234  | [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/) | 空间O(1)：找中点，翻转后半段, pre指向翻转后头节点，比较两段相同长度部分 |
| 202  | [202. 快乐数](https://leetcode-cn.com/problems/happy-number/) | 快慢指针思路，用这种方法判断是不是无限循环                   |
| 876  | [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) | 快慢指针                                                     |
| 26   | [26. Remove Duplicates from Sorted Array](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/) | 快慢指针原地修改题                                           |
| 27   | [27. Remove Element](https://leetcode-cn.com/problems/remove-element/) | 快慢指针原地修改题                                           |
| 283  | [283. Move Zeroes](https://leetcode-cn.com/problems/move-zeroes/) | 快慢指针原地修改题                                           |
| 16   | [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/) | 排序+三指针                                                  |
| 1537 | [1537. 最大得分](https://leetcode-cn.com/problems/get-the-maximum-score/) | 分段求值，相当于记录下岔路口时的 最优解。可以dp，也可以优化为双指针 |
| 143  | [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/) | 快慢指针找中点(向前取)，分出两段，后面的插入到前面           |
| 763  | [763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/) | map存取最后出现的位置+双指针循环更新                         |
| 56   | [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/) | 和57一样，先排序，再遍历。根据指针分析可以判断sort二维数组时，默认按每行第一个进行递增排序 |
| 327  | [327. 区间和的个数](https://leetcode-cn.com/problems/count-of-range-sum/) | 前缀和 + 归并排序。即如果是两个有序的数组，可以通过双指针求出符合条件的区间。此处相当于用归并构建了不断两个有序数组来求解 （是否有序 不影响最终结果，只不过有序的话方便求解，所以这种方式是正确的） |
| 5602 | [5602. 将 x 减到 0 的最小操作数](https://leetcode-cn.com/problems/minimum-operations-to-reduce-x-to-zero/) | 最快的方法是双指针，左右和=x 相当于 找中间最长的连续的和等于sum-x。或者用前缀和+哈希，二分等都可 |
| 632  | [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)(hard) | 小顶堆 + 多指针，每次维护n个值中的最大和最小值，然后不断更新min和max |
| 23.  | [合并k个有序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)(hard) | 小顶堆+多指针，和632类似                                     |
| 1675 | [1675. 数组的最小偏移量](https://leetcode-cn.com/problems/minimize-deviation-in-array/)（hard） | 相当于把每个元素可变的情况都加入进去，然后用632的代码求最小覆盖区间 |
| 165  | [165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/) | 双指针处理                                                   |
| 581  | [581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/) | 构建一个排序后数组，然后双指针                               |
| 324  | [324. 摆动排序 II](https://leetcode-cn.com/problems/wiggle-sort-ii/) | 排序后，后一半插入到前一半中，双指针，单数注意重复数字的处理 |
| 888  | [888. 公平的糖果棒交换](https://leetcode-cn.com/problems/fair-candy-swap/) | 排序+双指针, 根据差值情况更新diff                            |
| 15   | [15. 三数之和](https://leetcode-cn.com/problems/3sum/)       | 暴力枚举+二分$O(n^2\log n)$超时，使用枚举+双指针$O(n^2)$     |
| 160  | [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/) | 相当于让一个指针先走一个长度差的距离，然后再一起走，就能得到第一个相遇点 |
| 75   | [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/) | 类似三路快排，0放左边，1不动，2放右边                        |
| 80   | [80. 删除有序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/) | 模拟记录一个len，如果新来的元素不会构成连续三个相同元素，就加入序列 |

## 线段树/树状数组

[小结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 315  | [315. 计算右侧小于当前元素的个数](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)（hard） | 树状数组模版题                                               |
| 327  | [327. 区间和的个数](https://leetcode-cn.com/problems/count-of-range-sum/)（hard） | 用树状数组，对于每一个presum[i] 用BIT求前缀的方式 求 [presum[i]-upper, presum[i]-lower] 的元素个数 |
| 5564 | [5564. 通过指令创建有序数组](https://leetcode-cn.com/problems/create-sorted-array-through-instructions/)（hard） | 树状数组的模版题，和315类似                                  |
| 493  | [493. 翻转对](https://leetcode-cn.com/problems/reverse-pairs/) | 类似327，计算[1, maxVal] - [1, 2*nums[i]]，这样就得到了以i为右端点的翻转对 |
|      |                                                              |                                                              |
|      |                                                              |                                                              |


## 容器使用
[小结]()

### set和multiset

| No       | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 349      | [349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/) | 一个set存放第一个数组元素，一个set用于验证第二个数组中交集部分 |
| 218      | [218. 天际线问题](https://leetcode-cn.com/problems/the-skyline-problem/)（hard） | 碰到建筑左侧 高度加入multiset； 碰到右侧，弹出对应的高度。multiset+扫描 |
| 220      | [220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/) | 排序容器（set，ordered_map等）存储k个元素                    |
| Cf 1512D | [Corrupted Array](https://codeforces.ml/problemset/problem/1512/D) | 有关系式B=2A+x，所以基于multiset枚举x，再验证B=2A            |

### 堆/优先队列实现的堆

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 牛客 | [排队](https://ac.nowcoder.com/acm/contest/6488/C)           | 优先队列实现小顶堆+归并排序/树状数组求逆序对                 |
| 632  | [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/) | k指针+最小堆，维护最大值和最小值，取最优区间（或者分别维护最大最小堆也可） |
| 973  | [973. 最接近原点的 K 个点](https://leetcode-cn.com/problems/k-closest-points-to-origin/) | 本质上是topk问题，用优先队列实现的大顶堆即可                 |
| 295  | [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)（hard） | 两个堆，保持平衡，一个存放小一半，一个存放大一半的元素       |
| 767  | [767. 重构字符串](https://leetcode-cn.com/problems/reorganize-string/) | 构造出现频数大顶堆，每次出堆两个字符，加到ans上，这样能保证相邻字符不重复 |
| 1046 | [1046. 最后一块石头的重量](https://leetcode-cn.com/problems/last-stone-weight/) | 优先队列（堆）模拟                                           |
| 1705 | [1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/) | 优先队列（堆）模拟                                           |
| 5703 | [5703. 最大平均通过率](https://leetcode-cn.com/problems/maximum-average-pass-ratio/) | 堆，贪心策略每次选增量最大的                                 |
|      |                                                              |                                                              |



### hashmap和hash

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 350  | [350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/) | hash表，或者排序双指针                                       |
| 1365 | [1365. 有多少小于当前数字的数字](https://leetcode-cn.com/problems/how-many-numbers-are-smaller-than-the-current-number/) | hash存放比当前数小的数个数，从小到大累加来算                 |
| 49   | [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/) | 定义一个`unordered_map<string, vector<string>>`              |
| 149  | [149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/)（hard） | 本质就是暴力法，用hash做了简单优化                           |
| 290  | [290. 单词规律](https://leetcode-cn.com/problems/word-pattern/) | 双向映射，两个hash表存储关系                                 |
| 1733 | [1733. 需要教语言的最少人数](https://leetcode-cn.com/problems/minimum-number-of-people-to-teach/) | \1. 记录每个人会什么语言，\2. 找到需要解决沟通问题的朋友对， \3. 遍历语言，遍历朋友对，教语言 |
| 146  | [146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/) | 双向链表 + hashmap，头部存放最近使用的，尾部存放最近最久未使用的 |
| 705  | [705. 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/) | 链地址法，数组+链表                                          |
| 706  | [706. 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/) | 链地址法，数组+链表+自定义node                               |
| 781  | [781. 森林中的兔子](https://leetcode-cn.com/problems/rabbits-in-forest/) | 以兔子说的个数为组，进行分组统计（hash），对号入座           |



### 其它

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 169  | [169. 多数元素](https://leetcode-cn.com/problems/majority-element/) | 摩尔投票法，一个候选人                                       |
| 229  | [229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/) | 摩尔投票法，两个候选人                                       |
| 1287 | [1287. 有序数组中出现次数超过25%的元素](https://leetcode-cn.com/problems/element-appearing-more-than-25-in-sorted-array/) | 可以用摩尔投票法，找三个候选人                               |
| 164  | [164. 最大间距](https://leetcode-cn.com/problems/maximum-gap/)（hard） | 桶排序+鸽巢。基本原理就是：这么多元素，平均gap是(max-min)/(n-1)，很显然最大gap一定是大于等于平均gap的。我们让每个桶的size是这么多。这样的话，最大gap只可能出现在桶间，那么桶也只要维护最小和最大元素，然后遍历求桶间的最大间隔即可。 |
| 1122 | [1122. 数组的相对排序](https://leetcode-cn.com/problems/relative-sort-array/) | 简单计数排序/桶排序                                          |



## 字符串独立专题（前缀、后缀、字典树及其它技巧）

[小结]()
| No    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 459   | [459. 重复的子字符串](https://leetcode-cn.com/problems/repeated-substring-pattern/) | 重复子串的特性                                               |
| 1044  | [1044. 最长重复子串](https://leetcode-cn.com/problems/longest-duplicate-substring/) |                                                              |
| 67    | [67. 二进制求和](https://leetcode-cn.com/problems/add-binary/) | 模拟字符串n进制求和即可                                      |
| 989   | [989. 数组形式的整数加法](https://leetcode-cn.com/problems/add-to-array-form-of-integer/) | 大数加法                                                     |
| 43    | [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/) | 大数乘法                                                     |
| 17.13 | [面试题 17.13. 恢复空格](https://leetcode-cn.com/problems/re-space-lcci/) |                                                              |
| 336   | [336. 回文对](https://leetcode-cn.com/problems/palindrome-pairs/)（hard） |                                                              |
| 208   | [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/) | 字典树模版                                                   |
| 212   | [212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)（hard） | 字典树+回溯                                                  |
| 472   | [472. 连接词](https://leetcode-cn.com/problems/concatenated-words/)(hard) | 字典树+每个单词在树上的dfs搜索                               |
| 1707  | [1707. 与数组中元素的最大异或值](https://leetcode-cn.com/problems/maximum-xor-with-an-element-from-array/) | 0-1字典树, 字典树要存以当前节点为根节点的子树中的最小元素， 查询时从最高位开始处理，如果x_i当前位是1，那么优先走0分支；反之亦然 |
| 151   | [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/) | 先翻转整个字符串，再反转每个单词，再去除空格                 |



## 数学题/杂项题讨论题

[小结]()

| No      | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 961     | [961. 重复 N 次的元素](https://leetcode-cn.com/problems/n-repeated-element-in-size-2n-array/) | 推理：相当于N个不相同的元素，插入N个相同的元素，一定存在连续3个元素中有2个相同的值，这个值就是结果 |
| 189     | [189. Rotate Array](https://leetcode-cn.com/problems/rotate-array/) |                                                              |
| 16.18   | [面试题 16.18. 模式匹配](https://leetcode-cn.com/problems/pattern-matching-lcci/) | 暴力枚举，+ 严格分类讨论                                     |
| 31      | [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/) | 两遍扫描，找到一个较小数和较大数，且两者要接近，然后后面要排序，使得变化幅度小 |
| 60      | [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)（hard） | 回溯剪枝也很可能超时，使用康托展开和逆康托展开               |
| 134     | [134. 加油站](https://leetcode-cn.com/problems/gas-station/) | 要满足总gas-cost>=0，每个子部分gas-cost>=0                   |
| 48      | [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/) | 原地旋转。先转置，再换列                                     |
| 204     | [204. 计数质数](https://leetcode-cn.com/problems/count-primes/) | 经典素数筛算法                                               |
| 1012    | [1012. 至少有 1 位重复的数字](https://leetcode-cn.com/problems/numbers-with-repeated-digits/)（hard） | 数位dp+反面+排列组合                                         |
| 621     | [621. 任务调度器](https://leetcode-cn.com/problems/task-scheduler/) | 推公式 贪心                                                  |
| 1359    | [1359. 有效的快递序列数目](https://leetcode-cn.com/problems/count-all-valid-pickup-and-delivery-options/)（hard） | 排列组合                                                     |
| 398*    | [398. 随机数索引](https://leetcode-cn.com/problems/random-pick-index/) | 每次pick都看成在ans数组中，一个一个读入值为target的元素，然后用蓄水池抽样法随机返回索引 |
| 382*    | [382. 链表随机节点](https://leetcode-cn.com/problems/linked-list-random-node/) | 蓄水池抽样算法                                               |
| 384*    | [384. 打乱数组](https://leetcode-cn.com/problems/shuffle-an-array/) | 经典洗牌算法                                                 |
| 1232    | [1232. 缀点成线](https://leetcode-cn.com/problems/check-if-it-is-a-straight-line/) | 可以求直线一般式或者两点式，也可以使用线性相关特性，两个向量线性相关 -> 构成的行列式值为0 |
| 319*    | [319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/) | 转化为分解因子 -> 求1-n中平方数个数 -> 求平方根问题          |
| 73      | [73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/) | 首先先开两个标记 首行和首列是否要清0，然后对于非首行首列为0的，将其转换为对应的(row, 0), (0, col)，最后再根据flag和(row,0),(0,col)进行置零操作 |
| 1798    | [1798. 你能构造出连续值的最大数目](https://leetcode-cn.com/problems/maximum-number-of-consecutive-values-you-can-make/) | 数学题：因为要求从0开始，假设[0,x]可覆盖，现在遇到y，则[y, y+x]肯定能覆盖。要使得可以延伸区间，则一定有y<=x+1 |
| 5725    | [5725. 序列中不同最大公约数的数目](https://leetcode-cn.com/problems/number-of-different-subsequences-gcds/)(hard) | 一个序列中最大公约数为g，当且仅当这个序列中所有g的倍数的最大公约数为g。g[y] 表示当前遍历过的数中，y的倍数的最大公约数。g[y] 表示当前遍历过的数中，y的倍数的最大公约数。对每个数进行求因子 |
| cf1512E | [Permutation by Sum](https://codeforces.ml/problemset/problem/1512/E) | 本质就是k个格子，从大到小遍历元素，根据k个元素可满足的最大和最小值，判断当前元素是否可以放入这个格子 |

## 回文系列

[小结  ]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 125  | [125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/) | 首尾双指针                                                   |
| 680  | [680. 验证回文字符串 Ⅱ](https://leetcode-cn.com/problems/valid-palindrome-ii/) | 首尾双指针，结合递归判断                                     |
| 1328 | [1328. 破坏回文串](https://leetcode-cn.com/problems/break-a-palindrome/) |                                                              |
| 9    | [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/) | 可以转换为字符串，或者取出每位，计算其逆序的数，优化方案是不需要每一位都取出来，只要取出一半 |
| 409  | [409. 最长回文串](https://leetcode-cn.com/problems/longest-palindrome/) |                                                              |
| 5    | [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/) | 中心扩展暴力，或者manacher算法                               |
| 516  | [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/) | 注意子串和子序列的区别，子序列问题常用动态规划               |
| 214  | [214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)（hard） | 用kmp, s匹配~s 即可, 匹配剩下的部分加上 就是答案   或者manacher |
| 1312 | [1312. 让字符串成为回文串的最少插入次数](https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)（hard） | 经典回文序列dp法，但注意如何根据dp值还原生成的回文串         |



## 前缀和思想/差分

[小结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 560  | [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/) | 前缀和+map                                                   |
| 5471 | [5471. 和为目标值的最大数目不重叠非空子数组数目](https://leetcode-cn.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/) | 560的变形，要清空前缀和和map                                 |
| 1248 | [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/) | 前缀和+map, pre 存放前n项奇数的个数,map key->pre  value->出现次数 |
| 554  | [554. 砖墙](https://leetcode-cn.com/problems/brick-wall/)    | 前缀和+map, map key->preNum, value->出现次数                 |
| 1423 | [1423. 可获得的最大点数](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/) | 先计算前缀和，然后求left(0-i)和right(len-k+i+1, len-1)的最大值；或者反向思考，维护一个n-k长度的sliding window，使窗口内最小 |
| 238  | [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/) | 两次扫描，前缀积和后缀积                                     |
| 209  | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 前缀和构造有序数组，然后使用二分                             |
| 974  | [974. 和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/) | 状态压缩+前缀和                                              |
| 1371 | [1371. 每个元音包含偶数次的最长子字符串](https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/) | 状态压缩+前缀和                                              |
| 5485 | [5485. 找出最长的超赞子字符串](https://leetcode-cn.com/problems/find-longest-awesome-substring/)（hard） | 状态压缩+前缀和                                              |
| 303  | [303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/) | 前缀和模版                                                   |
| 104  | [304. 二维区域和检索 - 矩阵不可变](https://leetcode-cn.com/problems/range-sum-query-2d-immutable/) | 二维前缀和模版                                               |
| 307  |                                                              |                                                              |
| 1664 | [1664. 生成平衡数组的方案数](https://leetcode-cn.com/problems/ways-to-make-a-fair-array/) | 分奇偶的前缀和                                               |
| 1653 | [1653. 使字符串平衡的最少删除次数](https://leetcode-cn.com/problems/minimum-deletions-to-make-string-balanced/) | 用两个数组分别表示某个位置前缀a个数和后缀b个数，最终就是n-max(前缀+后缀) |
|      |                                                              |                                                              |
| 5615 | [5615. 使数组互补的最少操作次数](https://leetcode-cn.com/problems/minimum-moves-to-make-array-complementary/) | 好题，对于临界点的差分思想                                   |
| 1109 | [1109. 航班预订统计](https://leetcode-cn.com/problems/corporate-flight-bookings/) | 差分数组模版，diff 是每个航班订购座位数数组的差分数组        |
| 1094 | [1094. 拼车](https://leetcode-cn.com/problems/car-pooling/)  | 构造一个数组表示每一位置车上乘客数，用差分数组来求解，然后判断乘客数是否一直符合capacity要求 |



## 经典模板，思想和技巧

| No.  | <span style="white-space:nowrap;">Template</span>            | description                                                  | case                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | 快速幂 [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/), | 从位运算或者从幂数二分的角度，将幂运算从 $O(n)$ 提升到 $O(log_2n$) | [372. 超级次方](https://leetcode-cn.com/problems/super-pow/) |
| 2    | Manacher                                                     | 用于解决最长回文子串问题，本质是暴力中心扩展的优化           | [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)<br>[214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/) |
| 3    | KMP                                                          | 经典串匹配算法，也是对暴力法的优化加速                       | [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)<br>[214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)<br>[1392. 最长快乐前缀](https://leetcode-cn.com/problems/longest-happy-prefix/) |
| 4    | 原地hash/座位交换法                                          | 在规定不能用额外空间时，原地的交换，如把值1放到下标0处，值2放到下标1处，类似这些方式，可以有奇效 | [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)<br>[442. 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/)<br>[448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)<br>[1497. 检查数组对是否可以被 k 整除](https://leetcode-cn.com/problems/check-if-array-pairs-are-divisible-by-k/) |
| 5    | 归并排序求逆序对思想                                         | 分析归并排序的过程，可以求逆序对，并且在此基础上可以进一步扩展 | [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)<br>[牛客：排队](https://ac.nowcoder.com/acm/contest/6488/C)<br> |
| 6    | 巧妙编码                                                     | 将原数组编码为游程（出现次数）、                             | [696. 计数二进制子串](https://leetcode-cn.com/problems/count-binary-substrings/)<br> |
|      |                                                              |                                                              |                                                              |

## 补充：模拟退火/梯度下降/爬山算法

[小结（转载）](https://www.cnblogs.com/heaad/archive/2010/12/20/1911614.html)

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark     |
| ---- | ------------------------------------------------------------ | ---------- |
| 5463 | [5463. 服务中心的最佳位置](https://leetcode-cn.com/problems/best-position-for-a-service-centre/)（hard） | 梯度下降法 |
|      |                                                              |            |
|      |                                                              |            |
|      |                                                              |            |
|      |                                                              |            |
|      |                                                              |            |

# 说明
* 分类并不严格，菜🐔的自我挣扎
* 小结并不完善，但是参考别人的一定会写清楚