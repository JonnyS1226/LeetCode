## Leetcode

### 位运算	

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
| 717         | [717. 1比特与2比特字符](https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/) | 总是以0结尾，所以只要看最后一个0和倒数第二个0之间1的个数(即连续的1个数)，为奇数就返回false，偶数是true，可用异或计数 |
| 201         | [201. 数字范围按位与](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/) | 使用位移，找m和n的二进制数的公共前缀                         |
|             |                                                              |                                                              |

### 二分查找/分治减治思想

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

### 链表类型题

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



### 动态规划

[解法与总结]()

| No.   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 55    | [跳跃游戏](https://leetcode-cn.com/problems/jump-game/)      | dp[i]表示是否能跳到下标为i的元素，可以使用动态规划解决       |
| 983   | [最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/) | dp[i]表示前i天买票旅行的最低消费，dp[i]由dp[i-1],dp[i-7],dp[i-30]决定 |
| 70    | [爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)  | dp[i]表示爬i层楼的方法数，dp[i] = dp[i-1] + dp[i-2];         |
| 46    | [面试题46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/) | 类似70                                                       |
| 322   | [零钱兑换](https://leetcode-cn.com/problems/coin-change/)    | dp[i]表示凑总金额i所需最少硬币数，dp[i] = min(dp[i], dp[i-coin]+1); |
| 518   | [零钱兑换Ⅱ](https://leetcode-cn.com/problems/coin-change-2/) | dp[i]表示凑总金额i的方法数，dp[i] = $ \sum$ dp[i-coin];      |
| 72    | [编辑距离](https://leetcode-cn.com/problems/edit-distance/)  | 用 dp[i] [j] 表示 `A` 的前 `i` 个字母和 `B` 的前 `j` 个字母之间的编辑距离. |
| 1340  | [跳跃游戏Ⅴ](https://leetcode-cn.com/problems/jump-game-v/)   | dp[i]表示某一点i可以到达的最大点个数，dp[i] = 1 + max(max(dp[i-d]...dp[i-1]), max(dp[i+1, i+d]))，其中要排除位置高度大于i位置的部分 |
| 221   | [最大正方形](https://leetcode-cn.com/problems/maximal-square/) | dp(*i*,*j*) 表示以 (i, j)为右下角，且只包含1的正方形的边长最大值，dp(i, j) = min(dp(i-1, j), dp(i, j-1), dp(i-1, j-1)) + 1 |
| 1277  | [统计全为1的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/) | 同221                                                        |
| 53    | [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/) | *f*(*i*) = max{*f*(*i*−1)+*ai* , *ai*}                       |
| 152   | [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/) | 类似最大连续和，但是要同时维护最大乘积dp和最小乘积dp（应对负数乘积为正的情况） |
| 1567  | [1567. 乘积为正数的最长子数组长度](https://leetcode-cn.com/problems/maximum-length-of-subarray-with-positive-product/) | 同152思路，需要维护以i结尾乘积为正和乘积为负的两个dp         |
| 5521  | [5521. 矩阵的最大非负积](https://leetcode-cn.com/problems/maximum-non-negative-product-in-a-matrix/) | 同152，只不过是二维dp，维护最大值和最小值，因为最小值可能翻身成为最大值 |
| 343   | [343. Integer Break](https://leetcode-cn.com/problems/integer-break/) | dpi 表示n的题设下，分割整数后的乘积最大值                    |
| 746   | [746. Min Cost Climbing Stairs](https://leetcode-cn.com/problems/min-cost-climbing-stairs/) | dpi表示选择了i所需要的最小cost                               |
| 96    | [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/) | 讨论时分析每个数为根节点的情况，可以递推出：`dp[i] = dp[0]*dp[i-1] + dp[1]*dp[i-2] + ... + dp[i-1]*dp[0]` |
|       |                                                              |                                                              |
| 198   | [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/) | dp[i]表示前i+1个房屋最大偷窃金额，dp[i] = max(dp[i-1], dp[i-2] + nums[i]) |
| 213   | [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/) | 类似198，可以将环形划分解成[0:n-2]和[1:n-1]两个线性的dp，使用同198的递推式分段解决，求最大值 |
| 740   | [740. 删除与获得点数](https://leetcode-cn.com/problems/delete-and-earn/) | 可以转换为198问题                                            |
| 410   | [410. 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/) |                                                              |
| 516   | [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/) | 子序列问题的动态规划，dp(i)(j)表示s[i]-s[j]区间内最长回文子序列长度，注意区间dp要斜着打表或者反着打表。 <br>也可以转换为求逆字符串，再求最长公共子序列 |
| 300   | [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) | dp[i]=max{dp[j]}+1，if num[i] > num[j], 其中i>j，其中dp[i]表示前i个得最长递增序列长度，且nums[i]必须选择 |
| 1143  | [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/) | dp[i] [j]代表text1前i个字符和text2前j个字符的最长公共子序列  |
|       |                                                              |                                                              |
| 718   | [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/) | dp[i] [j]表示A前i个和B前j个 最长公共子数组长度(且要求取到公共子数组必须以i和j结尾) |
| 97    | [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)（hard） |                                                              |
| 5419  | [5419. 两个子序列的最大点积](https://leetcode-cn.com/problems/max-dot-product-of-two-subsequences/) | 类似72，1143，  两个数组的dp， $O(n^2)$                      |
| 139   | [139. 单词拆分](https://leetcode-cn.com/problems/word-break/) | 动态规划, dp[i]表示前i个字符是否能被分隔                     |
| 140   | [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/) | 动态规划 + 回溯                                              |
| 1139  | [1139. 最大的以 1 为边界的正方形](https://leetcode-cn.com/problems/largest-1-bordered-square/) |                                                              |
| 85    | [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/) |                                                              |
| 392   | [392. 判断子序列](https://leetcode-cn.com/problems/is-subsequence/)（进阶挑战） |                                                              |
|       |                                                              |                                                              |
|       |                                                              |                                                              |
| 64    | [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) | $dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]$        |
| 10    | [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)（hard） | 用dp i  j 表示 s的前 i 个字符与 p中的前 j 个字符是否能够匹配。 |
| 5411  | [5411. 摘樱桃 II](https://leetcode-cn.com/problems/cherry-pickup-ii/)（hard） | 三层dp dp[ijk]表示第i行，机器人1在j列，机器人2在k列的最大樱桃数 |
| 5431  | [5431. 给房子涂色 III](https://leetcode-cn.com/problems/paint-house-iii/)（hard） | 三维dp，dp i j k 表示第i个房子，涂了第j个颜色，且形成了k个社区的最小花费 |
| 837   | [837. 新21点](https://leetcode-cn.com/problems/new-21-game/) | dp[i]表示当前和为i（i < K）时获胜的概率， dp[i] = 摸j点的概率(1/w) 乘以 摸完之后成功的概率(dp[]i+j])，并遍历j求和 |
| 1494  | [1494. 并行课程 II](https://leetcode-cn.com/problems/parallel-courses-ii/)（hard） |                                                              |
| 32    | [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)（hard） | 有多种方法，栈，dp，双向扫描。此处用dp，dp[i]表示以i位置结尾的最长有小括号子串长度。更新时，如果当前位置是(，显然长度为0，如果当前位置是右括号，那么要尝试找到与之对应左括号，需要判断i-dp[i-1]-1的位置是否是左括号，如果是：dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]  (还需要看匹配位置之前有没有有小括号) |
| 44    | [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)（hard） | dp[i][j]表示s前i个和p前j个是否能匹配                         |
| 174   | [174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)（hard） | 二维逆序dp                                                   |
| 309   | [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | dp`[i][0]`表示持有股票；dp`[i][1]`表示不持有股票，处于冷冻期;dp`[i][2]`表示不持有股票，不处于冷冻期。这里的「处于冷冻期」指的是在第 i 天结束之后的状态 |
| 877   | [877. 石子游戏](https://leetcode-cn.com/problems/stone-game/) | `dp[i][j]`表示从i到j序列，先手和后手的差值；递推时分析 如果选开头堆如何更新，选末尾堆如何更新即可推出递推式 |
| 1140  | [1140. 石子游戏 II](https://leetcode-cn.com/problems/stone-game-ii/) |                                                              |
| 1406  | [1406. 石子游戏 III](https://leetcode-cn.com/problems/stone-game-iii/)（hard） |                                                              |
| 5447  | [5447. 石子游戏 IV](https://leetcode-cn.com/problems/stone-game-iv/)（hard） | 博弈dp，dp[i] 表示对于数i是否能先手赢                        |
| 1025  | [1025. 除数博弈](https://leetcode-cn.com/problems/divisor-game/) | 同石子游戏Ⅳ                                                  |
| 312   | [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)（hard） | 本质和矩阵链乘法 一样的dp；`dp[i][j] = v[i] * v[k] * [j] + dp[i][k] + dp[k][j];` |
| LCP13 | [LCP 13. 寻宝](https://leetcode-cn.com/problems/xun-bao/)（hard） |                                                              |
| 5486  | [5486. 切棍子的最小成本](https://leetcode-cn.com/problems/minimum-cost-to-cut-a-stick/)（hard） | 倒过来想，区间dp，`dp[i][j] `表示区间cuts[i]到cuts[j]的距离的合并的最小代价 |
| 1000  | [1000. 合并石头的最低成本](https://leetcode-cn.com/problems/minimum-cost-to-merge-stones/) |                                                              |
| 546   | [546. 移除盒子](https://leetcode-cn.com/problems/remove-boxes/) |                                                              |
| 1024  | [1024. 视频拼接](https://leetcode-cn.com/problems/video-stitching/) | 动态规划问题，`dp[i]` 表示将区间`[0,i)`覆盖所需要的最少子区间的数量 |
| 845   | [845. 数组中的最长山脉](https://leetcode-cn.com/problems/longest-mountain-in-array/) | 两遍扫描，类似dp思想，设定left和right数组表示向左向右能扩展的最大距离 |
|       |                                                              |                                                              |







### 贪心算法

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 45   | [跳跃游戏Ⅱ](https://leetcode-cn.com/problems/jump-game-ii/submissions/) | 贪心思想：每次选择能到达的最远路径，并且方法数+1，在此基础上再跳下一次 |
| 55   | [跳跃游戏](https://leetcode-cn.com/problems/jump-game/)      | 除了使用dp，也可以使用贪心思想：也是尽可能选择最远的跳，维护一个当前可跳最远距离 |
| 1029 | [两地调度](https://leetcode-cn.com/problems/two-city-scheduling/) | 首先将这 2N 个人全都安排飞往 BB 市，再选出 N 个人改变它们的行程，让他们飞往 AA 市。如果选择改变一个人的行程，那么公司将会额外付出 price_A - price_B 的费用，所以只要这部分最小即可 |
| 991  | [991. 坏了的计算器](https://leetcode-cn.com/problems/broken-calculator/) | 逆向思维，y+1或者y/2，贪心让y尽可能/2                        |
|      |                                                              |                                                              |



### 二叉树及树专题

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 98   | [验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/) | 中序遍历满足递增才是二叉搜索树                               |
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
| 1028 | [1028. 从先序遍历还原二叉树](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/)（hard） | 通过-确定层级关系，控制出入栈                                |
| 101  | [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/) | 递归：相当于两个指针，分别比较左右子树；迭代：一次从队列取出两个 比较值是否相等或者是否只有一个为空 |
|      |                                                              |                                                              |
| 124  | [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)（hard） | 递归 dfs                                                     |
| 543  | [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/) | 类似124，dfs                                                 |
| 99   | [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)（hard） |                                                              |
| 95   | [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/) | 考虑枚举[start,end]中的值 i 为当前二叉搜索树的根，再对划分出的两部分递归求解，最后左子树右子树各选择一颗接上去即可 |
| 110  | [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/) |                                                              |
|      | 完全二叉树                                                   |                                                              |
| 404  | [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/) | 先序遍历，递归或迭代，找到**满足左叶子条件**就记录结果       |
| 114  | [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/) | 按根右左的遍历，通过右指针组成链表                           |
| 430  | [430. 扁平化多级双向链表](https://leetcode-cn.com/problems/flatten-a-multilevel-doubly-linked-list/) | 同114，把child看成左，next看成右即可                         |
| 235  | [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | 利用二叉搜索树的性质求LCA                                    |
| 236  | [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) |                                                              |



### DFS/BFS/四种最短路算法(dijkstra,bellman-ford,spfa,floyd)

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 130  | [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) | 从边界开始dfs(或BFS)，找到不被包围的，其他就是被包围的       |
| 417  | [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/) | 从两个边界开始 两次dfs（或BFS），都遍历到的地方就是结果集一部分 |
| 5426 | [5426. 重新规划路线](https://leetcode-cn.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/) | $O(n^2)$超时，可以建两种顺序的图，BFS，也可以并查集          |
| 127  | [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/) |                                                              |
| 785  | [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/) | 经典染色法，dfs或者bfs                                       |
| 886  | [886. 可能的二分法](https://leetcode-cn.com/problems/possible-bipartition/) |                                                              |
| 1349 | [1349. 参加考试的最大学生数](https://leetcode-cn.com/problems/maximum-students-taking-exam/)（hard） |                                                              |
| 126  | [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)（hard） | bfs层级遍历，并标层级号， 然后dfs从结束点回溯，找到路径存储。 |
| 433  | [433. 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/) |                                                              |
| 5211 | [5211. 概率最大的路径](https://leetcode-cn.com/problems/path-with-maximum-probability/) | 先要证明这种0-1之间的乘法最长路 可以用 dijkstra/bellman-ford/spfa求最短路的方法求 （反证法）。然后就可用堆优化的dijkstra，bellman-ford，或者spfa（这题没卡spfa） |
| 5410 | [5410. 课程安排 IV](https://leetcode-cn.com/problems/course-schedule-iv/) | 可以用floyd算法判断两点是否有通路，或者使用并查集            |
| 733  | [733. 图像渲染](https://leetcode-cn.com/problems/flood-fill/) | 题意就是类似油漆桶工具，存储原始color，然后使用bfs或者dfs    |
| 5490 | [5490. 吃掉 N 个橘子的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-eat-n-oranges/) | 可以用带map缓存的bfs或双向bfs                                |
| 529  | [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/) | dfs                                                          |
| 679  | [679. 24 点游戏](https://leetcode-cn.com/problems/24-game/)（hard） | 纯暴力，4种运算，4个数， dfs回溯                             |
| 5482 | [5482. 二维网格图中探测环](https://leetcode-cn.com/problems/detect-cycles-in-2d-grid/)（hard） | 带前置节点的dfs或者bfs                                       |
| 841  | [841. 钥匙和房间](https://leetcode-cn.com/problems/keys-and-rooms/) | 建图dfs， 或者拓扑排序判环                                   |
| 78   | [78. 子集](https://leetcode-cn.com/problems/subsets/)        | 简单dfs，或者直接遍历，遇到一个数就把所有子集加上该数组成新的子集 |
| 1034 | [1034. 边框着色](https://leetcode-cn.com/problems/coloring-a-border/) | DFS求连通分量的边界（通过控制属于该分量返回1，不属于返回0来实现） |
|      |                                                              |                                                              |
|      |                                                              |                                                              |
|      |                                                              |                                                              |

### 拓扑排序

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 207  | [207. 课程表](https://leetcode-cn.com/problems/course-schedule/) | 拓扑排序标准模板，判断是否是DAG                              |
| 210  | [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/) | 带结果集的拓扑排序                                           |
| 329  | [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/) |                                                              |
| 1203 | [1203. 项目管理](https://leetcode-cn.com/problems/sort-items-by-groups-respecting-dependencies/) |                                                              |
| 851  | [851. 喧闹和富有](https://leetcode-cn.com/problems/loud-and-rich/) | 富->穷，构建有向图，拓扑排序，每次遍历邻接点时找所有点中最安静的 |
|      |                                                              |                                                              |



### 并查集

[解法与总结]()

| No    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 5426  | [5426. 重新规划路线](https://leetcode-cn.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/) | 此处使用并查集                                               |
| 5410  | [5410. 课程安排 IV](https://leetcode-cn.com/problems/course-schedule-iv/) | 此处使用并查集                                               |
| 130   | [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) |                                                              |
| 959   | [959. 由斜杠划分区域](https://leetcode-cn.com/problems/regions-cut-by-slashes/) |                                                              |
| 17.07 | [面试题 17.07. 婴儿名字](https://leetcode-cn.com/problems/baby-names-lcci/) |                                                              |
| 128   | [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)（hard） | 维护一个map(size)，将num，num-1，num+1这样的merge，并根据连通分量的size不断更新ans |
| 990   | [990. Satisfiability of Equality Equations](https://leetcode-cn.com/problems/satisfiability-of-equality-equations/) | 等号则连通，不等号则判断左右两个是否在一个连通里，如果在，则返回false |
| 785   | [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/) |                                                              |
| 130   | [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/) |                                                              |
| 5497  | [5497. 查找大小为 M 的最新分组](https://leetcode-cn.com/problems/find-latest-group-of-size-m/) |                                                              |
|       |                                                              |                                                              |
|       |                                                              |                                                              |


### 回溯剪枝

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 46   | [46. 全排列](https://leetcode-cn.com/problems/permutations/) | 回溯剪枝，可以用交换法，也可以纯回溯                         |
| 47   | [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/) | 回溯剪枝，判断好去重条件                                     |
| 784  | [784. 字母大小写全排列](https://leetcode-cn.com/problems/letter-case-permutation/) | 回溯剪枝，但结果集不需要等到搜索到叶子才添加，而是每一个节点更改都要添加 |
| 93   | [93. 复原IP地址](https://leetcode-cn.com/problems/restore-ip-addresses/) |                                                              |
| 140  | [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/) | 回溯剪枝                                                     |
| 491  | [491. 递增子序列](https://leetcode-cn.com/problems/increasing-subsequences/) | 回溯剪枝                                                     |
| 37   | [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)（hard） | 回溯剪枝                                                     |
| 5520 | [5520. 拆分字符串使唯一子字符串的数目最大](https://leetcode-cn.com/problems/split-a-string-into-the-max-number-of-unique-substrings/) | dfs+回溯                                                     |
| 698  | [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/) | 回溯，构建k个和为target的集合，在每个集合构建时用回溯        |
| 473  | [473. 火柴拼正方形](https://leetcode-cn.com/problems/matchsticks-to-square/) | No.698 的k=4的特殊情况                                       |
|      |                                                              |                                                              |
|      |                                                              |                                                              |




### 栈/单调栈问题

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |      |
| :--- | ------------------------------------------------------------ | :----------------------------------------------------------- | ---- |
| 42   | [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)（hard） | 典型单调递减栈题目                                           |      |
| 84   | [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/) | 求以每个矩形高度为框出来高度的最大矩形面积（这一步通过单调栈，对于某个矩形高度，找向左延伸第一个小于它的，向右延伸第一个小于它的，然后求面积），再在里面取最大的。 |      |
| 496  | [496. 下一个更大元素 I](https://leetcode-cn.com/problems/next-greater-element-i/) | 构造一个单调递减栈，找到一个比栈顶大的就出栈，这个元素就是出栈元素后面第一个比它大的 |      |
| 503  | [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/) | 构造一个单调递减栈，与496区别在于循环判断，如[4321]相当于用496的方法计算[43214321] |      |
| 739  | [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/) | 同496，维护一个单调递减栈                                    |      |
| 239  | [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/) |                                                              |      |
| 901  | [901. 股票价格跨度](https://leetcode-cn.com/problems/online-stock-span/) | 与496，739一样的思路                                         |      |
| 402  | [402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/) | 贪心的想法+单调递增栈                                        |      |
| 33   | [面试题33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/) | 递归分治O($n^2$)，*可以用单调栈实现O(n)                      |      |
| 5420 | [5420. Final Prices With a Special Discount in a Shop](https://leetcode-cn.com/problems/final-prices-with-a-special-discount-in-a-shop/) | 单调栈的简单应用                                             |      |
| 394  | [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/) | 维护数字栈和字符栈，碰到[入栈，碰到]出栈                     |      |
| 1499 | [1499. 满足不等式的最大值](https://leetcode-cn.com/problems/max-value-of-equation/) |                                                              |      |
| 5459 | [5459. 形成目标数组的子数组最少增加次数](https://leetcode-cn.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/)（hard） | 可以利用单调栈思想：考虑每个元素左侧相邻元素的贡献值，但不同于常规单调栈，不需要所有出栈都计算 |      |
| 1544 | [1544. 整理字符串](https://leetcode-cn.com/problems/make-the-string-great/) | 用数组模拟栈，或者直接用栈                                   |      |

### 滑动窗口/单调队列

[解法与总结]()

| No.  | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1052 | [1052. 爱生气的书店老板](https://leetcode-cn.com/problems/grumpy-bookstore-owner/) | 可以维护一个大小为X的滑动窗口，O(xn)                         |
| 209  | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 典型滑窗，也可以用前缀和+二分法                              |
| 76   | [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)（hard） | 滑窗，判断窗口内是否满足要求，若不满足则扩大窗口，满足则尝试缩小 |
| 424  | [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/) | 滑动窗口，窗口内条件right - left + 1 - tmpMaxCnt <= k        |
| 1456 | [1456. 定长子串中元音的最大数目](https://leetcode-cn.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/) | map+滑动窗口                                                 |
| 5423 | [5423. Find Two Non-overlapping Sub-arrays Each With Target Sum](https://leetcode-cn.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/) | 从后往前的滑动窗口+dp                                        |
| 239  | [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)（hard） |                                                              |
| 1499 | [1499. 满足不等式的最大值](https://leetcode-cn.com/problems/max-value-of-equation/) | 即求 `max(yi + yj + xj - xi) = max(xj + yj) + max(yi - xi), i < j`，转换后就变成了239题，单调队列求滑动窗口内（区间内）最大值 |
| 632  | [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)（hard） |                                                              |

### 快慢指针/双指针

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
| 26   | [26. Remove Duplicates from Sorted Array](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/) | 快慢指针原地修改题                                           |
| 27   | [27. Remove Element](https://leetcode-cn.com/problems/remove-element/) | 快慢指针原地修改题                                           |
| 283  | [283. Move Zeroes](https://leetcode-cn.com/problems/move-zeroes/) | 快慢指针原地修改题                                           |
| 16   | [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/) | 排序+三指针                                                  |
| 1537 | [1537. 最大得分](https://leetcode-cn.com/problems/get-the-maximum-score/) | 分段求值，相当于记录下岔路口时的 最优解。可以dp，也可以优化为双指针 |
| 143  | [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/) | 快慢指针找中点(向前取)，分出两段，后面的插入到前面           |
| 763  | [763. 划分字母区间](https://leetcode-cn.com/problems/partition-labels/) | map存取最后出现的位置+双指针循环更新                         |
| 56   | [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/) | 和57一样，先排序，再遍历。根据指针分析可以判断sort二维数组时，默认按每行第一个进行递增排序 |
|      |                                                              |                                                              |

### 线段树/树状数组

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark |
| ---- | ------------------------------------------------------------ | ------ |
| 315  | [315. 计算右侧小于当前元素的个数](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)（hard） |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |


### hash/set/桶/计数/堆等容器使用
[解法与总结]()
| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 349  | [349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/) | 一个set存放第一个数组元素，一个set用于验证第二个数组中交集部分 |
| 350  | [350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/) | hash表，或者排序双指针                                       |
| 牛客 | [排队](https://ac.nowcoder.com/acm/contest/6488/C)           | 优先队列实现小顶堆+归并排序/树状数组求逆序对                 |
| 632  | [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/) | k指针+最小堆，维护最大值和最小值，取最优区间（或者分别维护最大最小堆也可） |
| 169  | [169. 多数元素](https://leetcode-cn.com/problems/majority-element/) | 摩尔投票法，一个候选人                                       |
| 229  | [229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/) | 摩尔投票法，两个候选人                                       |
| 1287 | [1287. 有序数组中出现次数超过25%的元素](https://leetcode-cn.com/problems/element-appearing-more-than-25-in-sorted-array/) | 可以用摩尔投票法，找三个候选人                               |
| 1365 | [1365. 有多少小于当前数字的数字](https://leetcode-cn.com/problems/how-many-numbers-are-smaller-than-the-current-number/) | hash存放比当前数小的数个数，从小到大累加来算                 |


### 字符串独立专题（前缀、后缀、字典树及其它技巧）

[解法与总结]()
| No    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                  |
| ----- | ------------------------------------------------------------ | ----------------------- |
| 459   | [459. 重复的子字符串](https://leetcode-cn.com/problems/repeated-substring-pattern/) | 重复子串的特性          |
| 1044  | [1044. 最长重复子串](https://leetcode-cn.com/problems/longest-duplicate-substring/) |                         |
| 67    | [67. 二进制求和](https://leetcode-cn.com/problems/add-binary/) | 模拟字符串n进制求和即可 |
| 989   | [989. 数组形式的整数加法](https://leetcode-cn.com/problems/add-to-array-form-of-integer/) | 大数加法                |
| 43    | [43. 字符串相乘](https://leetcode-cn.com/problems/multiply-strings/) | 大数乘法                |
| 17.13 | [面试题 17.13. 恢复空格](https://leetcode-cn.com/problems/re-space-lcci/) |                         |
| 336   | [336. 回文对](https://leetcode-cn.com/problems/palindrome-pairs/)（hard） |                         |



### 数学题/细节分类讨论题

[解法与总结]()

| No    | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 961   | [961. 重复 N 次的元素](https://leetcode-cn.com/problems/n-repeated-element-in-size-2n-array/) | 推理：相当于N个不相同的元素，插入N个相同的元素，一定存在连续3个元素中有2个相同的值，这个值就是结果 |
| 189   | [189. Rotate Array](https://leetcode-cn.com/problems/rotate-array/) |                                                              |
| 16.18 | [面试题 16.18. 模式匹配](https://leetcode-cn.com/problems/pattern-matching-lcci/) | 暴力枚举，+ 严格分类讨论                                     |
|       |                                                              |                                                              |
|       |                                                              |                                                              |
|       |                                                              |                                                              |

### 回文系列

[解法与总结]()

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
|      |                                                              |                                                              |



### 前缀和/积

[解法与总结]()

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 560  | [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/) | 前缀和+map                                                   |
| 5471 | [5471. 和为目标值的最大数目不重叠非空子数组数目](https://leetcode-cn.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/) | 560的变形，要清空前缀和和map                                 |
| 1248 | [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/) | 前缀和+map, pre 存放前n项奇数的个数,map key->pre  value->出现次数 |
| 554  | [554. 砖墙](https://leetcode-cn.com/problems/brick-wall/)    | 前缀和+map, map key->preNum, value->出现次数                 |
| 1423 | [1423. 可获得的最大点数](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/) | 先计算前缀和，然后求left(0-i)和right(len-k+i+1, len-1)的最大值 |
| 238  | [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/) | 两次扫描，前缀积和后缀积                                     |
| 209  | [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | 前缀和构造有序数组，然后使用二分                             |
| 974  | [974. 和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/) | 状态压缩+前缀和                                              |
| 1371 | [1371. 每个元音包含偶数次的最长子字符串](https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/) | 状态压缩+前缀和                                              |
| 5485 | [5485. 找出最长的超赞子字符串](https://leetcode-cn.com/problems/find-longest-awesome-substring/)（hard） | 状态压缩+前缀和                                              |



### 经典模板，思想和技巧

| No.  | <span style="white-space:nowrap;">Template</span>            | description                                                  | case                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | 快速幂 [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/), | 从位运算或者从幂数二分的角度，将幂运算从 $O(n)$ 提升到 $O(log_2n$) | [372. 超级次方](https://leetcode-cn.com/problems/super-pow/) |
| 2    | Manacher                                                     | 用于解决最长回文子串问题，本质是暴力中心扩展的优化           | [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)<br>[214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/) |
| 3    | KMP                                                          | 经典串匹配算法，也是对暴力法的优化加速                       | [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)<br>[214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/) |
| 4    | 原地hash/座位交换法                                          | 在规定不能用额外空间时，原地的交换，如把值1放到下标0处，值2放到下标1处，类似这些方式，可以有奇效 | [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)<br>[442. 数组中重复的数据](https://leetcode-cn.com/problems/find-all-duplicates-in-an-array/)<br>[448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)<br>[1497. 检查数组对是否可以被 k 整除](https://leetcode-cn.com/problems/check-if-array-pairs-are-divisible-by-k/) |
| 5    | 归并排序求逆序对思想                                         | 分析归并排序的过程，可以求逆序对，并且在此基础上可以进一步扩展 | [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)<br>[牛客：排队](https://ac.nowcoder.com/acm/contest/6488/C)<br> |
| 6    | 巧妙编码                                                     | 将原数组编码为游程（出现次数）、                             | [696. 计数二进制子串](https://leetcode-cn.com/problems/count-binary-substrings/)<br> |
|      |                                                              |                                                              |                                                              |

### 补充：模拟退火/梯度下降/爬山算法

[解法与总结（转载）](https://www.cnblogs.com/heaad/archive/2010/12/20/1911614.html)

| No   | <span style="white-space:nowrap;">Title&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;</span> | Remark |
| ---- | ------------------------------------------------------------ | ------ |
| 5463 | [5463. 服务中心的最佳位置](https://leetcode-cn.com/problems/best-position-for-a-service-centre/)（hard） |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |
|      |                                                              |        |