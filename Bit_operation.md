## 位运算

### 位运算使用技巧

1. 基本位运算操作，与、或、异或、非，左右移（其中左右移常用于快速乘除法）。以及简单的用位运算判断奇偶，取负等。

```
&, |, ^, ~, <<, >>
```

2. **位运算交换两个数**，用到偶数个元素异或为0的特性，这里即`a^b^b=a,a^a^b=b`， 代码表示为

```c++
void swap(int& a, int& b) {
		a = a ^ b;
		b = b ^ a;		// b = b ^ a ^ b = a
    a = a ^ b			// a = a ^ b ^ a = b
}
```

3. **求最大的2的整次幂数**，`x & (-x)`

* 从位运算的角度来看，`x & (-x) = x & (~x + 1)`，即将x的二进制数，只留下最右侧的1，其余全为0，因此这样的操作可以用来求不大于x的2的整次幂数
* 这样的操作在树状数组中常用来求lowbit

4. **高低位变换**， 如16位无符号数，要求将其高8位和低8位进行交换，可以如下处理：

```c++
unsigned short x = 34520
x = (x >> 8) | (x << 8)
```

5. **二进制逆序**

```c++
unsigned short a = 34520;

a = ((a & 0xAAAA) >> 1) | ((a & 0x5555) << 1);
a = ((a & 0xCCCC) >> 2) | ((a & 0x3333) << 2);
a = ((a & 0xF0F0) >> 4) | ((a & 0x0F0F) << 4);
a = ((a & 0xFF00) >> 8) | ((a & 0x00FF) << 8);
```

6. **将最右边的1变为0**， `n & (n-1)` 
7. **统计二进制中1的个数** ，使用6中的技巧，可以如下处理：

```c++
int cnt = 0;
while (n) {
	  n = n & (n - 1);
	  cnt++;
}
```



### 状态压缩中常用的技巧

1. 要求集合中不能有两个相邻的元素

```c++
if ((mask >> 1) & mask) continue;
```

2. 在限定必须不取某些元素的前提下枚举子集

```c++
// 限定mask的某些位必须不在集合中
for (int subset = mask; subset >= 0; subset = (subset - 1) & mask)
```

3. 在限定必须取某些元素的前提下枚举子集

```c++
// 限定mask的某些位必须在集合中
for (int subset = mask; subset < (1 << m); subset = (subset + 1) | mask)
```

4. 找出二进制中恰好含有k个1的所有数

```c++
for (int mask = 0; mask < (1 << n);) {
		int tmp = mask & (-mask);
  	mask = (mask + tmp) | (((mask ^ (mask + tmp)) >> 2) / tmp);
}
```







### 参考

- [1] https://www.zhihu.com/question/38206659
- [2] http://graphics.stanford.edu/~seander/bithacks.html#OperationCounting
- [3] http://www.matrix67.com/blog/archives/263

