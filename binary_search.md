## 二分查找

### 1. 二分查找模版

1. 自己最常用的:

```c++
int lo = 0, hi = n;
while (lo < hi) {
  // 这句存在越界可能，可以换成 int mid = lo + (hi - lo) / 2;
  int mid = (lo + hi) >> 1;
  if (nums[mid] >= target) hi = mid;
  else lo = mid + 1;
}
// 判断找到的是否符合条件
return lo;
```

2. 另一种:

```c++
int lo = 0, hi = n - 1;
while (lo <= hi) {
  int mid = lo + (hi - lo) / 2;
  if (nums[mid] > target) hi = mid - 1;
  else if (nums[mid] < target) lo = mid + 1;
  else return mid;
}

```

3. `lower_bound(begin,end,target)`: 在数组要求的范围内查找第一个大于等于target的数字，找到则返回其地址，否则返回end。
4. `upper_bound(begin,end,target)`: 在数组要求的范围内查找第一个大于target的数字，找到则返回其地址，否则返回end。

### 2. 二分查找使用场景举例

1. 在有序数组或者半有序数组中，搜索符合条件的元素下标。处理临界条件和如何设定条件是关键。
2. 二分答案，是对朴素枚举答案法的优化，这类问题，如何判断符合条件是关键，一般需要结合贪心的思想，在较低的时间复杂度下完成判定是否符合条件。

### 3. 分治思想的使用

