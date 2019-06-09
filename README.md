# numpy 学习

<https://www.numpy.org/>

NumPy是使用Python进行科学计算的基础包。它包含其他内容

+ 一个强大的N维数组对象
+ 复杂的（广播）功能
+ 用于集成C / C ++和Fortran代码的工具
+ 有用的线性代数，傅里叶变换和随机数功能

除了明显的科学用途外，NumPy还可以用作通用数据的高效多维容器。可以定义任意数据类型。这使NumPy能够无缝快速地与各种数据库集成。

NumPy根据BSD许可证授权，只需很少的限制即可重复使用。

## 基础

NumPy的主要对象是同构多维数组。它是一个元素表（通常是数字），都是相同的类型，由正整数元组索引。在NumPy维度中称为轴(axes)。

例如，3D空间[1,2,1]中的点的坐标具有一个轴。该轴有3个元素，所以我们说它的长度为3.在下图所示的例子中，数组有2个轴。第一轴的长度为2，第二轴的长度为3。

```python
[[ 1., 0., 0.],
 [ 0., 1., 2.]]
```

NumPy的数组类称为`ndarray`。它也被别名数组所知。请注意，`numpy.array`与标准Python库类`array.array`不同，后者仅处理一维数组并提供较少的功能。 `ndarray`对象的更重要的属性是：

**ndarray.ndim**
数组的轴数（尺寸）。
**ndarray.shape**
数组的大小。这是一个整数元组，表示每个维度中数组的大小。对于具有n行和m列的矩阵，`shape`将为`(n，m)`。因此，`shape`元组的长度是轴的数量`ndim`。
**ndarray.size**
数组的元素总数。这等于`shape`元素的乘积。
**ndarray.dtype**
描述数组中元素类型的对象。可以使用标准Python类型创建或指定`dtype`。此外，NumPy还提供自己的类型。 `numpy.int32`，`numpy.int16`和`numpy.float64`就是一些例子。
**ndarray.itemsize**
数组中每个元素的大小（以字节为单位）。例如，`float64`类型的元素数组具有`itemsize` 8（= 64/8），而`complex32`类型之一具有`itemsize` 4（= 32/8）。它相当于`ndarray.dtype.itemsize`。
**ndarray.data**
包含数组实际元素的缓冲区。通常，我们不需要使用此属性，因为我们将使用索引工具访问数组中的元素。

## 例子

```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<type 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<type 'numpy.ndarray'>
```

## 生成数组

```python
>>> import numpy as np
# 一维数组
>>> a = np.array([2,3,4])
>>> a
array([2, 3, 4])

# 多维数组
>>> b = np.array([(1.5,2,3), (4,5,6)])
>>> b
array([[ 1.5,  2. ,  3. ],
       [ 4. ,  5. ,  6. ]])

# 指定数组元素的类型
>>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[ 1.+0.j,  2.+0.j],
       [ 3.+0.j,  4.+0.j]])

# 生成特定的数组
>>> np.zeros( (3,4) )
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
>>> np.ones( (2,3,4), dtype=np.int16 )  # dtype can also be specified
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) ) # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])

# 根据排列生成数组
>>> np.arange( 10, 30, 5 ) # 参数：起始值，最大值，间隔大小
array([10, 15, 20, 25])

# arange 对浮点数支持不好，生成的数据可能不准确，这时需要用到 linspace

>>> np.linspace( 0, 2, 9 )  # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
```

## 基本操作

数组上的算术运算符应用于元素。创建一个新数组并填充结果。

```python
>>> a = np.array( [20,30,40,50] )
>>> b = np.arange( 4 )
>>> b
array([0, 1, 2, 3])
>>> c = a-b
>>> c
array([20, 29, 38, 47])
>>> b**2
array([0, 1, 4, 9])
>>> 10*np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a<35
array([ True, True, False, False])
```

与许多矩阵语言不同，产品运算符*在NumPy数组中以元素方式运行。矩阵乘积可以使用@运算符（在python> = 3.5中）或点函数或方法执行：

```python
>>> A = np.array( [[1,1],
...             [0,1]] )
>>> B = np.array( [[2,0],
...             [3,4]] )
>>> A * B                       # elementwise product
array([[2, 0],
       [0, 4]])
>>> A @ B                       # matrix product
array([[5, 4],
       [3, 4]])
>>> A.dot(B)                    # another matrix product
array([[5, 4],
       [3, 4]])
```

某些操作（例如+ =和* =）用于修改现有数组而不是创建新数组。

```python
>>> a = np.ones((2,3), dtype=int)
>>> b = np.random.random((2,3))
>>> a *= 3
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> b += a
>>> b
array([[ 3.417022  ,  3.72032449,  3.00011437],
       [ 3.30233257,  3.14675589,  3.09233859]])
>>> a += b                  # b is not automatically converted to integer type
Traceback (most recent call last):
  ...
TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

## 通用功能

NumPy提供熟悉的数学函数，例如`sin`，`cos`和`exp`(自然指数)。在NumPy中，这些被称为“通用函数”（`ufunc`）。在NumPy中，这些函数在数组上以元素方式运行，产生一个数组作为输出。

```python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B) # 自然对数底e 的幂
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B) # 开方
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])
```

## 索引，切片和迭代

一维数组可以被索引，切片和迭代，就像列表和其他Python序列一样。

```python
>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
>>> a[2]
8
>>> a[2:5]
array([ 8, 27, 64])
>>> a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
>>> a
array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,   729])
>>> a[ : :-1]                                 # reversed a
array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000])
>>> for i in a:
...     print(i**(1/3.))
...
nan
1.0
nan
3.0
nan
5.0
6.0
7.0
8.0
9.0
```

多维数组每个轴可以有一个索引。这些索引以逗号​​分隔的元组给出：

```python
>>> def f(x,y):
...     return 10*x+y
...
>>> b = np.fromfunction(f,(5,4),dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2,3]
23
>>> b[0:5, 1]                       # each row in the second column of b
array([ 1, 11, 21, 31, 41])
>>> b[ : ,1]                        # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
>>> b[1:3, : ]                      # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
```

当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片

```python
>>> b[-1]       # the last row. Equivalent to b[-1,:]
array([40, 41, 42, 43])
```

b [i]中括号内的表达式被视为i，后跟多个实例：根据需要表示剩余的轴。 NumPy还允许您使用点作为b [i，...]来写这个。

点（...）表示生成完整索引元组所需的冒号。例如，如果x是一个包含5个轴的数组，那么

+ x[1,2,...] is equivalent to x[1,2,:,:,:],
+ x[...,3] to x[:,:,:,:,3] and
+ x[4,...,5,:] to x[4,:,:,5,:].

```python
>>> c = np.array( [[[  0,  1,  2], # a 3D array (two stacked 2D arrays)
...                 [ 10, 12, 13]],
...                [[100,101,102],
...                 [110,112,113]]])
>>> c.shape
(2, 2, 3)
>>> c[1,...]                    # same as c[1,:,:] or c[1]
array([[100, 101, 102],
       [110, 112, 113]])
>>> c[...,2]                    # same as c[:,:,2]
array([[  2,  13],
       [102, 113]])
```

对多维数组进行迭代是针对第一个轴完成的：

```python
>>> for row in b:
...     print(row)
...
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```

However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:

```python
>>> for element in b.flat:
...     print(element)
...
0
1
2
3
```

## 形状操纵

### 更改数组的形状

数组的形状由沿每个轴的元素数量给出：

```python
>>> a = np.floor(10*np.random.random((3,4)))
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)
```

可以使用各种命令更改阵列的形状。请注意，以下三个命令都返回已修改的数组，但不更改原始数组：

```python
>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
>>> a.reshape(6,2)  # returns the array with a modified shape
array([[ 2.,  8.],
       [ 0.,  6.],
       [ 4.,  5.],
       [ 1.,  1.],
       [ 8.,  9.],
       [ 3.,  6.]])
>>> a.T  # returns the array, transposed
array([[ 2.,  4.,  8.],
       [ 8.,  5.,  9.],
       [ 0.,  1.,  3.],
       [ 6.,  1.,  6.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

由ravel（）产生的数组中元素的顺序通常是“C风格”，也就是说，最右边的索引“变化最快”，因此[0,0]之后的元素是[0,1] 。如果将数组重新整形为其他形状，则该数组将被视为“C风格”。 NumPy通常创建按此顺序存储的数组，因此ravel（）通常不需要复制其参数，但如果数组是通过获取另一个数组的切片或使用不常见的选项创建的，则可能需要复制它。还可以使用可选参数指示函数ravel（）和reshape（），以使用FORTRAN样式的数组，其中最左边的索引变化最快。

reshape函数以修改的形状返回其参数，而ndarray.resize方法修改数组本身：

```python
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.resize((2,6))
>>> a
array([[ 2.,  8.,  0.,  6.,  4.,  5.],
       [ 1.,  1.,  8.,  9.,  3.,  6.]])
```

如果在重新整形操作中将尺寸指定为-1，则会自动计算其他尺寸：

```python
>>> a.reshape(3,-1)
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
```

### 堆叠在一起的不同阵列

几个阵列可以沿不同的轴堆叠在一起：

```python
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
```

函数column_stack将1D数组作为列堆叠到2D数组中。它相当于仅针对2D数组的hstack：

```python
>>> from numpy import newaxis
>>> np.column_stack((a,b))     # with 2D arrays
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
>>> a = np.array([4.,2.])
>>> b = np.array([3.,8.])
>>> np.column_stack((a,b))     # returns a 2D array
array([[ 4., 3.],
       [ 2., 8.]])
>>> np.hstack((a,b))           # the result is different
array([ 4., 2., 3., 8.])
>>> a[:,newaxis]               # this allows to have a 2D columns vector
array([[ 4.],
       [ 2.]])
>>> np.column_stack((a[:,newaxis],b[:,newaxis]))
array([[ 4.,  3.],
       [ 2.,  8.]])
>>> np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same
array([[ 4.,  3.],
       [ 2.,  8.]])
```

另一方面，函数row_stack相当于任何输入数组的vstack。通常，对于具有两个以上维度的数组，hstack堆栈沿着它们的第二个轴，vstack堆栈沿着它们的第一个轴，并且连接允许可选参数给出连接应该发生的轴的数量。

在复杂情况下，r_和c_对于通过沿一个轴堆叠数字来创建数组非常有用。它们允许使用范围文字（“：”）

```python
>>> np.r_[1:4,0,4]
array([1, 2, 3, 0, 4])
```

### 将一个阵列拆分成几个较小的阵列

使用hsplit，您可以沿着水平轴分割数组，方法是指定要返回的同样形状的数组的数量，或者指定应该进行除法的列：

```python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
>>> np.hsplit(a,3)   # Split a into 3
[array([[ 9.,  5.,  6.,  3.],
       [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
       [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
       [ 2.,  2.,  4.,  0.]])]
>>> np.hsplit(a,(3,4))   # 在第三列和第四列之后拆分
[array([[ 9.,  5.,  6.],
       [ 1.,  4.,  9.]]), array([[ 3.],
       [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]
```

vsplit沿垂直轴分割，array_split允许指定要分割的轴。

### Copies and Views

Deep Copy

```python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])

>>> a = np.arange(int(1e8))
>>> b = a[:100].copy()
>>> del a  # the memory of ``a`` can be released.
```

### Functions and Methods Overview

**Array Creation**
    arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like
**Conversions**
    ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat
**Manipulations**
    array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
**Questions**
    all, any, nonzero, where
**Ordering**
    argmax, argmin, argsort, max, min, ptp, searchsorted, sort
**Operations**
    choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum
**Basic Statistics**
    cov, mean, std, var
**Basic Linear Algebra**
    cross, dot, outer, linalg.svd, vdot
