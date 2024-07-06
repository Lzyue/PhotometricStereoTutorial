### 补充代码

#### step 1：

**解方程$Ax = b$**

光度立体法的基本原理是通过求解线性方程组来估计法线。对于每个像素位置 $i$，都有一个方程：
$$
M_i=L\cdot N_i
$$
其中：

- $M_i$是在第 $i$ 个像素处不同光照下的观测值（像素亮度），即测量矩阵的第$i$行。
- $L$ 是光源矩阵，表示各个光源的方向。
- $N_i$ 是在第 iii 个像素处的表面法线。

通过最小二乘法求解法线$N_i$

```python
A = self.L.T
b = self.M.T
N, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

A是光源矩阵的转置，b 是测量矩阵的转置，N 是计算得到的法线矩阵。

#### Step 2：

**归一化法线向量**

由于法线向量的范数应该为1，需要对计算得到的法线向量进行归一化。

```python
self.N = normalize(N.T, axis=1)
```



### 遇到的问题

在服务器上运行，会报以下错误：

```cmd
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/pub/miniconda3/envs/pytorch_2.0.1/lib/python3.9/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, minimal, minimalegl, offscreen, vnc, webgl.
```

因为发现调用的函数涉及到cv2.imshow，改为使用本机运行，报以下错误：

```cmd
cv2.error: OpenCV(4.9.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1272: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
```

解决：pip uninstall opencv-python-headless

可以运行得到cv2.imshow的图片，但是关闭图片会报错，也无法保存图片：

```cmd
cv2.error: OpenCV(4.10.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 
'normal map' in function 'cvDestroyWindow'
```

解决：改写了一下demo.py，直接存储生成的图片，不使用cv库了

```python
# Display and save the normal map
normal_map_img = psutil.disp_normalmap(normal=rps.N, height=rps.height, width=rps.width)
cv2.imwrite('normal_map.png', normal_map_img)    # Save the normal map as an image
```

上述这部分改成：

```python
# 使用 matplotlib 保存和显示法线图
N = np.reshape(rps.N, (rps.height, rps.width, 3))
N = (N - np.min(N)) / (np.max(N) - np.min(N)) * 255.0
N = N.astype(np.uint8)

# 显示图像
plt.imshow(N)
plt.title('normal map')
plt.show()

# 保存图像
plt.imsave('estimated_normalmap.png', N)

print("done.")
```



### 运行结果

图片：estimated_normalmap.png



