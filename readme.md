本项目先修课程包括 python基本语法，矩阵分析

本项目主要内容包括 常用库介绍，单应性变换，照相机模型，增强现实，三维重建，图像聚类，图像搜索, 图像搜索，图像识别与跟踪

## CV基本应用

识别：detection, tracking, recognition, image-search
重建：3D-Reconstruction, AR


# PIL图像库使用
[PIL E-bool](http://effbot.org/imagingbook/)

```python
from PIL import Image
import numpy as np
from pylab import *
# 读取图片
p = Image.open('gerrard.jpg')
# 图片尺寸
print p.size

# 读取图片并转换为灰度图
p = Image.open('gerrard.jpg').convert('L')

# 读取图片到数组
arr = np.array(p_gray)

# show image
p.show()
gray()
imshow(arr)
show()

# 创建缩略图, resize, rotate
p.thumbnail((80, 80))
resizedP = p.resize((80, 80))
rotatedP = p.rotate(45)

# 裁剪
p = Image.open('gerrard.jpg')
p.crop((50,0,150,150)).show()

# 旋转并黏贴
p_rotated_crop = p.crop((50,0,150,150)).transpose(Image.ROTATE_180)
p.paste(p_rotated_crop, box)
```

### 图像模糊(微信红包图片)
图像模糊就是将图像和一个高斯核进行卷积操作

$$I_\sigma = I*G_\sigma$$
$$G_\sigma = \frac{1}{2\pi\sigma}e^{-(x^2+y^2)/2\sigma^2}$$

```python
from scipy.ndimage import filters

im = array(Image.open('gerrard.jpg').convert('L'))
im2 = filters.gaussian_filters(im,5)
Image.fromarray(im2).show()
```

### 图像梯度
梯度 $$|\nabla I|=\sqrt{I_x^2+I_y^2}$$
角度 $$arctan(I_x, I_y)$$
导数计算也可以通过卷积来实现
$$I_x=I*D_x, I_y=I*D_y$$

```python
imx, imy = np.zeros(im.shape), np.zeros(im.shape)
filters.sobel(im, 1, imx)
filters.sobel(im, 0, imy)
map(lambda arr : Image.fromarray(arr).show(), [imx, imy, np.sqrt(imx**2, imy**2)])
```


# opencv
[github地址](https://github.com/opencv)

### 直方图均衡化

```python
import cv2
myimg=cv2.imread('sky.jpg')
img=cv2.cvtColor(myimg,cv2.COLOR_BGR2GRAY)
newimg=cv2.equalizeHist(img)
cv2.imshow('src',img)
cv2.imshow('dst',newimg)
```


### 人脸识别实例1
在给定的图片中找到人脸

```python
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
img = cv2.imread('blacklaker.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.5, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
```

### 人脸识别实例2
在摄像头捕获的实时视频中识别人脸

```python
def detect(img, cascade):
    faces = cascade.detectMultiScale(img, 1.3, 4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return []
    faces[:,2:] += faces[:,:2]
    return faces

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    rects = detect(gray, cascade)
    vis = img.copy()
    draw_rects(vis, rects, (0, 255, 0))

    cv2.imshow('facedetect', vis)

    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
```

## 主成分分析

```python
```

# 单应性变换
单应性变换是指将平面内的点映射到另一个平面内的二维投影变换。常用于图像配准，图像纠正和纹理扭曲，以及创建全景图像。

$$\left[
\begin{matrix}
 x' \\
 y' \\
 \omega'
\end{matrix}
\right] = \left[
\begin{matrix}
h_1 &  h_2 &  h_3 \\
h_4 &  h_5 &  h_6 \\
h_7 &  h_8 &  h_9  \\
\end{matrix}
\right]\left[
\begin{matrix}
x \\
y \\
\omega
\end{matrix}
\right]
$$

求解H矩阵共有8个自由度，而每一组点都可以列出两条方程，所以，只要有四组点，就可以找到对应的变换矩阵

### 仿射变换

仿射变换包含了一个可逆矩阵和一个平移向量，可以用于很多应用，比如图像扭曲

$$\left[
\begin{matrix}
 x' \\
 y' \\
 1
\end{matrix}
\right] = \left[
\begin{matrix}
a_1 &  a_2 &  t_x \\
a_3 &  a_4 &  t_y \\
0  &   0   &  1  \\
\end{matrix}
\right]\left[
\begin{matrix}
x \\
y \\
1
\end{matrix}
\right]
$$

求解仿射变换的H矩阵的时候，只有六个自由度，所以，我们只要知道三个对应点，就可以求出H。

相似变化，如下，当s=1时，称为刚体变换
$$\left[
\begin{matrix}
 x' \\
 y' \\
 1
\end{matrix}
\right] = \left[
\begin{matrix}
s*cos(\theta) & -s*sin(\theta)  &  t_x \\
s*sin(\theta) &  s*cos(\theta) &  t_y \\
0  &   0   &  1  \\
\end{matrix}
\right]\left[
\begin{matrix}
x \\
y \\
1
\end{matrix}
\right]
$$

```python
from scipy import ndimage
im = np.array(Image.open('gerrard.jpg').convert('L'))
H = np.array([[1.7, .3, -100], [.3, 1.8, -100], [0, 0, 1]])
im2 = ndimage.affine_transform(im, H[:2, :2], (H[0,2], H[1,2]))
figure()
gray()
imshow(im)
figure()
gray()
imshow(im2)
show()
```

### 实战image_in_image
把一张图片插入另一张图片中的特定位置（需要进行一些缩放，平移，旋转，扭曲）

# 增强现实

### 照相机模型

$$\lambda x = PX&& 将三维的点X投影为二维图像点x,P为一个3*4的矩阵（称为照相机矩阵或者投影矩阵）

相机矩阵可以分解为 $$P = K[R|t]$$。 其中K为内标定矩阵，由照相机自身的参数（焦距，光心位置）决定；而R为照相机的旋转矩阵，t描述相机的三维平移。

[oxford开源数据集](http://www.robots.ox.ac.uk/~vgg/data/data-mview.html)

```python
import camera
points = np.loadtxt('house.p3d').T
points = np.vstack((points, np.ones(points.shape[1])))

P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-10]])
cam = camera.Camera(P)
x = cam.project(points)

plot(x[0],x[1], 'k.')
show()
```

尝试旋转照相机，看看投影的变化轨迹。

### 关于相机标定
相机标定的过程对于CV非常重要，其本质就是求出照相机内标定矩阵的过程。内标定矩阵主要由照相机的光心位置和焦距来决定。很多时候我们假定光心位于图像的中心。获得正确的焦距则比较复杂，需要进行精确测量。

$$f_x = \frac{d_x}{d_X}dZ$$
$$f_y = \frac{d_y}{d_Y}dZ$$
dZ表示标定物体到相机的距离，dx,dy为物体宽和高在图像上的像素数，dX,dY为物体实际宽度和高度


### 增强现实

# 三维重建


# 图像聚类

# 图像内容分类

# 识别与追踪

# 推荐阅读
[python计算机视觉](http://shop.oreilly.com/product/0636920022923.do)
[quora](https://www.quora.com/What-are-the-best-resources-for-learning-Computer-Vision)

