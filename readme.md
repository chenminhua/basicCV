# CV基本应用
识别：detection, tracking, recognition
重建：3D-Reconstruction, AR

## PIL图像库使用

```python
from PIL import Image
# 读取图片
p = Image.open('gerrard.jpg')
# 图片尺寸
print p.size
# 显示图片
p.show()

# 读取图片并转换为灰度图
p = Image.open('gerrard.jpg').convert('L')

# 保存图片
Image.open('gerrard.jpg').convert('L').save('gerrard_onechannel.png')

# 创建缩略图
p = Image.open('gerrard.jpg')
p.thumbnail((80, 80))
p.save('gerrard_80*80.png')

# resize
resizedP = p.resize((80, 80))

# rotate
rotatedP = p.rotate(45)

# 裁剪
p = Image.open('gerrard.jpg')
box = (100,100,200,200)
p_crop = p.crop(box)
p_crop.show()

# 黏贴
p_rotated_crop = p_crop.transpose(Image.ROTATE_180)
p.paste(p_rotated_crop, box)
```

## matplotlib

```python
from pylab import *
p = Image.open('gerrard.jpg').convert('')
figure()
im = array(p)
print im.shape, im.dtype
# 画出图像的轮廓
contour(im, origin='image')
figure()
# 画出图像直方图
hist(im.flatten(), 128)
show()
```

## 直方图均衡化

```python
def histeq(im, bins=256):
  imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
  cdf = imhist.cumsum()
  cdf = 255 * cdf / cdf[-1]
  im2 = interp(im.flatten(), bins[:-1], cdf)
  return im2.reshape(im.shape), cdf

im = array(Image.open('night.jpeg').convert('L'))
# 直方图均衡化
im2, cdf = imtools.histeq(im)
# 对比图像的变化
Image.fromarray(im).show()
Image.fromarray(im2).show()
# 对比直方图变化
hist(im.flatten(), 128)
hist(im.flatten(), 128)
```

## 图像模糊

```python
```

## 图像去噪

```python
```

## 主成分分析

```python
```


# 增强现实

### 单应性变换

### 照相机模型

### 增强现实

# 三维重建


# 图像聚类

# 图像内容分类

# 图像分割

# 识别与追踪

### opencv

```
```
