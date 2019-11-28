### 代码：[https://github.com/YunYang1994/stereo_calib/blob/master/camera_calib.py](https://github.com/YunYang1994/stereo_calib/blob/master/camera_calib.py)

今天我们将会通过这个博客来了解相机的内参、外参 (intrinsic、extrinsic parameters) 以及去畸变过程。

## 图像畸变
当今廉价的针孔相机会由于透镜制造精度以及组装工艺的偏差会引入畸变，导致原始图像失真。这两种主要变形是**径向畸变**和**切向畸变**。

**径向畸变**会造成图像的直线会显得弯曲。当我们远离图像中心时，这种效果会更明显。例如下图中，棋盘的两个边缘用红线标记。但是我们会看到边框不是直线，并且与红线不匹配。


<p align="center">
    <img width="30%" src="https://user-images.githubusercontent.com/30433053/69615340-a7273400-106f-11ea-86ca-a1b742add8b2.png" style="max-width:40%;">
    </a>
</p>

可以通过以下方程进行修正：

<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/69699797-c8e1f300-1123-11ea-9071-7c03b2b4de37.png" style="max-width:50%;">
    </a>
</p>

类似地，另一个畸变是**切向畸变**，这是由于摄像透镜未完全平行于成像平面对齐而发生的。因此，图像中的某些区域可能看起来比预期的要近。解决方法如下:

<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/69699861-f29b1a00-1123-11ea-8fc8-2774e5bb016f.png" style="max-width:50%;">
    </a>
</p>

综上，我们需要求解出5个参数，把它们统称为**畸变参数**：

<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/69699917-20805e80-1124-11ea-8822-0185a75b0661.png" style="max-width:50%;">
    </a>
</p>

## 坐标系变换

<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/69796577-3a47a180-1209-11ea-9bc0-cd4121fe10c9.png" style="max-width:50%;">
    </a>
</p>

根据立体视觉的知识，很容易得到图像像素坐标系与世界坐标系之间的转换关系：

<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/69615650-37657900-1070-11ea-9840-f61d0d326862.png" style="max-width:50%;">
    </a>
</p>

- (u,v) 表示该物体对应的像素坐标
- (fx, fy) 表示的是焦距矢量分别在 x 轴与 y 轴上的投影长度（像素单位）
- (uo, vo) 表示成像平面的原点在像素坐标系中的位置（由于工艺上的制造误差，像素平面的原点与成像平面的原点之间会有偏移）

归一化焦距参数 ( **fx**、**fy** ) 和光心坐标 (**uo**、**vo** ) 等参数为内部参数，它反映的是相机机坐标系到图像坐标系之间的投影关系；**R** 和 **T** 组成了相机的外部参数，它描述了相机坐标系和世界坐标系之间的旋转和平移关系。这里补充一下相机坐标系与世界坐标系的关系：

<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/69794182-c60aff00-1204-11ea-88df-dea4a06e170f.png" style="max-width:40%;">
    </a>
</p>

**假如我们设定相机的中心为世界坐标系中心的话，那么 R 则为单位矩阵，T 则为零矩阵，这样就比较省事**。

## 相机的标定

通过上述关系我们可以发现，假如一些图片的像素坐标已知，世界坐标系也已知，那么不就可以求解出相机的内参和外参了嘛。这里使用的是一个 7X6 的棋盘方格，然后利用 **cv2.findChessboardCorners** 这个函数，它能帮助我们找到一些角点位置。一旦我们找到它们了，就可以利用 **cv2.cornerSubPix** 更精确地定位它们的位置，然后再将它们画出来。

```python
import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
```

## 相机的校准
现在我们已经通过相机的标定过程找到棋盘上的角点了，并且知道每个方格的尺寸 (square size)，因此可以知道每个角点在真实世界和图像像素上的坐标。最后就可以通过校准求解出相机的内参和外参矩阵。<br>

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
```

我们可以通过 **cv2.calibrateCamera** 函数求出以下关键信息：

- 相机矩阵(camera matrix)， 它包含了相机的焦距和光心坐标的信息
- 畸变系数(distortion coefficients)，它包含了相机的畸变参数
- 旋转矢量(rotation vector)
- 平移矢量(translation vectors)


## 图像去畸变
现在我们已经获得了相机的内外参数信息了，我们就可以对其中一张图片进行去畸变。在进行去畸变之前，我们需要对输入图片进行一些预处理操作。

```python
img = cv2.imread('left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
```
函数 **cv2.getOptimalNewCameraMatrix** 是调节视场大小，为1时视场大小不变，小于1时缩放视场.

### 1. Using cv2.undistort()
方法很直接，直接调用该函数就可以得到去畸变区域

```python
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
```

### 2. Using remapping

方法有点绕: 它是首先通过找到从扭曲图像到未扭曲图像的映射函数,然后使用重新映射得到去畸变区域

```python
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
```
但是两种方法的结果都一样，如下图所示，我们可以看到棋盘的边缘已经变直了, 说明已经达到去畸变的效果了。

<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/69792707-103eb100-1202-11ea-96da-57f80509c635.png" style="max-width:40%;">
    </a>
</p>
由于边缘的径向畸变比中心的径向畸变严重
上图是去畸变之后的图像，径向畸变已经移除，但是我们会发现原图的边缘区域要比中心弯的更厉害。为什么会出现这个情况，这是因为边缘的径向畸变比中心的径向畸变严重。

<p align="center">
    <img width="30%" src="https://user-images.githubusercontent.com/30433053/69792920-76c3cf00-1202-11ea-8d80-f1ee15499775.png" style="max-width:30%;">
    </a>
</p>
