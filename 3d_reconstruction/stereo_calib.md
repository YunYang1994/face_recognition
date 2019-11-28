### 代码：[https://github.com/YunYang1994/stereo_calib/blob/master/stereo_calib.py](https://github.com/YunYang1994/stereo_calib/blob/master/stereo_calib.py)

今天我们将来了解双目相机的参数标定和校正对齐过程。

## 参数标定
整个标定过程主要分为两部分:

### step1:
首先对左右摄像机进行单目标定，获取左右相机的标定参数;

```python
ret, M1, D1, R1, T1 = cv2.calibrateCamera(objpoints, imgpoints_l, imageSize, None, None)
ret, M2, D2, R2, T2 = cv2.calibrateCamera(objpoints, imgpoints_r, imageSize, None, None)
```

其中 **M** 代表的是相机的内部参数（焦距和光心距离），**D** 表示的是相机的畸变参数， **R** 和 **T** 分别表示的是相机的旋转矢量和平移矢量。

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/69800389-1b004280-1210-11ea-8ffd-b8e4c46ba9cf.png" style="max-width:60%;">
    </a>
</p>

本文标定图像的大小是 640*480，理想主点坐标为图像的中心位置，即(320,240)，标定所得结果为(319, 239) ，近似于理想值。

### step2:
然后在单目标定的基础上，获取左右摄像机之间的旋转矩阵和平移向量，从而完成立体标定。

```python
rms, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, M1, D1, M2, D2, imageSize,
                        criteria=(cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 100, 1e-5), flags=flags)
```

<p align="center">
    <img width="60%" src="https://user-images.githubusercontent.com/30433053/69800389-1b004280-1210-11ea-8ffd-b8e4c46ba9cf.png" style="max-width:60%;">
    </a>
</p>

**旋转向量**中的各个分量很小，表明左右相机之间没有发生旋转，该结果与左右摄像机水平摆放的实际情景相符合。**平移向量**表示左右摄像机在三维世界中 X、Y、Z 轴方向上的距离，本文双目立体视觉 系统的基线 `B=60mm` ，标定结果和实际值近似相等。另外两个方向上的距离近似于零，这和实际情况相符合。

## 校正对齐

校正对齐采用的是[Bouguet](http://hvrl.ics.keio.ac.jp/charmie/doc/CameraCalibration/BouguetCameraCalibrationToolbox.pdf) 算法, 这里不做赘述。校正对齐的大致流程如下：

<p align="center">
    <img width="40%" src="https://user-images.githubusercontent.com/30433053/69801124-8991d000-1211-11ea-93c6-5d8afd2465db.png" style="max-width:40%;">
    </a>
</p>


```python
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, D1, M2, D2, imageSize, R, T, 
                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=imageSize)
mapx1, mapy1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, imageSize, cv2.CV_16SC2)
mapx2, mapy2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, imageSize, cv2.CV_16SC2)
```

首先利用畸变系数对图像对做矫正，然后双目标定的参数作为输入，调用 OpenCV 中 stereoRectify 与 initUndistortRectifyMap 函数得到 Bouguet 算法中所需的旋转参数与投影参数，最 后利用 remap 函数获取双目校正、裁剪后的图像对。



