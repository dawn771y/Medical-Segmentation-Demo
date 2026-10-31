import numpy as np
import SimpleITK as sitk


def load_medical_image(file_path):
  """读取医学图像并打印基本信息"""
  image = sitk.ReadImage(file_path)
  print(f"成功读取: {file_path}")
  print(f" - Spacing: {image.GetSpacing()}")
  print(f" - Size: {image.GetSize()}")
  return image


def resample_image(image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
  """将医学图像重采样到统一的 Voxel Spacing"""
  original_spacing = image.GetSpacing()
  original_size = image.GetSize()

  out_size = [
      int(round(original_size[i] * original_spacing[i] / out_spacing[i]))
      for i in range(3)
  ]

  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(
      sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
  )
  resample.SetOutputSpacing(out_spacing)
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputDirection(image.GetDirection())
  resample.SetSize(out_size)
  resample.SetTransform(sitk.Transform())

  return resample.Execute(image)


def window_and_normalize(img_np, window_min=-100, window_max=200):
  """灰度窗宽窗位截断与归一化"""
  img_clipped = np.clip(img_np, window_min, window_max)
  img_normalized = (img_clipped - window_min) / (
      window_max - window_min + 1e-8
  )
  return img_normalized