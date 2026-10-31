# Medical Image Segmentation Demo

基于 MONAI 与 SimpleITK 的 3D 医学图像处理与分割基础项目。

## 功能特性
- 使用 SimpleITK 进行医学图像的读取与物理空间信息对齐。
- 实现 3D 图像的 Spacing 重采样与灰度窗宽窗位预处理。
- 基于 MONAI 框架搭建 3D U-Net 模型与滑窗推理流水线（完美解决大图显存溢出 OOM 问题）。