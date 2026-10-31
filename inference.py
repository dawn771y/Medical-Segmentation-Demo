import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet


def main():
  # 1. 检查当前设备（有 GPU 用 GPU，没有用 CPU）
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"当前使用的计算设备: {device}")

  # 2. 实例化 MONAI 的 3D U-Net 模型
  # in_channels=1 对应我们的单通道测试图像，out_channels=3 对应预测类别
  model = UNet(
      spatial_dims=3,
      in_channels=1,
      out_channels=3,
      channels=(16, 32, 64, 128, 256),
      strides=(2, 2, 2, 2),
      num_res_units=2,
  ).to(device)

  # 切换到评估模式
  model.eval()
  print("3D U-Net 模型初始化成功！")

  # 3. 构造一个模拟的 3D 输入张量 (Batch=1, Channel=1, Depth=96, Height=96, Width=96)
  dummy_input = torch.randn(1, 1, 96, 96, 96).to(device)

  # 4. 使用 MONAI 的杀手锏：滑窗推理 (Sliding Window Inference)
  print("正在执行滑窗推理...")
  with torch.no_grad():
    output = sliding_window_inference(
        inputs=dummy_input,
        roi_size=(64, 64, 64),  # 切成小方块送入模型的大小
        sw_batch_size=2,  # 每次并行推理的块数
        predictor=model,
    )

  print(f"推理完成！输出预测结果的张量形状: {output.shape}")


if __name__ == "__main__":
  main()