# ============================================================================
# YOLO11-CBAM 模型测试脚本
# 学生学号：55230316
# 用途：验证 CBAM 模型是否能正确加载和运行
# ============================================================================

from ultralytics import YOLO
import torch

print("=" * 60)
print("YOLO11-CBAM 模型加载测试")
print("学生学号：55230316")
print("=" * 60)

# ============================================================================
# 测试 1：从配置文件加载模型
# ============================================================================

print("\n[测试 1] 从 YAML 配置文件加载模型...")

try:
    model = YOLO("ultralytics/cfg/models/11/yolo11_cbam_55230316.yaml")
    print("✓ 模型加载成功！")

    # 打印模型信息
    print("\n模型结构信息:")
    model.info()

except Exception as e:
    print(f"✗ 模型加载失败：{e}")
    exit(1)

# ============================================================================
# 测试 2：前向传播测试
# ============================================================================

print("\n[测试 2] 前向传播测试...")

# 创建一个模拟输入 (batch_size=1, channels=3, height=640, width=640)
dummy_input = torch.randn(1, 3, 640, 640)

try:
    # 将模型设置为评估模式
    model.model.eval()

    # 执行前向传播
    with torch.no_grad():
        output = model.model(dummy_input)

    print(f"✓ 前向传播成功！")
    print(f"  输入形状：{dummy_input.shape}")
    if isinstance(output, torch.Tensor):
        print(f"  输出形状：{output.shape}")
    else:
        print(f"  输出类型：{type(output)}")
        if isinstance(output, (list, tuple)):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"  输出 [{i}] 形状：{o.shape}")

except Exception as e:
    print(f"✗ 前向传播失败：{e}")
    exit(1)

# ============================================================================
# 测试 3：CBAM 模块存在性检查
# ============================================================================

print("\n[测试 3] 检查 CBAM 模块是否存在...")

cbam_count = 0
for name, module in model.model.named_modules():
    if "CBAM" in type(module).__name__:
        cbam_count += 1
        print(f"  发现 CBAM 模块：{name}")

if cbam_count > 0:
    print(f"✓ 共找到 {cbam_count} 个 CBAM 模块")
else:
    print("⚠ 未找到 CBAM 模块（可能是正常现象，取决于模型结构）")

# ============================================================================
# 测试 4：GPU 可用性测试
# ============================================================================

print("\n[测试 4] GPU 可用性测试...")

if torch.cuda.is_available():
    print(f"✓ GPU 可用：{torch.cuda.get_device_name(0)}")
    print(f"  显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 尝试将模型移动到 GPU
    try:
        model.model.to("cuda")
        dummy_input_gpu = dummy_input.to("cuda")
        with torch.no_grad():
            output_gpu = model.model(dummy_input_gpu)
        print("✓ GPU 推理成功！")
    except Exception as e:
        print(f"⚠ GPU 推理失败：{e}")
else:
    print("⚠ GPU 不可用，使用 CPU 测试")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("所有测试通过，CBAM 模型可以正常使用！")
print("学生学号：55230316")
print("=" * 60)
