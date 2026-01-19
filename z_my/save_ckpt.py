import torch

# 指定权重文件路径
# 指定权重文件路径
checkpoint_path = '../ckpt/freq3dlane_openlane.pth'

save_path = '../ckpt/openlane_latest.pth'  # 保存模型状态的路径


# 加载权重文件
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 使用 CPU 加载
except Exception as e:
    print(f"Error loading the checkpoint: {e}")
    exit()

# 提取 model_state
if 'model_state' in checkpoint:
    model_state_dict = checkpoint['model_state']

    # 创建新的检查点字典
    new_checkpoint = {
        'model_state': model_state_dict,
        'optimizer_state': None  # 设置为 NoneType
    }

    # 保存新的检查点
    torch.save(new_checkpoint, save_path)
    print(f"Modified checkpoint saved to {save_path}")

    # 显示新的检查点内容
    print("New Checkpoint contents:")
    for key in new_checkpoint.keys():
        print(f"{key}: {type(new_checkpoint[key])}, shape: {getattr(new_checkpoint[key], 'shape', 'N/A')}")
else:
    print("No model_state found in checkpoint.")
