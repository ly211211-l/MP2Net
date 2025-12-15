# GroupNorm 替换与 Backbone BN 冻结实施文档

## 目标
- Backbone（DLA-34/ResNet）保留 BatchNorm2d，训练时冻结：`eval()` + `requires_grad=False`，保留 `track_running_stats=True`（复用预训练统计）。
- 新增层 BatchNorm2d 全部改为 GroupNorm（自适应组数，适合 batch_size=1）。
- 兼容预训练权重，接口不变；训练脚本确保 Backbone BN 冻结，DataParallel 路径正确。

## 主要修改点

### 1) 每个模型文件新增 GroupNorm 辅助函数
```python
def get_group_norm(num_channels, num_groups=32, eps=1e-5, affine=True):
    num_groups = min(num_groups, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
```
最坏退化为 1 组，等效 LayerNorm 风格，适合奇数/小通道数。

### 2) 将新增层的 BatchNorm2d 替换为 GroupNorm
需替换位置（示例文件）：
- `lib/models/dladcn_gru.py`
  - `DeformConv.actf` 中 BN → `get_group_norm(cho)`
  - `MaskProp.bn` → `get_group_norm(dim)`
  - `DeConvGRUNet.proj0 / proj_hm` 中 BN → `get_group_norm(16)`
  - `_make_deconv_layer` 中 BN → `get_group_norm(planes)`
  - 初始化时同时处理 BN/GN：
    ```python
    for m in self.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    ```
- `lib/models/resnet_fpn_dcn.py`：同上（`DeformConv`、`MaskProp`、`DeConvGRUNet`、`_make_deconv_layer`）。
- `lib/models/pose_dla_dcn.py`：若使用，按相同方式替换。
- `lib/models/deconv_gru.py`：若有未注释 BN，同步替换。

### 3) 冻结 Backbone BN（train_satmtb.py / train.py）
在模型创建后、optimizer 之前插入：
```python
def freeze_backbone_bn(model, model_name):
    """
    冻结 backbone 中的 BatchNorm2d：
    - eval(): 使用预训练 running stats，不更新
    - requires_grad=False: 不更新参数
    """
    frozen = 0
    if hasattr(model, 'backbone'):
        # 冻结 backbone
        for _, m in model.backbone.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                frozen += 1
        # PoseResNet 有 base_layer，也一并冻结
        if hasattr(model, 'base_layer'):
            for _, m in model.base_layer.named_modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    frozen += 1
    else:
        # 降级方案：按名称匹配
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                in_backbone = False
                if model_name == 'DLADCN':
                    in_backbone = 'backbone' in name and any(k in name for k in ['base_layer','level0','level1','level2','level3','level4'])
                elif model_name == 'ResFPN':
                    in_backbone = 'backbone' in name and any(k in name for k in ['bn1','layer1','layer2','layer3','layer4'])
                else:
                    in_backbone = 'backbone' in name.lower()
                if in_backbone:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    frozen += 1
    return frozen

# 使用
frozen_count = freeze_backbone_bn(model, opt.model_name)
print(f'冻结 Backbone BN 数量: {frozen_count}')
```
说明：
- 保留 `track_running_stats=True`（默认），才能复用预训练统计。
- 若同时使用 `train.py`，需同步插入该逻辑。

### 4) 训练循环保证 Backbone BN 持续冻结（lib/Trainer/base_trainer.py）
在 `run_epoch` 的 train 分支：
```python
if phase == 'train':
    model_with_loss.train()
    # 获取真实模型（兼容 DataParallel）
    model = model_with_loss.module.model if hasattr(model_with_loss, 'module') else model_with_loss.model
    # 确保 backbone/base_layer 的 BN 始终 eval
    if hasattr(model, 'backbone'):
        for m in model.backbone.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        if hasattr(model, 'base_layer'):
            for m in model.base_layer.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
    else:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d) and 'backbone' in name.lower():
                m.eval()
```

## 兼容性与注意事项
- 预训练权重：Backbone BN 保留，可正常加载并复用 running stats；新增层 GN 无预训练，随机初始化。
- 旧 checkpoint：若包含新增层 BN 参数，加载时需 `strict=False`（跳过不匹配）。
- GroupNorm 组数：自适应；奇数/小通道会退化为可整除的最大组数，最坏为 1 组，行为可接受。
- 性能：GN 比 BN 略慢，但 batch_size=1 时稳定性收益更重要。
- 覆盖范围：确保 `dladcn_gru.py`、`resnet_fpn_dcn.py`、`pose_dla_dcn.py`（如使用）、`deconv_gru.py`（如有 BN）均已替换。
- 脚本一致性：`train_satmtb.py` 与 `train.py` 都需添加冻结逻辑，避免行为差异。

## 验证建议
- **GroupNorm 替换检查**：统计非 backbone 的 `GroupNorm` 与 `BatchNorm2d` 数量，确认新增层已替换。
- **Backbone BN 冻结检查**：打印前几个 backbone/base_layer BN 的 `training` / `requires_grad` 状态，应为 `False / False`。
- **首轮训练检查**：首个 iteration 确认没有 backbone BN 处于 train 模式。

## 预期效果
| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| Backbone BN | train，更新统计 | eval，复用预训练统计 |
| 新增层 BN | BatchNorm2d，bs=1 不稳 | GroupNorm，bs=1 稳定 |
| 训练稳定性 | 可能震荡 | 显著改善 |
| 评估一致性 | 可能不一致 | 更一致 |








