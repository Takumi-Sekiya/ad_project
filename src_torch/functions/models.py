import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_weights(m):
    """
    Kerasの HeNormal() に相当する初期化関数
    PyTorchの Linear および Conv3d に適用する
    """
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ==========================================
# 1. Simple 3DCNN
# ==========================================
class Simple3DCNN(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super().__init__()
        
        # Conv ブロック 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(32)
        )
        
        # Conv ブロック 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(64)
        )
        
        # Conv ブロック 3
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(128)
        )
        
        # Conv ブロック 4
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(256)
        )
        
        # Conv ブロック 5 & GAP
        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1, padding=0), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # GlobalAveragePooling3D相当
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(512, 100), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(100, 60), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(60, 20), nn.ReLU(),
            nn.Linear(20, 1) # 出力層 (linear)
        )
        
        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==========================================
# 2. Multimodal 3DCNN
# ==========================================
class Multimodal3DCNN(nn.Module):
    def __init__(self, in_channels, numerical_features_dim, dropout_rate):
        super().__init__()
        
        # 画像特徴抽出ブランチ (Simple3DCNNのBackboneと同じ)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(256)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1, padding=0), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.img_fc = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU()
        )
        
        # 結合後の回帰ブランチ
        self.combined_fc = nn.Sequential(
            nn.Linear(32 + numerical_features_dim, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        self.apply(_init_weights)

    def forward(self, img_input, numerical_input):
        x = self.conv1(img_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        img_features = self.img_fc(x)
        
        # 特徴量の結合
        combined_features = torch.cat((img_features, numerical_input), dim=1)
        output = self.combined_fc(combined_features)
        return output

# ==========================================
# 3. ResNet Blocks
# ==========================================
class ResNetBasicBlock(nn.Module):
    """3D ResNet用のBasic Block (Kerasの記述順を再現)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.bn(out)
        
        # 修正前: out += self.shortcut(x)
        out = out + self.shortcut(x)  # ← 修正後（新しいテンソルとして計算する）
        
        return F.relu(out)

class RegulatedResNetBasicBlock(nn.Module):
    """正則化意識のBasic Block (Kerasの記述順を再現)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # 修正前: out += self.shortcut(x)
        out = out + self.shortcut(x)  # ← 修正後（新しいテンソルとして計算する）
        
        return F.relu(out)

# ==========================================
# 4. 3D ResNet Models
# ==========================================
class ResNet3D(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(32)
        )
        
        self.layer1 = ResNetBasicBlock(32, 64, stride=1)
        self.layer2 = ResNetBasicBlock(64, 64, stride=1)
        
        self.layer3 = ResNetBasicBlock(64, 128, stride=2)
        self.layer4 = ResNetBasicBlock(128, 128, stride=1)
        
        self.layer5 = ResNetBasicBlock(128, 256, stride=2)
        self.layer6 = ResNetBasicBlock(256, 256, stride=1)
        
        self.layer7 = ResNetBasicBlock(256, 512, stride=2)
        self.layer8 = ResNetBasicBlock(512, 512, stride=1)
        
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        self.apply(_init_weights)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class RegulatedResNet3D(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm3d(32)
        )
        
        self.layer1 = ResNetBasicBlock(32, 32, stride=1)
        self.layer2 = ResNetBasicBlock(32, 32, stride=1)
        
        self.layer3 = RegulatedResNetBasicBlock(32, 64, stride=2)
        self.layer4 = RegulatedResNetBasicBlock(64, 64, stride=1)
        
        self.layer5 = RegulatedResNetBasicBlock(64, 128, stride=2)
        self.layer6 = RegulatedResNetBasicBlock(128, 128, stride=1)
        
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        self.apply(_init_weights)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ==========================================
# 5. Model Registry & Builder
# ==========================================
def build_simple_3dcnn(input_shape: tuple, config: dict) -> nn.Module:
    in_channels = input_shape[0]
    return Simple3DCNN(in_channels, config['model']['params']['dropout_rate'])

def build_multimodal_3dcnn(input_shape: tuple, config: dict) -> nn.Module:
    in_channels = input_shape[0]
    num_dim = len(config['data'].get('tabular_input_features', []))
    if num_dim == 0:
        raise ValueError("'tabular_input_features' must be specified for Multimodal3DCNN.")
    return Multimodal3DCNN(in_channels, num_dim, config['model']['params']['dropout_rate'])

def build_3d_resnet(input_shape: tuple, config: dict) -> nn.Module:
    in_channels = input_shape[0]
    return ResNet3D(in_channels, config['model']['params']['dropout_rate'])

def build_regulated_3d_resnet(input_shape: tuple, config: dict) -> nn.Module:
    in_channels = input_shape[0]
    return RegulatedResNet3D(in_channels, config['model']['params']['dropout_rate'])

MODEL_REGISTRY = {
    "Simple3DCNN": build_simple_3dcnn,
    "Multimodal3DCNN": build_multimodal_3dcnn,
    "3DResNet": build_3d_resnet,
    "Regulated3DResNet": build_regulated_3d_resnet,
}

def build_model(input_shape: tuple, config: dict) -> nn.Module:
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered. Available: {list(MODEL_REGISTRY.keys())}")
    
    build_fn = MODEL_REGISTRY[model_name]
    return build_fn(input_shape=input_shape, config=config)