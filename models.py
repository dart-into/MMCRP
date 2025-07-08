import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class BwCombinedInteractModel50(nn.Module):
    def __init__(self, num_classes, interaction_layer="layer2", interaction_ratio=0.1, weights=ResNet50_Weights.IMAGENET1K_V1):
        super(BwCombinedInteractModel50, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.interaction_layer_name = interaction_layer
        self.interaction_ratio = interaction_ratio

        def create_base_model():
            model = resnet50(weights=weights)
            # 冻结早期层
            for name, param in model.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
            model.fc = nn.Identity()
            return model.to(self.device)

        # 创建三个分支的基础模型
        self.full_image_model = create_base_model()
        self.black_image_model = create_base_model()
        self.white_image_model = create_base_model()

        self.reduce_dim_full = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ).to(self.device)

        self.reduce_dim_black = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ).to(self.device)

        self.reduce_dim_white = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ).to(self.device)

        self.classifier = nn.Linear(512 * 3, num_classes).to(self.device)
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]

    def _feature_interaction(self, full, black, white):
        """
        通道级别的特征交互
        full: 给出10%的通道给black和white
        black/white: 各自给出5%的通道给full
        """
        if full is None or black is None or white is None:
            return full, black, white       
        device = full.device
        black = black.to(device)
        white = white.to(device)        
        # 获取维度信息
        B, C, H, W = full.shape

        
        num_channels_from_full = int(C * 0.10)  # full给出10%
        num_channels_from_bw = int(C * 0.05)    # black/white各给出5%

    
        full_channels = torch.randperm(C)[:num_channels_from_full]
        black_channels = torch.randperm(C)[:num_channels_from_bw]
        white_channels = torch.randperm(C)[:num_channels_from_bw]

        # 创建交互后的特征图
        new_full = full.clone()
        new_black = black.clone()
        new_white = white.clone()

        new_black[:, full_channels] = full[:, full_channels]
        new_white[:, full_channels] = full[:, full_channels]

        new_full[:, black_channels] = black[:, black_channels]
        new_full[:, white_channels] = white[:, white_channels]

        return new_full, new_black, new_white

    def forward(self, full_image, black_image, white_image):
        full_image = full_image.to(self.device)
        black_image = black_image.to(self.device)
        white_image = white_image.to(self.device)

        # 前向传播通过基础层
        f = self.full_image_model.maxpool(
            self.full_image_model.relu(self.full_image_model.bn1(self.full_image_model.conv1(full_image))))
        b = self.black_image_model.maxpool(
            self.black_image_model.relu(self.black_image_model.bn1(self.black_image_model.conv1(black_image))))
        w = self.white_image_model.maxpool(
            self.white_image_model.relu(self.white_image_model.bn1(self.white_image_model.conv1(white_image))))

        # 通过各个层并在指定层进行特征交互
        for layer_name in self.layer_names:
            full_layer = getattr(self.full_image_model, layer_name)
            black_layer = getattr(self.black_image_model, layer_name)
            white_layer = getattr(self.white_image_model, layer_name)

            f = full_layer(f)
            b = black_layer(b)
            w = white_layer(w)

            if layer_name == self.interaction_layer_name:
                f, b, w = self._feature_interaction(f, b, w)

        f = torch.flatten(self.full_image_model.avgpool(f), 1)
        b = torch.flatten(self.black_image_model.avgpool(b), 1)
        w = torch.flatten(self.white_image_model.avgpool(w), 1)

        f = self.reduce_dim_full(f)
        b = self.reduce_dim_black(b)
        w = self.reduce_dim_white(w)
        combined_features = torch.cat([f, b, w], dim=1)
        output = self.classifier(combined_features)

        return output
    

def forward_with_weights(self, full_image, black_image, white_image, weights):
    device = self.device
    full_image = full_image.to(device)
    black_image = black_image.to(device)
    white_image = white_image.to(device)

    named_weights = {name: w for name, w in zip([n for n, _ in self.named_parameters()], weights)}

    def functional_block(image, prefix):
        x = F.conv2d(image, named_weights[f'{prefix}.conv1.weight'], named_weights.get(f'{prefix}.conv1.bias'), stride=2, padding=3)
        x = F.batch_norm(x, running_mean=None, running_var=None,
                         weight=named_weights.get(f'{prefix}.bn1.weight'), bias=named_weights.get(f'{prefix}.bn1.bias'), training=True)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for layer_name in self.layer_names:
            layer = getattr(self.full_image_model, layer_name)
            for blk_idx, block in enumerate(layer):
                identity = x
                if hasattr(block, 'downsample') and block.downsample is not None:
                    identity = F.conv2d(x, named_weights[f'{prefix}.{layer_name}.{blk_idx}.downsample.0.weight'],
                                        named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.downsample.0.bias'),
                                        stride=block.downsample[0].stride, padding=0)
                    identity = F.batch_norm(identity, running_mean=None, running_var=None,
                                             weight=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.downsample.1.weight'),
                                             bias=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.downsample.1.bias'), training=True)
                out = F.conv2d(x, named_weights[f'{prefix}.{layer_name}.{blk_idx}.conv1.weight'],
                               named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.conv1.bias'), stride=block.conv1.stride, padding=block.conv1.padding)
                out = F.batch_norm(out, running_mean=None, running_var=None,
                                    weight=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.bn1.weight'),
                                    bias=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.bn1.bias'), training=True)
                out = F.relu(out)
                out = F.conv2d(out, named_weights[f'{prefix}.{layer_name}.{blk_idx}.conv2.weight'],
                               named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.conv2.bias'), stride=block.conv2.stride, padding=block.conv2.padding)
                out = F.batch_norm(out, running_mean=None, running_var=None,
                                    weight=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.bn2.weight'),
                                    bias=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.bn2.bias'), training=True)
                out = F.relu(out)
                out = F.conv2d(out, named_weights[f'{prefix}.{layer_name}.{blk_idx}.conv3.weight'],
                               named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.conv3.bias'), stride=block.conv3.stride, padding=block.conv3.padding)
                out = F.batch_norm(out, running_mean=None, running_var=None,
                                    weight=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.bn3.weight'),
                                    bias=named_weights.get(f'{prefix}.{layer_name}.{blk_idx}.bn3.bias'), training=True)
                x = F.relu(out + identity)

                if layer_name == self.interaction_layer_name and blk_idx == len(layer) - 1:
                    return x
        return x

    f = functional_block(full_image, 'full_image_model')
    b = functional_block(black_image, 'black_image_model')
    w = functional_block(white_image, 'white_image_model')

    f, b, w = self._feature_interaction(f, b, w)

    def flatten_and_reduce(x, reduce_layer, prefix):
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = F.linear(x, named_weights[f'{prefix}.0.weight'], named_weights[f'{prefix}.0.bias'])
        x = F.batch_norm(x, running_mean=None, running_var=None,
                         weight=named_weights.get(f'{prefix}.1.weight'), bias=named_weights.get(f'{prefix}.1.bias'), training=True)
        x = F.relu(x)
        return x

    f = flatten_and_reduce(f, self.reduce_dim_full, 'reduce_dim_full')
    b = flatten_and_reduce(b, self.reduce_dim_black, 'reduce_dim_black')
    w = flatten_and_reduce(w, self.reduce_dim_white, 'reduce_dim_white')

    combined_features = torch.cat([f, b, w], dim=1)
    output = F.linear(combined_features, named_weights['classifier.weight'], named_weights['classifier.bias'])

    return output
