import torch
from torch import nn
import torch.nn.functional as F

class MLARUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv',
        img_dim=512,
        patch_dim=2,
        embedding_dim=1024,
        num_heads=4,
        num_layers=1,
        hidden_dim=1024*2,
        dropout_rate=0
    ):
        super(MLARUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.CBAM2 = CBAM(64, 2, 7)
        self.ACRM4 = ACRM(256, 8, 7, embedding_dim // 4, img_dim // 4, patch_dim)
        self.ACRM5 = ACRM(512, 16, 7, embedding_dim // 2, img_dim // 8, patch_dim)
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.LACRM = LACRM(embedding_dim, 7, img_dim=img_dim//16, patch_dim=patch_dim, hidden_dim=hidden_dim, number_heads=num_heads, number_layers=num_layers, dropout_rate=dropout_rate)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i == 0:
                x = self.CBAM2(x)
            if i == 2:
                x = self.ACRM4(x)
                x_ACRM1 = x
            if i == 3:
                x = self.ACRM5(x)
                x_ACRM0 = x
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        x = self.LACRM(x)

        for i, up in enumerate(self.up_path):
            if i == 0:
                x = up(x, blocks[-i - 1]) + x_ACRM0
            if i == 1:
                x = up(x, blocks[-i - 1]) + x_ACRM1
            if i > 1:
                x = up(x, blocks[-i - 1])

        output = self.last(x)

        return output


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        block.append(nn.Dropout2d(p=0.15)) # edited
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class Transformer_Encoder(nn.Module):
    def __init__(self, img_dim, patch_dim, num_channels, embedding_dim, num_heads, num_layers, hidden_dim, dropout_rate=0):
        super(Transformer_Encoder, self).__init__()
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.img_dim = img_dim
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.out_dim = patch_dim * patch_dim * num_channels
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=nn.LayerNorm(embedding_dim))
        self.dropout_layer1 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()

        x = self.encoder(x, src_key_padding_mask=mask)

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        CA = self.ca(x)
        out = x * CA
        SA = self.sa(out)
        result = out * SA
        return result

class ACRM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size=7, num_channels=1024, img_dim=1, patch_dim=1):
        super(ACRM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.TE1 = Transformer_Encoder(1, 1, num_channels, num_channels, 4,
                                                      1, 2048, 0)
        self.sa = SpatialAttention(kernel_size)
        self.TE2 = Transformer_Encoder(img_dim, patch_dim, 1, patch_dim*patch_dim, 4,
                            1, 1024, 0)

    def forward(self, x):
        CA = self.ca(x)
        CA = self.TE1(CA) + CA
        out = x * CA
        SA = self.sa(out)
        SA = self.TE2(SA) + SA
        result = out * SA
        return result

class CrossChannelAttention(nn.Module):
    def __init__(self, in_planes, number_heads=8, number_layers=2, dropout_rate=0):
        super(CrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.max_pool = nn.AdaptiveMaxPool2d(2)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.TE = Transformer_Encoder(2, 1, in_planes // 8, in_planes // 8, number_heads, number_layers, in_planes // 2, dropout_rate)
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, stride=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.TE(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.TE(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CrossSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, img_dim=128, patch_dim=2, hidden_dim=1024, number_heads=8, number_layers=2, dropout_rate=0):
        super(CrossSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.TE = Transformer_Encoder(img_dim, patch_dim, 2, 2 * patch_dim * patch_dim, number_heads, number_layers, hidden_dim, dropout_rate)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.TE(x)
        x = self.conv1(x)
        return self.sigmoid(x)

class LACRM(nn.Module):
    def __init__(self, in_planes, kernel_size=7, img_dim=32, patch_dim=2, hidden_dim=1024, number_heads=8, number_layers=2, dropout_rate=0):
        super(LACRM, self).__init__()
        self.ca = CrossChannelAttention(in_planes, number_heads, number_layers, dropout_rate)
        self.sa = CrossSpatialAttention(kernel_size, img_dim, patch_dim, hidden_dim, number_heads, 3, dropout_rate)

    def forward(self, x):
        CA = self.ca(x)
        out = x * CA
        SA = self.sa(out)
        result = out * SA
        return result
