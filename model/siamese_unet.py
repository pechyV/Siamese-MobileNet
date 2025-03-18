import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 

class SiameseUNet(nn.Module):
    def __init__(self):
        super(SiameseUNet, self).__init__()

        # Encoder
        self.enc_conv1 = self.conv_block(1, 32)  # Menší počet filtrů
        self.enc_conv2 = self.conv_block(32, 64)
        self.enc_conv3 = self.conv_block(64, 128)
        self.enc_conv4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)  # Sníženo z 512 na 256

        # Decoder
        self.dec_conv4 = self.conv_block(256 + 256, 256, True) 
        self.dec_conv3 = self.conv_block(256 + 128, 128, True)
        self.dec_conv2 = self.conv_block(128 + 64, 64, True)
        self.dec_conv1 = self.conv_block(64 + 32, 32, True)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

        self.apply(self.initialize_weights)

    def conv_block(self, in_channels, out_channels, use_dropout=False):
        """Konvoluční blok"""
        if in_channels < 128:
            # Use standard convolutions for smaller layers
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            # Apply DepthwiseSeparableConv for larger layers
            # Apply a 1x1 convolution to reduce channels to match the expected input for depthwise separable conv
            layers = [
                nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0),  # Reduce channels to 128
                DepthwiseSeparableConv(128, out_channels),  # Apply Depthwise Separable Conv
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        if use_dropout:
            layers.append(nn.Dropout2d(p=0.1))

        return nn.Sequential(*layers)


    def forward_once(self, x):
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.enc_conv3(F.max_pool2d(x2, kernel_size=2))
        x4 = self.enc_conv4(F.max_pool2d(x3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(x4, kernel_size=2))
        return [x1, x2, x3, x4, bottleneck]

    def forward(self, t1, t2):
        features_t1 = self.forward_once(t1)
        features_t2 = self.forward_once(t2)

        diff = [torch.abs(ft1 - ft2) for ft1, ft2 in zip(features_t1, features_t2)]

        # Decoder
        x = self.dec_conv4(torch.cat([F.interpolate(diff[-1], scale_factor=2), diff[-2]], dim=1))
        x = F.relu(x)  # Activation

        x = self.dec_conv3(torch.cat([F.interpolate(x, scale_factor=2), diff[-3]], dim=1))
        x = F.relu(x)

        x = self.dec_conv2(torch.cat([F.interpolate(x, scale_factor=2), diff[-4]], dim=1))
        x = F.relu(x)

        x = self.dec_conv1(torch.cat([F.interpolate(x, scale_factor=2), diff[-5]], dim=1))
        x = F.relu(x)

        return self.final_conv(x)


    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.pointwise(F.relu(self.depthwise(x))))



def get_model(device_index):
    model = SiameseUNet()
    
    # Nastavení zařízení podle indexu GPU
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_index}')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
        model.to(device)
    else:
        logging.info("No GPU available, using CPU.")
        model.to('cpu')
    
    return model
