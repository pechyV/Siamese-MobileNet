import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class SiameseUNet(nn.Module):
    def __init__(self):
        super(SiameseUNet, self).__init__()

        # Load MobileNetV2 as an encoder
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.encoder = mobilenet.features 

        # Identify skip connection layers
        self.skip_layers = [2, 4, 7, 14]  

        # Bottleneck layer to match output channels
        self.bottleneck = nn.Conv2d(1280, 256, kernel_size=1)

        # Decoder blocks (Ensure correct input channels)
        self.dec_conv4 = self.conv_block(256 + 160, 256) 
        self.dec_conv3 = self.conv_block(256 + 64, 128)   
        self.dec_conv2 = self.conv_block(128 + 32, 64)    
        self.dec_conv1 = self.conv_block(64 + 24, 32)    


        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Depthwise Separable Convolution (MobileNet-style)"""
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def forward_once(self, x):
        """Forward pass through MobileNetV2 encoder"""
        x = x.repeat(1, 3, 1, 1) 
        skip_features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_layers:
                skip_features.append(x)

        bottleneck = self.bottleneck(x) 
        return skip_features + [bottleneck]

    def forward(self, t1, t2):
        features_t1 = self.forward_once(t1)
        features_t2 = self.forward_once(t2)

        # Compute absolute difference
        diff = [torch.abs(ft1 - ft2) for ft1, ft2 in zip(features_t1, features_t2)]

        # Decoder process
        x = self.dec_conv4(torch.cat([
            F.interpolate(diff[-1], size=diff[-2].shape[2:], mode='bilinear', align_corners=True),
            diff[-2]
        ], dim=1))

        x = self.dec_conv3(torch.cat([
            F.interpolate(x, size=diff[-3].shape[2:], mode='bilinear', align_corners=True),
            diff[-3]
        ], dim=1))

        x = self.dec_conv2(torch.cat([
            F.interpolate(x, size=diff[-4].shape[2:], mode='bilinear', align_corners=True),
            diff[-4]
        ], dim=1))

        x = self.dec_conv1(torch.cat([
            F.interpolate(x, size=diff[-5].shape[2:], mode='bilinear', align_corners=True),
            diff[-5]
        ], dim=1))

        # Final output
        x = self.final_conv(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True) 

        return x


def get_model(device_index):
    model = SiameseUNet()
    
    # Set device according to GPU index
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_index}')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
        model.to(device)
    else:
        logging.info("No GPU available, using CPU.")
        model.to('cpu')
    
    return model
