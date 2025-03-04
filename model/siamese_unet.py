import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 
from model.cbam import CBAM

class SiameseUNet(nn.Module):
    def __init__(self):
        super(SiameseUNet, self).__init__()
        
        self.enc_conv1 = self.conv_block(1, 64, use_cbam=False)
        self.enc_conv2 = self.conv_block(64, 128, use_cbam=False)
        self.enc_conv3 = self.conv_block(128, 256, use_cbam=False)
        self.enc_conv4 = self.conv_block(256, 512, use_cbam=False)

        # CBAM je pouze zde v bottlenecku!
        self.bottleneck = self.conv_block(512, 1024, use_cbam=True)  

        self.dec_conv4 = self.conv_block(1024 + 512, 512, use_cbam=False)
        self.dec_conv3 = self.conv_block(512 + 256, 256, use_cbam=False)
        self.dec_conv2 = self.conv_block(256 + 128, 128, use_cbam=False)
        self.dec_conv1 = self.conv_block(128 + 64, 64, use_cbam=False)

        
        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels, use_cbam=False):
        """Konvoluční blok s možností přidání CBAM (pouze pro bottleneck)."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            layers.append(CBAM(out_channels)) 
        return nn.Sequential(*layers)

    
    def forward_once(self, x):
        """Passes a single input through the encoder path."""
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.enc_conv3(F.max_pool2d(x2, kernel_size=2))
        x4 = self.enc_conv4(F.max_pool2d(x3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(x4, kernel_size=2))
        return [x1, x2, x3, x4, bottleneck]
    
    def forward(self, t1, t2):
        """Forward pass for both input images."""
        features_t1 = self.forward_once(t1)
        features_t2 = self.forward_once(t2)
        
        # Compute absolute difference
        diff = [torch.abs(ft1 - ft2) for ft1, ft2 in zip(features_t1, features_t2)]
        
        # Decoder path
        x = self.dec_conv4(torch.cat([
            F.interpolate(diff[-1], scale_factor=2, mode='bilinear', align_corners=True),
            diff[-2]
        ], dim=1))
        x = self.dec_conv3(torch.cat([
            F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
            diff[-3]
        ], dim=1))
        x = self.dec_conv2(torch.cat([
            F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
            diff[-4]
        ], dim=1))
        x = self.dec_conv1(torch.cat([
            F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
            diff[-5]
        ], dim=1))
        
        # Final output layer
        x = self.final_conv(x)
        return torch.sigmoid(x)

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