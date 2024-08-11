import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, relu=None):
        super(BasicConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, 
                      padding = 0, bias=False,),
                      
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.relu = relu
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_block(x)
        if self.relu:
            x = self.relu(x)
        return x
    

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.ch = channel # Featuer의 channel값 추출
        self.DS = reduction #연산량 감소를 위한 DownScale값

        # Faeture의 (C, H, W) -> (C, 1, 1)로 squeeze
        self.squeeze_path = nn.AdaptiveAvgPool2d((1, 1))

        self.excitation_path = nn.Sequential(
            nn.Linear(self.ch, self.ch//self.DS, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.ch//self.DS, self.ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, ch, _, _ = x.size() #입력된 feature의 bs, ch정보 추출
        # (bs, ch, 1, 1)이 된것을 (bs, ch)로 줄이기
        y = self.squeeze_path(x).view(bs, ch)
        # excitation_path를 통해 계산된 Attention Score의 shape를
        # (bs, ch)에서 (bs, ch, 1, 1)로 원복
        y = self.excitation_path(y).view(bs, ch, 1, 1)
        # expand_as를 써서 (bs, ch, H, W)로 늘리고 Attention Score를 곱함
        Recalibration = x * y.expand_as(x)

        return Recalibration

class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = BasicConv(in_ch, in_ch, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(in_ch, in_ch, kernel_size=3, stride=1, relu=False)
        self.se_block = SEBlock(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se_block(out)
        out += identity
        out = self.relu(out)
        return out
    
class BasicTRConv(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=1, relu=None):
        super(BasicTRConv, self).__init__()

        self.tr_conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 
                               kernel_size=upsample, stride=upsample,
                               padding=0, output_padding=0, 
                               bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.relu = relu
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tr_conv_block(x)
        if self.relu:
            x = self.relu(x)
        return x
    

class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()

        #인코더
        self.encoder = nn.Sequential(
            BasicConv(3, 32, kernel_size=9, stride=1, relu=True),
            
            BasicConv(32, 64, kernel_size=3, stride=2, relu=True),
            BasicConv(64, 128, kernel_size=3, stride=2, relu=True),
        )

        #Residual Layer
        residual_layer = []
        for i in range(5):
            residual_layer.append(ResidualBlock(128))
        
        self.res_block = nn.Sequential(*residual_layer)


        #디코더
        self.decoder = nn.Sequential(
            BasicTRConv(128, 64, upsample=2, relu=True),
            BasicTRConv(64, 32, upsample=2, relu=True),

            BasicTRConv(32, 3, upsample=1, relu=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_block(x)
        x = self.decoder(x)

        return x