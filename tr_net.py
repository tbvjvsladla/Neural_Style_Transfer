import torch
import torch.nn as nn
import torchprofile #flops 측정 라이브러리


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, **kwargs):
        super(BasicConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, bias=False, **kwargs),
            # BN에서 IN오로 교체 + IN이 Trainable하게 변경
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x
    
class DepthSep(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthSep, self).__init__()
        # depthwise에는 pad = 1옵션이 있는 것을 ReflectionPad2d(pad=1)옵션으로 변경
        self.ref_pad = nn.ReflectionPad2d(padding=1)
        self.depthwise = BasicConv(in_ch, in_ch, kernel_size=3, stride=stride,
                                   groups = in_ch)
        self.pointwise = BasicConv(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.ref_pad(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class DepthSepRes(nn.Module): #채널변동이 없는 레이어는 모두 Residual block로 처리
    def __init__(self, in_ch):
        super(DepthSepRes, self).__init__()
        self.ref_pad = nn.ReflectionPad2d(padding=1)
        self.depthwise = BasicConv(in_ch, in_ch, kernel_size=3, stride=1,
                                   groups = in_ch)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, bias=False),
            # BN에서 IN오로 교체 + IN이 Trainable하게 변경
            nn.InstanceNorm2d(in_ch, affine=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.ref_pad(x)
        out = self.depthwise(x)
        out = self.pointwise(x)
        out += identity #residual_connection
        out = self.relu(out)

        return out
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.ch = channel # Featuer의 channel값 추출
        self.DS = reduction #연산량 감소를 위한 DownScale값
        self.squeeze_path = nn.AdaptiveAvgPool2d((1, 1))

        self.excitation_path = nn.Sequential(
            nn.Linear(self.ch, self.ch//self.DS, bias=False),
            nn.ReLU(),
            nn.Linear(self.ch//self.DS, self.ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, ch, _, _ = x.size()
        y = self.squeeze_path(x).view(bs, ch)
        y = self.excitation_path(y).view(bs, ch, 1, 1)
        Recalibration = x * y.expand_as(x)
        return Recalibration

class UpSample(nn.Module):
    #Upsample는 ConvTranspose2d를 사용하기
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # self.ref_pad = nn.ReflectionPad2d(padding=1)
        self.up_conv = nn.ConvTranspose2d(channel, channel//2, 
                                kernel_size=2, stride=2,)
        self.IN = nn.InstanceNorm2d(channel//2, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.ref_pad(x)
        x = self.up_conv(x)
        x = self.IN(x)
        x = self.relu(x)

        return x
    
class SEMobileNetTR(nn.Module):
    def __init__(self, width_multiplier=1):
        super(SEMobileNetTR, self).__init__()

        self.alpha = width_multiplier #네트워크 각 층의 필터 개수를 조정

        self.head = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            BasicConv(3, int(32*self.alpha), kernel_size=3, stride=2)
        )

        self.M_DepthSep = nn.ModuleDict()
        self.M_SEBlock = nn.ModuleDict()

        self.M_DepthSep['ds1'] = nn.Sequential(
            DepthSep(int(32*self.alpha), int(64*self.alpha)),
            DepthSep(int(64*self.alpha), int(128*self.alpha), stride=2),
        )
        self.M_SEBlock['se1'] = SEBlock(int(128*self.alpha))

        self.M_DepthSep['ds2'] = nn.Sequential(
            DepthSepRes(int(128*self.alpha)),
            DepthSep(int(128*self.alpha), int(256*self.alpha), stride=2),
        )
        self.M_SEBlock['se2'] = SEBlock(int(256*self.alpha))

        self.M_DepthSep['ds3'] = nn.Sequential(
            DepthSepRes(int(256*self.alpha)),
            DepthSep(int(256*self.alpha), int(512*self.alpha), stride=2),
        )
        self.M_SEBlock['se3'] = SEBlock(int(512*self.alpha))

        self.M_DepthSep['ds4'] = nn.Sequential(
            DepthSepRes(int(512*self.alpha)),
            DepthSepRes(int(512*self.alpha)),
            DepthSepRes(int(512*self.alpha)),
            DepthSepRes(int(512*self.alpha)),
            DepthSepRes(int(512*self.alpha)),
        )
        self.M_SEBlock['se4'] = SEBlock(int(512*self.alpha))

        self.upsample = nn.Sequential(
            UpSample(int(512 * self.alpha)),
            DepthSepRes(int(256*self.alpha)),
            
            UpSample(int(256 * self.alpha)),
            DepthSepRes(int(128*self.alpha)),

            UpSample(int(128 * self.alpha)),
            DepthSepRes(int(64*self.alpha)),

            UpSample(int(64 * self.alpha)),
            # UpSample(int(32 * self.alpha)),
        )

        self.tail = nn.Sequential(
            BasicConv(int(32*self.alpha), 3, kernel_size=1)
        )


    def forward(self, x):
        x = self.head(x)

        for i in range(1, 5):
            DWS_block = self.M_DepthSep[f'ds{i}']

            for DWS_layer in DWS_block:
                x = DWS_layer(x)

            if i > 2:
                x = self.M_SEBlock[f'se{i}'](x)

        x = self.upsample(x)
        x = self.tail(x)

        return x
    



def debug(model, input_size):
    debug_tensor = torch.rand(1, *input_size)
    output = model(debug_tensor)

    # FLOPs 계산
    flops = torchprofile.profile_macs(model, debug_tensor)

    print(f"In_tensor: {debug_tensor.size()}")
    print(f"Feature Out: {output.size()}")
    print(f"FLOPs: {(flops / 1e6):.2f} MFLOPs")  # MFLOPs 단위로 출력


if __name__ == '__main__':
    #디버그 함수 사용 예시
    debug(SEMobileNetTR,input_size=(3, 256, 256))