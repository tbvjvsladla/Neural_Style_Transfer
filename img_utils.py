import torch
from torchvision.transforms import v2

from PIL import Image
import numpy as np

# 이미지를 받아서 텐서 자료형으로 변환하는 함수
# pad_factor는 비율값 [0.1, 0.2, 0.3, 0.4]로 넣어야 한다.
def preprocess_img(img_path, tensor_size, normal_val, 
                   device, pad_factor=None, patch=1):
    assert int(patch ** 0.5) ** 2 == patch, "패치는 제곱수여야 함"

    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    img_shape = [height, width] #원본 이미지의 크기정보를 따로 저장

    #이미지를 패치로 나누기
    patches, patch_size = [], []
    if patch != 1:
        patch_side = int(patch ** 0.5)
        patch_size = [height // patch_side, width // patch_side]
        for n in range(patch):
            i, j = divmod(n, patch_side) #몫, 나머지를 i, j로 활용
            patch_img = img.crop((j * patch_size[1], 
                                  i * patch_size[0], 
                                  (j + 1) * patch_size[1], 
                                  (i + 1) * patch_size[0]))
            patches.append(patch_img)
    else:
        patches.append(img)
        patch_size = [height, width]

    # 패딩 값 설정 (pad_factor가 None이 아닌 경우에만)
    padding = None
    if pad_factor is not None:
        padding = [
            int(patch_size[1] * pad_factor[0]),    # left
            int(patch_size[0] * pad_factor[1]),    # top
            int(patch_size[1] * pad_factor[2]),    # right
            int(patch_size[0] * pad_factor[3])     # bottom
        ]

    # 변환 함수 정의
    def define_transform(tensor_size, normal_val, pad_factor):
        transform_list = []
        if pad_factor is not None:
            transform_list += [v2.Pad(padding, padding_mode='reflect')] #반사패딩 설정
        
        transform_list += [
            v2.Resize(tensor_size), #Tensor_size로 리사이징, tensor_size는 튜플
            v2.ToImage(),  # 이미지를 Tensor 자료형으로 변환
            v2.ToDtype(torch.float32, scale=True), #텐서 자료형을 [0~1]로 정규화
            v2.Normalize(mean=normal_val['mean'], std=normal_val['std']) #데이터셋 표준화
        ]
        return v2.Compose(transform_list)

    transformation = define_transform(tensor_size, normal_val, pad_factor)
    # 각 패치별로 transformation을 적용 -> 텐서 자료형변환
    patch_tensors = [transformation(patch) for patch in patches]
    img_tensor = torch.stack(patch_tensors).to(device)
    # 이렇게 하면 img_tensor의 shape는 (batch_size=patch, 3, Tensor_size로, Tensor_size로)가 됨

    return img_tensor, img_shape



# 텐서 자료형의 이미지를 원복하는 함수
def deprocess_img(img_tensor, img_shape, normal_val, pad_factor=None):
    tenser_size = img_tensor.shape[2:] #텐서 이미지의 H, W 추출
    # 이미지가 패치로 나눠어진 경우 해당 정보를 추출
    batch_size = img_tensor.size(0)
    patch_side = int(batch_size ** 0.5)
    patch_size = (img_shape[0] // patch_side, img_shape[1] // patch_side)

    patches_tensor = img_tensor.detach() #img_tensor가 그래디언트 추적이 활성화 되어 있으면 중단
    if pad_factor is not None:
        # 반사패딩으로 발생한 crop 비율 계산
        crop_pad = [int(pad_factor[1] / (1+ pad_factor[1]+pad_factor[3]) * tenser_size[0]), #top
                    int(pad_factor[0] / (1+ pad_factor[0]+pad_factor[2]) * tenser_size[1]), #left 
                    int(1 / (1+ pad_factor[1]+pad_factor[3]) * tenser_size[0]),             #height
                    int(1 / (1+ pad_factor[0]+pad_factor[2]) * tenser_size[1])              #width
        ]
        #반사 패딩 항목을 제거
        patches_tensor = [v2.functional.crop(img_tensor[i], crop_pad[0]+1, # top
                                                            crop_pad[1]+1, # left
                                                            crop_pad[2]+1, # height
                                                            crop_pad[3]+1) # width
                                        for i in range(batch_size)]

    #패치 텐서를 각 패치별 크기로 원복(리사이징)
    patches_tensor = [v2.functional.resize(patches_tensor[i], patch_size) for i in range(batch_size)]
    #np 자료형으로 변환
    patches_np = [patch_tensor.cpu().numpy().squeeze().transpose(1, 2, 0) for patch_tensor in patches_tensor]

    # 이미지에 표준화 되어 있던걸 원복
    patches_np = [(patch_np * np.array(normal_val['std']) + np.array(normal_val['mean'])) for patch_np in patches_np]
    patches_np = [np.clip(patch_np, 0, 1) for patch_np in patches_np]

    # 패치를 하나의 이미지로 결합
    rows = [np.concatenate(patches_np[i * patch_side:(i + 1) * patch_side], axis=1) for i in range(patch_side)]
    full_img = np.concatenate(rows, axis=0)
    img = (full_img * 255).astype(np.uint8)

    # numpy 배열을 PIL 이미지로 변환
    img = Image.fromarray(img)

    return img



#입력 이미지 정의하기 : 랜덤 노이즈로 이미지 정의
def create_random_noise_img(img_shape):
    # 랜덤 픽셀 값으로 채워진 배열 생성
    height, width = img_shape
    random_data = np.random.rand(height, width, 3) * 255
    random_data = random_data.astype(np.uint8) #이미지 자료형으로 변환

    # 랜덤 노이즈 이미지를 PIL 이미지로 변환
    noise_image = Image.fromarray(random_data, 'RGB')
    
    return noise_image