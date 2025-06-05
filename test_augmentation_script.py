


def plot_grayscale(img, title=None, cmap='gray'):
    """
    打印单个通道的灰度图

    参数:
        img: 输入图像，可以是PIL图像、numpy数组或PyTorch张量
        title: 图像标题，默认为None
        cmap: 颜色映射，默认为'gray'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    plt.figure(figsize=(8, 8))

    # 处理不同类型的输入
    if isinstance(img, Image.Image):
        # 如果是PIL图像，转换为numpy数组
        img_array = np.array(img)
    elif isinstance(img, torch.Tensor):
        # 如果是PyTorch张量，转换为numpy数组
        img_array = img.detach().cpu().numpy()

        # 处理多通道张量 (C,H,W)
        if len(img_array.shape) == 3:
            # 如果是单通道图像，但形状为(1,H,W)
            if img_array.shape[0] == 1:
                img_array = img_array[0]  # 提取单个通道
            else:
                # 多通道图像，仅使用第一个通道
                img_array = img_array[0]
                print("警告：输入为多通道图像，仅显示第一个通道")
    else:
        # 假设是numpy数组
        img_array = img

    plt.imshow(img_array, cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.axis('on')
    plt.tight_layout()
    plt.show()



# ====================you can use this code to test whether the argumentation is right set====================
# ====================the code is not completed, you should customized it yourself====================

idx=409
img_path="data_split/eval/volume-409.png"
seg_path="data_split/eval/segmentation-409.png"
img = Image.open(img_path)
segmentation = Image.open(seg_path)

tran=JointTransforms([
    JointPILToTensor(),
    Jointhflip(p=1),
    Jointvflip(p=1),
    Jointrandomcrop()
])
tran(img,segmentation)