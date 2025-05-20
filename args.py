import os
from dataclasses import dataclass,field
import torch
import time
import torch.optim.lr_scheduler as lr_scheduler

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """实验名称"""

    track: bool = True
    """是否使用 wandb 跟踪实验"""

    wandb_project_name: str = "medicine_segment"
    """wandb 项目名称"""

    learning_rate: float = 2.5e-4
    """优化器初始学习率"""

    learning_rate_min: float = 1e-5
    """学习率下限，仅在选择特定学习率策略时有效"""

    epochs: int = 16
    """总训练轮数"""

    batchsize: int = 4
    """每次训练迭代的批量大小"""

    datadir_path: str = "data_split"
    """数据存放目录。训练集应为 f'{datadir_path}/train'，验证集同理"""

    channel_list: list = field(default_factory=lambda: [64, 128, 256])
    """每一层的通道数，通常对应模型的下采样过程"""

    time: str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    """当前运行时间，自动初始化"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """训练运行的设备，优先使用 GPU"""

    threshold: float = 0.5
    """掩码分割的阈值，例如二值化输出"""

    num_workers: int = 4
    """dataloader 数据加载的并发 worker 数"""

    log_dir: str = ""
    '''日志与 checkpoint 保存的目录，运行时设置'''

    net: str = "dpt"
    """使用的模型类型，可选 'unet' 或 'dpt'"""

    class_reduction: bool = True
    """DiceLoss 是否先对类别求平均再对 batch 求平均"""

    lr_schedule: str = "warmup_cos_anneling"
    """学习率衰减策略，'cos_anneling_warmstart' 或 'warmup_cos_anneling'"""

    decay: float = 0
    '''用于 EMA（指数滑动平均）的衰减因子，若不使用 EMA 可设为 0'''

    resume: str = None
    """若设置，则用于恢复训练的 checkpoint 文件路径"""

    wandb_id: str = None
    """wandb 的实验 id。若未设置则自动创建。用于日志恢复，通常 checkpoint 文件已包含该 id。"""

    # 变量（成员）分布及说明总结：
    # - 实验相关：exp_name, track, wandb_project_name, log_dir, time, wandb_id
    # - 优化与调度相关：learning_rate, learning_rate_min, lr_schedule, decay, epochs, resume
    # - 数据与任务/模型相关：batchsize, datadir_path, channel_list, net, threshold, num_workers, class_reduction, device
