import os
import nibabel as nib
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# ====================the code is used to divide the nii docu into png====================
# ====================the nii codu can't be used to train net directly because of the strong correlations between one nii docu====================

def save_nii_slices_worker(args):
    """工作函数，用于多进程处理单个NII文件"""
    train_dir, idx, output_dir, start_count = args
    local_count = 0

    # 加载体积和分割文件
    volume_path = os.path.join(train_dir, f"volume-{idx}.nii")
    seg_path = os.path.join(train_dir, f"segmentation-{idx}.nii")

    volume_img = nib.load(volume_path)
    seg_img = nib.load(seg_path)

    volume_data = volume_img.get_fdata()
    seg_data = seg_img.get_fdata()

    # 处理z轴上的每个切片
    for i in range(0, volume_data.shape[2]):
        # 获取体积切片并归一化
        volume_slice = volume_data[:, :, i]
        if volume_slice.max() != volume_slice.min():
            volume_slice = ((volume_slice - volume_slice.min()) /
                            (volume_slice.max() - volume_slice.min() + 1e-6) * 255)
        volume_slice = volume_slice.astype(np.uint8)

        # 获取分割切片
        seg_slice = (seg_data[:, :, i]).astype(np.uint8)

        # 转换为PIL图像
        volume_img_slice = Image.fromarray(volume_slice, mode="L")
        seg_img_slice = Image.fromarray(seg_slice, mode="L")

        # 保存两个切片
        slice_idx = start_count + local_count
        volume_output_path = os.path.join(output_dir, f"volume-{slice_idx}.png")
        seg_output_path = os.path.join(output_dir, f"segmentation-{slice_idx}.png")

        volume_img_slice.save(volume_output_path)
        seg_img_slice.save(seg_output_path)

        local_count += 1

    return local_count


def create_directories():
    """创建数据分割所需的目录"""
    os.makedirs("data_split/train", exist_ok=True)
    os.makedirs("data_split/eval", exist_ok=True)


def process_files_parallel(file_indices, train_dir, output_dir, num_processes=None):
    """并行处理多个NII文件"""
    if num_processes is None:
        num_processes = 6

    print(f"使用 {num_processes} 个进程进行并行处理")

    # 准备工作任务
    tasks = []
    start_count = 0
    for idx in file_indices:
        nii_file = os.path.join(train_dir, f"volume-{idx}.nii")
        if os.path.exists(nii_file):
            tasks.append((train_dir, idx, output_dir, start_count))
            # 预先计算每个文件的切片数，以便为下一个文件设置正确的起始计数
            volume_img = nib.load(nii_file)
            start_count += volume_img.shape[2]

    # 使用进程池并行处理
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(save_nii_slices_worker, tasks), total=len(tasks)))

    return sum(results)  # 返回处理的总切片数


if __name__ == "__main__":
    create_directories()
    # 处理训练数据
    train_dir = "data/train"
    total_files = len(os.listdir(train_dir)) // 2
    split_idx = int(total_files * 0.9)

    # 训练文件
    print("处理训练数据...")
    train_files = list(range(split_idx))
    train_count = process_files_parallel(train_files, train_dir, "data_split/train")
    print(f"处理了 {train_count} 个训练切片")

    # 评估文件
    print("处理评估数据...")
    eval_files = list(range(split_idx, total_files))
    eval_count = process_files_parallel(eval_files, train_dir, "data_split/eval")
    print(f"处理了 {eval_count} 个评估切片")