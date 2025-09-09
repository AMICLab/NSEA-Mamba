import torchvision
import random
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader.dataloader import PETDataset
from pydicom import dcmread, write_file
import numpy as np
import csv
from models.builder import EncoderDecoder
from configs.config import C  # 导入配置文件
import torch.nn.functional as F
from pytorch_msssim import ssim
from pydicom import dcmread, dcmwrite
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataloader.Entropy import DynamicEntropyModel

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# 设置随机种子
seed = 1777777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# 定义保存案例图像的函数
def save_case_images(val_data_loader, model, save_dir, epoch):
    npy_save_dir = os.path.join(save_dir, 'npy_output', f'val_epoch_{epoch}')
    os.makedirs(npy_save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        case_file_indices = {} # 初始化文件编号
        for x_UNCORRECTED, NORMAL, UNCORRECTED_path, x_entropy in val_data_loader:
            case_name = os.path.basename(os.path.dirname(os.path.dirname(UNCORRECTED_path[0])))
            case_npy_save_dir = os.path.join(npy_save_dir, case_name)
            os.makedirs(case_npy_save_dir, exist_ok=True)

            # 获取当前病例的文件编号，如果没有则初始化为1
            if case_name not in case_file_indices:
                case_file_indices[case_name] = 1
            case_file_index = case_file_indices[case_name]

            x_UNCORRECTED, NORMAL, x_entropy = x_UNCORRECTED.to(device), NORMAL.to(device), x_entropy.to(device)
            x_entropy = F.threshold(x_entropy, 0.06, 0, False)
            outputs = model(x_UNCORRECTED, x_entropy)

            if outputs.shape != NORMAL.shape:
                print(f"Warning: Output shape {outputs.shape} does not match NORMAL shape {NORMAL.shape}")
                continue

            x_entropy = x_entropy.squeeze().cpu().detach().numpy()
            x_UNCORRECTED = x_UNCORRECTED.squeeze().cpu().detach().numpy()
            outputs = outputs.squeeze().cpu().detach().numpy()
            NORMAL = NORMAL.squeeze().cpu().detach().numpy()

            npy_x_entropy_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_entropy.npy')
            np.save(npy_x_entropy_save_path, x_entropy)
            npy_x_UNCORRECTED_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_input.npy')
            np.save(npy_x_UNCORRECTED_save_path, x_UNCORRECTED)
            npy_outputs_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_output.npy')
            np.save(npy_outputs_save_path, outputs)
            npy_NORMAL_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_NORMAL.npy')
            np.save(npy_NORMAL_save_path, NORMAL)

            # 递增当前病例的文件编号
            case_file_indices[case_name] += 1

def calculate_psnr(mse):
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1/ torch.sqrt(mse))


def calculate_ssim(NORMAL, outputs):
    return ssim(NORMAL, outputs, data_range=1, size_average=True)


def calculate_mae(NORMAL, outputs):
    return torch.mean(torch.abs(NORMAL - outputs)).item()


def main():
    save_dir = '/home/siat/ycy/NSEA-Mamba/saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    train_csv_file_path = os.path.join(save_dir, 'training.csv')
    val_csv_file_path = os.path.join(save_dir, 'validation.csv')

    with open(train_csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'MSE_loss', 'SSIM', 'MAE', 'PSNR'])

    with open(val_csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'MSE_loss', 'SSIM', 'MAE', 'PSNR'])

    train_dataset = PETDataset(
        data_list_path=C.test_source,
        base_path=C.base_path,
        entropy_base_path=save_dir,  # 传递熵图像的基础路径
        transform=transforms.ToTensor()
    )
    train_data_loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)

    val_dataset = PETDataset(
        data_list_path=C.eval_source,
        base_path=C.base_path,
        entropy_base_path=save_dir,  # 传递熵图像的基础路径
        transform=transforms.ToTensor()
    )
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=C.num_workers)

    model = EncoderDecoder(cfg=C)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    print('Model and data loaders are ready.')

    # 加载第200个epoch的模型权重
    # model.load_state_dict(
    #        torch.load(os.path.join(save_dir, 'model_epoch_280.pth'), map_location=device, weights_only=True))
    # start_epoch = 281  # 从第201个epoch开始训练

    for epoch in range(1, C.nepochs + 1):
        model.train()
        train_loss = 0.0
        train_ssim_total = 0.0
        train_mae = 0.0
        train_batch_num = 0

        for x_UNCORRECTED, NORMAL, UNCORRECTED_path, x_entropy in tqdm(train_data_loader, desc='Training', leave=True):
            x_UNCORRECTED, NORMAL, x_entropy = x_UNCORRECTED.to(device), NORMAL.to(device), x_entropy.to(device)
            x_entropy = F.threshold(x_entropy, 0.06, 0, False)
            optimizer.zero_grad()
            outputs = model(x_UNCORRECTED, x_entropy)
            loss = criterion(NORMAL.float(), outputs.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                ssim_trainue = calculate_ssim(NORMAL, outputs)
                train_ssim_total += ssim_trainue

                train_mae += calculate_mae(NORMAL, outputs)

                train_batch_num += 1

        scheduler.step()

        average_loss = train_loss / train_batch_num
        average_ssim = train_ssim_total / train_batch_num
        average_mae = train_mae / train_batch_num
        average_psnr = calculate_psnr(torch.tensor(average_loss))

        print(
            f'train MSE_loss: {average_loss:.4f}, SSIM: {average_ssim:.4f}, MAE: {average_mae:.4f}, PSNR: {average_psnr:.4f}')

        with open(train_csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch, average_loss, average_ssim, average_mae, average_psnr])

        model.eval()
        val_loss = 0.0
        val_ssim_total = 0.0
        val_mae = 0.0
        val_batch_num = 0

        with torch.no_grad():
            for x_UNCORRECTED, NORMAL, UNCORRECTED_path, x_entropy in tqdm(val_data_loader, desc='valing', leave=True):
                x_UNCORRECTED, NORMAL, x_entropy = x_UNCORRECTED.to(device), NORMAL.to(device), x_entropy.to(device)
                x_entropy = F.threshold(x_entropy, 0.06, 0, False)
                outputs = model(x_UNCORRECTED, x_entropy)
                loss = criterion(NORMAL.float(), outputs.float())
                val_loss += loss.item()

                ssim_value = calculate_ssim(NORMAL, outputs)
                val_ssim_total += ssim_value

                val_mae += calculate_mae(NORMAL, outputs)

                val_batch_num += 1

        average_val_loss = val_loss / val_batch_num
        average_val_ssim = val_ssim_total / val_batch_num
        average_val_mae = val_mae / val_batch_num
        average_val_psnr = calculate_psnr(torch.tensor(average_val_loss))

        print(
            f'val MSE_loss: {average_val_loss:.4f}, SSIM: {average_val_ssim:.4f}, MAE: {average_val_mae:.4f}, PSNR: {average_val_psnr:.4f}')

        with open(val_csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch, average_val_loss, average_val_ssim, average_val_mae, average_val_psnr])

        if epoch % C.checkpoint_step == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
            save_case_images(val_data_loader, model, save_dir, epoch)


if __name__ == '__main__':
    main()
