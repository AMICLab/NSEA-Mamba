import os
import torch
from tqdm import tqdm
from dataloader.Entropy_dataloader import PETDataset
from models.builder import EncoderDecoder
from configs.config import C
from dataloader.Entropy import DynamicEntropyModel
from torch.utils.data import DataLoader
from torchvision import transforms

def calculate_and_save_entropy(save_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dynamic_entropy_calculator = DynamicEntropyModel().to(device)

    train_dataset = PETDataset(
        data_list_path=C.test_source,
        base_path=C.base_path,
        transform=transforms.ToTensor()
    )
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=C.num_workers)

    for x_UNCORRECTED, NORMAL, UNCORRECTED_path in tqdm(train_data_loader, desc='Calculating Entropy'):
        x_UNCORRECTED = x_UNCORRECTED.to(device)  # 确保数据在正确的设备上
        x_entropy = dynamic_entropy_calculator(x_UNCORRECTED)  # 计算熵

        case_name = os.path.basename(os.path.dirname(os.path.dirname(UNCORRECTED_path[0])))
        file_name = os.path.basename(UNCORRECTED_path[0]).split('.')[0]  # 提取文件名
        save_path = os.path.join(save_dir, 'entropy_npy', case_name, f'{file_name}_entropy.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存路径存在

        # 保存熵图像，不改变其尺度
        torch.save(x_entropy.cpu(), save_path)  # 直接保存到CPU，不使用.squeeze()

if __name__ == '__main__':
    save_dir = '/home/siat/ycy/NSEA-Mamba/saved_models'
    calculate_and_save_entropy(save_dir)
