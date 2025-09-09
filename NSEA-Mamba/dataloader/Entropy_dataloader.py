import pydicom
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from torch.nn.functional import interpolate
from torch.nn import functional as F

# 设置pydicom使用GDCM解码器
pydicom.config.dimse_protocol = 'GDCM'

def dicom_hhmmss(t: object) -> object:    # dicom存时间格式:小时、分钟、秒(每个占两位),这里转化回秒
    t = str(t)
    if len(t) == 5:     # 有些提取时漏了个0，小时的位置只有一位，这里把零补上
        t = '0'+t
    h_t = float(t[0:2])
    m_t = float(t[2:4])
    s_t = float(t[4:6])
    return h_t*3600+m_t*60+s_t

def get_suv(path_img,path_GT):
    dcm=dcmread(path_GT)
    RadiopharmaceuticalInformationSequence = dcm.RadiopharmaceuticalInformationSequence[0]
    RadiopharmaceuticalStartTime = str(RadiopharmaceuticalInformationSequence['RadiopharmaceuticalStartTime'].value)
    RadionuclideTotalDose = str(RadiopharmaceuticalInformationSequence['RadionuclideTotalDose'].value)
    RadionuclideHalfLife = str(RadiopharmaceuticalInformationSequence['RadionuclideHalfLife'].value)
    ##放在一起
    dcm_tag = str(dcm.SeriesTime)+'\n'+str(dcm.AcquisitionTime)+'\n'+str(dcm.PatientWeight)+'\n'+RadiopharmaceuticalStartTime+'\n'
    dcm_tag = dcm_tag+RadionuclideTotalDose+'\n'+RadionuclideHalfLife+'\n'+str(dcm.RescaleSlope)+'\n'+str(dcm.RescaleIntercept)
    dcm_tag = dcm_tag.split('\n')

    norm = False
    [ST, AT, PW, RST, RTD, RHL, RS, RI] = dcm_tag  # AT基本等于ST,RI一般是0
    decay_time = dicom_hhmmss(ST) - dicom_hhmmss(RST)
    decay_dose = float(RTD) * pow(2, -float(decay_time) / float(RHL))
    SUVbwScaleFactor = (1000 * float(PW)) / decay_dose

    # if norm:
    # PET_SUV = (PET * np.array(RS).astype('float') + np.array(RI).astype('float')) * SUVbwScaleFactor  # 标准公式做法
    # else:
    GT_SUV = dcm.pixel_array * SUVbwScaleFactor  # 非标准做法，但和软件得到的SUV较为一致
    dcm1 = dcmread(path_img)
    img_SUV = dcm1.pixel_array * SUVbwScaleFactor

    return img_SUV,GT_SUV


class PETDataset(Dataset):
    def __init__(self, data_list_path, base_path, transform=None):
        self.data_list_path = data_list_path
        self.base_path = base_path
        self.transform = transform
        self.data_list = self._read_data_list()

    def _read_data_list(self):
        data_list = []
        with open(self.data_list_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) == 3:
                    UNCORRECTED_path, _, NORMAL_path = parts
                    UNCORRECTED_path = UNCORRECTED_path.strip()
                    NORMAL_path = NORMAL_path.strip()
                    # 将相对路径转换为绝对路径
                    UNCORRECTED_path = os.path.join(self.base_path, UNCORRECTED_path)
                    NORMAL_path = os.path.join(self.base_path, NORMAL_path)
                    data_list.append((UNCORRECTED_path, NORMAL_path))
                else:
                    print(f"Skipping invalid line with too many or too few elements: {line}")
        return data_list

    def __getitem__(self, index):
        UNCORRECTED_path, NORMAL_path = self.data_list[index]

        try:
            image_suv_UNCORRECTED, NORMAL_suv = get_suv(UNCORRECTED_path, NORMAL_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error processing files {UNCORRECTED_path}, {NORMAL_path}: {e}")

        image_suv_nd_UNCORRECTED = np.array(image_suv_UNCORRECTED, dtype=np.float32)
        NORMAL_suv_nd = np.array(NORMAL_suv, dtype=np.float32) if NORMAL_suv is not None else None

        # Ensure the image is a 3D array with shape (C, H, W)
        if image_suv_nd_UNCORRECTED.ndim == 2:
            image_suv_nd_UNCORRECTED = image_suv_nd_UNCORRECTED[np.newaxis, :, :]  # Add channel dimension

        # Downsample images to 180x180
        image_suv_nd_UNCORRECTED = F.interpolate(torch.from_numpy(image_suv_nd_UNCORRECTED).unsqueeze(0),
                                                 size=(192, 192), mode='area')

        if NORMAL_suv_nd is not None:
            if NORMAL_suv_nd.ndim == 2:
                NORMAL_suv_nd = NORMAL_suv_nd[np.newaxis, :, :]  # Add channel dimension
            NORMAL_suv_nd = F.interpolate(torch.from_numpy(NORMAL_suv_nd).unsqueeze(0), size=(192, 192),
                                          mode='area')


        x_UNCORRECTED = image_suv_nd_UNCORRECTED.squeeze(0)  # Remove the extra dimension added for interpolation
        NORMAL = NORMAL_suv_nd.squeeze(0) if NORMAL_suv_nd is not None else None

        # print(f'size: {x_UNCORRECTED.size()}')

        return x_UNCORRECTED, NORMAL, UNCORRECTED_path



    def __len__(self):
        return len(self.data_list)
