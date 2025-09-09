import pydicom
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from torch.nn.functional import interpolate
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as transforms

# 设置pydicom使用GDCM解码器
pydicom.config.dimse_protocol = 'GDCM'

def dicom_hhmmss(t: object) -> object:
    t = str(t)
    if len(t) == 5:
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
    dcm_tag = str(dcm.SeriesTime)+'\n'+str(dcm.AcquisitionTime)+'\n'+str(dcm.PatientWeight)+'\n'+RadiopharmaceuticalStartTime+'\n'
    dcm_tag = dcm_tag+RadionuclideTotalDose+'\n'+RadionuclideHalfLife+'\n'+str(dcm.RescaleSlope)+'\n'+str(dcm.RescaleIntercept)
    dcm_tag = dcm_tag.split('\n')

    norm = False
    [ST, AT, PW, RST, RTD, RHL, RS, RI] = dcm_tag
    decay_time = dicom_hhmmss(ST) - dicom_hhmmss(RST)
    decay_dose = float(RTD) * pow(2, -float(decay_time) / float(RHL))
    SUVbwScaleFactor = (1000 * float(PW)) / decay_dose

    GT_SUV = dcm.pixel_array * SUVbwScaleFactor
    dcm1 = dcmread(path_img)
    img_SUV = dcm1.pixel_array * SUVbwScaleFactor

    return img_SUV,GT_SUV

class PETDataset(Dataset):
    def __init__(self, data_list_path, base_path, entropy_base_path, transform=None):
        self.data_list_path = data_list_path
        self.base_path = base_path
        self.entropy_base_path = entropy_base_path
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

        if image_suv_nd_UNCORRECTED.ndim == 2:
            image_suv_nd_UNCORRECTED = image_suv_nd_UNCORRECTED[np.newaxis, :, :]

        image_suv_nd_UNCORRECTED = F.interpolate(torch.from_numpy(image_suv_nd_UNCORRECTED).unsqueeze(0),
                                                 size=(192, 192), mode='area')

        if NORMAL_suv_nd is not None:
            if NORMAL_suv_nd.ndim == 2:
                NORMAL_suv_nd = NORMAL_suv_nd[np.newaxis, :, :]
            NORMAL_suv_nd = F.interpolate(torch.from_numpy(NORMAL_suv_nd).unsqueeze(0), size=(192, 192),
                                          mode='area')

        x_UNCORRECTED = image_suv_nd_UNCORRECTED.squeeze(0)
        NORMAL = NORMAL_suv_nd.squeeze(0) if NORMAL_suv_nd is not None else None

        # Load entropy image
        case_name = os.path.basename(os.path.dirname(os.path.dirname(UNCORRECTED_path)))
        file_name = os.path.basename(UNCORRECTED_path).split('.')[0]
        x_entropy = os.path.join(self.entropy_base_path, 'entropy_npy', case_name, f'{file_name}_entropy.pt')
        # 修改这里：使用 'cpu' 作为 map_location，并在之后将 tensor 移动到 GPU（如果需要）
        x_entropy = torch.load(x_entropy, map_location=torch.device('cpu'), weights_only=True).squeeze(0)

        return x_UNCORRECTED, NORMAL, UNCORRECTED_path, x_entropy

    def __len__(self):
        return len(self.data_list)
