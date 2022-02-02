import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

from .data_loader import RescaleT
from .data_loader import ToTensorLab
from .data_loader import SalObjDataset

from . import utils
import cv2


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_output(image_name, pred, d_dir, d_dir_ovl):
    # overlay the importance map on the input image
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    pb_np = np.array(imo)
    pb_np = pb_np[:, :, :1].squeeze()

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    # save the importance map
    pred_fp = d_dir + imidx + '.png'
    imo.save(pred_fp)

    img_ini = Image.open(image_name).convert('RGB')
    img_imp = cv2.imread(d_dir + imidx + '.png')
    img_imp = img_imp[:, :, :1]
    img_imp = img_imp.squeeze()
    fname = os.path.join(d_dir_ovl + imidx + '_ovl' + '.png')
    utils.overlay_imp_on_img(img_ini, img_imp, fname, colormap='jet')

    return pred_fp, pb_np


def get_imp(img_name, prediction_dir, prediction_dir_ovl, visimp_model):

    # predict the importance map
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(prediction_dir_ovl, exist_ok=True)
    img_name_list = [img_name]

    # --------- dataloader -----------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

    net = visimp_model

    # --------- inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _, _, _, _, _, _, _ = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to prediction_dir folder
        pred_name, pred_np = save_output(img_name_list[i_test], pred, prediction_dir, prediction_dir_ovl)

    return pred_name, pred_np
