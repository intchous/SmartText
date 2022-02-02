import os
import torch.utils.data as data
import cv2
import math
import numpy as np
from generate_candidates import gen_boxes_multi

MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_expand(image, annotation, exp_prop=5):
    anno_w = math.floor(abs(float(annotation[3] - annotation[1])))
    anno_h = math.floor(abs(float(annotation[2] - annotation[0])))
    lx = min(annotation[0], annotation[2])
    rx = max(annotation[0], annotation[2])
    ly = min(annotation[1], annotation[3])
    ry = max(annotation[1], annotation[3])
    new_lx = max(0, int(lx - exp_prop * anno_h))
    new_ly = max(0, int(ly - exp_prop * anno_w))
    new_rx = min(image.shape[0], int(rx + exp_prop * anno_h))
    new_ry = min(image.shape[1], int(ry + exp_prop * anno_w))

    new_image = image[new_lx:new_rx, new_ly:new_ry].copy()

    new_anno_lx = lx - new_lx
    new_anno_ly = ly - new_ly
    new_anno_rx = rx - new_lx
    new_anno_ry = ry - new_ly

    return new_image, [new_anno_lx, new_anno_ly, new_anno_rx, new_anno_ry, annotation[4]]


class TransformFunctionTest_RoE(object):

    def __call__(self, image, image_size, image_fp, usr_slogan, font_fp, is_devi, visimp_model, proc_fa_dir, ratio_list,
                 text_spacing, exp_prop, grid_num, sali_coef, max_text_area_coef, min_text_area_coef, min_font_size,
                 max_font_size, font_inc_unit):

        visimp_pred_dir = proc_fa_dir + 'visimp_pred/'
        visimp_pred_dir_ovl = visimp_pred_dir
        box_dict, img_visimp = gen_boxes_multi(img_name=image_fp,
                                               visimp_pred_dir=visimp_pred_dir,
                                               visimp_pred_dir_ovl=visimp_pred_dir_ovl,
                                               visimp_model=visimp_model,
                                               usr_slogan=usr_slogan,
                                               font_fp=font_fp,
                                               base_dat_dir=proc_fa_dir,
                                               is_devi=is_devi,
                                               ratio_list=ratio_list,
                                               text_spacing=text_spacing,
                                               grid_num=grid_num,
                                               sali_coef=sali_coef,
                                               max_text_area_coef=max_text_area_coef,
                                               min_text_area_coef=min_text_area_coef,
                                               min_font_size=min_font_size,
                                               max_font_size=max_font_size,
                                               font_inc_unit=font_inc_unit)

        box_list = box_dict['new_anno_list']
        len_box_list = len(box_list)
        bboxes = []
        for ik in range(len_box_list):
            bboxes.append([
                box_list[ik][0]['xl'], box_list[ik][0]['yl'], box_list[ik][0]['xr'], box_list[ik][0]['yr'],
                box_list[ik][0]['tl_cnt']
            ])

        len_bboxes = len(bboxes)
        transformed_bboxes = {}
        transformed_bboxes['xmin'] = []
        transformed_bboxes['ymin'] = []
        transformed_bboxes['xmax'] = []
        transformed_bboxes['ymax'] = []
        source_bboxes = list()
        resized_images = []
        mx_w = 0
        mx_h = 0
        sto_image = np.array(image)

        for i in range(len_bboxes):
            image = np.array(sto_image)
            # exp_prop: expanding coefficient of the text region
            image, tmp_bbox = find_expand(image=image, annotation=bboxes[i], exp_prop=exp_prop)

            scale = float(image_size) / float(min(image.shape[:2]))
            h = round(image.shape[0] * scale / 32.0) * 32
            w = round(image.shape[1] * scale / 32.0) * 32
            resized_image = cv2.resize(image, (int(w), int(h))) / 256.0
            # img = cv2.resize(img, (crop_size, crop_size), interpolation = cv2.INTER_AREA)
            rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
            rgb_std = np.array(RGB_STD, dtype=np.float32)
            resized_image = resized_image.astype(np.float32)
            resized_image -= rgb_mean
            resized_image = resized_image / rgb_std
            if (resized_image.shape[0] > mx_h):
                mx_h = resized_image.shape[0]
            if (resized_image.shape[1] > mx_w):
                mx_w = resized_image.shape[1]

            scale_height = image.shape[0] / float(resized_image.shape[0])
            scale_width = image.shape[1] / float(resized_image.shape[1])
            resized_images.append(resized_image)

            # source_bboxes.append([round(bbox[0] * scale_height),round(bbox[1] * scale_width),round(bbox[2] * scale_height),round(bbox[3] * scale_width)])
            source_bboxes.append(
                [round(bboxes[i][0]),
                 round(bboxes[i][1]),
                 round(bboxes[i][2]),
                 round(bboxes[i][3]), bboxes[i][4]])
            transformed_bboxes['xmin'].append(tmp_bbox[1] / scale_width)
            transformed_bboxes['ymin'].append(tmp_bbox[0] / scale_height)
            transformed_bboxes['xmax'].append(tmp_bbox[3] / scale_width)
            transformed_bboxes['ymax'].append(tmp_bbox[2] / scale_height)

        len_resized_images = len(resized_images)
        for i in range(len_resized_images):
            r_itm = resized_images[i].copy()
            pre_h = r_itm.shape[0]
            pre_w = r_itm.shape[1]
            r_itm = np.pad(r_itm, ((0, mx_h - pre_h), (0, mx_w - pre_w), (0, 0)), 'constant')
            r_itm = r_itm.transpose((2, 0, 1))
            resized_images[i] = r_itm.copy()

        return resized_images, transformed_bboxes, source_bboxes, box_list


class TransformFunctionTest_RoD(object):

    def __call__(self, image, image_size, image_fp, usr_slogan, font_fp, is_devi, visimp_model, proc_fa_dir, ratio_list,
                 text_spacing, grid_num, sali_coef, max_text_area_coef, min_text_area_coef, min_font_size,
                 max_font_size, font_inc_unit):

        visimp_pred_dir = proc_fa_dir + 'visimp_pred/'
        visimp_pred_dir_ovl = visimp_pred_dir
        box_dict, img_visimp = gen_boxes_multi(img_name=image_fp,
                                               visimp_pred_dir=visimp_pred_dir,
                                               visimp_pred_dir_ovl=visimp_pred_dir_ovl,
                                               visimp_model=visimp_model,
                                               usr_slogan=usr_slogan,
                                               font_fp=font_fp,
                                               base_dat_dir=proc_fa_dir,
                                               is_devi=is_devi,
                                               ratio_list=ratio_list,
                                               text_spacing=text_spacing,
                                               grid_num=grid_num,
                                               sali_coef=sali_coef,
                                               max_text_area_coef=max_text_area_coef,
                                               min_text_area_coef=min_text_area_coef,
                                               min_font_size=min_font_size,
                                               max_font_size=max_font_size,
                                               font_inc_unit=font_inc_unit)

        box_list = box_dict['new_anno_list']
        len_box_list = len(box_list)
        bboxes = []
        for ik in range(len_box_list):
            bboxes.append([
                box_list[ik][0]['xl'], box_list[ik][0]['yl'], box_list[ik][0]['xr'], box_list[ik][0]['yr'],
                box_list[ik][0]['tl_cnt']
            ])

        transformed_bbox = {}
        transformed_bbox['xmin'] = []
        transformed_bbox['ymin'] = []
        transformed_bbox['xmax'] = []
        transformed_bbox['ymax'] = []
        source_bboxes = list()

        scale = float(image_size) / float(min(image.shape[:2]))
        h = round(image.shape[0] * scale / 32.0) * 32
        w = round(image.shape[1] * scale / 32.0) * 32
        resized_image = cv2.resize(image, (int(w), int(h))) / 256.0
        rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
        rgb_std = np.array(RGB_STD, dtype=np.float32)
        resized_image = resized_image.astype(np.float32)
        resized_image -= rgb_mean
        resized_image = resized_image / rgb_std

        scale_height = image.shape[0] / float(resized_image.shape[0])
        scale_width = image.shape[1] / float(resized_image.shape[1])

        for bbox in bboxes:
            # source_bboxes.append([round(bbox[0] * scale_height),round(bbox[1] * scale_width),round(bbox[2] * scale_height),round(bbox[3] * scale_width)])
            source_bboxes.append([round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3]), bbox[4]])
            transformed_bbox['xmin'].append(bbox[1] / scale_width)
            transformed_bbox['ymin'].append(bbox[0] / scale_height)
            transformed_bbox['xmax'].append(bbox[3] / scale_width)
            transformed_bbox['ymax'].append(bbox[2] / scale_height)

        resized_image = resized_image.transpose((2, 0, 1))
        return resized_image, transformed_bbox, source_bboxes, box_list


class setup_test_dataset(data.Dataset):

    def __init__(self,
                 usr_slogan,
                 font_fp,
                 visimp_model,
                 proc_fa_dir,
                 is_devi=False,
                 image_size=256.0,
                 dataset_dir='testsetDir',
                 model_type='RoD',
                 ratio_list=[1, 1, 1, 1, 1],
                 text_spacing=20,
                 exp_prop=5,
                 grid_num=120,
                 sali_coef=2.6,
                 max_text_area_coef=17,
                 min_text_area_coef=7,
                 min_font_size=10,
                 max_font_size=500,
                 font_inc_unit=5):
        self.image_size = float(image_size)
        self.dataset_dir = dataset_dir
        image_lists = os.listdir(self.dataset_dir)
        self._imgpath = list()
        self._annopath = list()
        for image in image_lists:
            if (is_image_file(image)):
                self._imgpath.append(os.path.join(self.dataset_dir, image))

        self.model_type = model_type
        if (self.model_type == 'RoE'):
            self.transform = TransformFunctionTest_RoE()
        else:
            self.transform = TransformFunctionTest_RoD()
        self.usr_slogan = usr_slogan
        self.font_fp = font_fp
        self.is_devi = is_devi
        self.visimp_model = visimp_model
        self.proc_fa_dir = proc_fa_dir
        self.ratio_list = ratio_list
        self.text_spacing = text_spacing
        self.exp_prop = exp_prop
        self.grid_num = grid_num
        self.sali_coef = sali_coef
        self.max_text_area_coef = max_text_area_coef
        self.min_text_area_coef = min_text_area_coef
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.font_inc_unit = font_inc_unit

    def __getitem__(self, idx):
        image = cv2.imread(self._imgpath[idx])
        # to rgb
        image = image[:, :, (2, 1, 0)]

        if (self.model_type == 'RoE'):
            if self.transform:
                resized_images, transformed_bboxes, source_bboxes, box_list = self.transform(
                    image, self.image_size, self._imgpath[idx], self.usr_slogan, self.font_fp, self.is_devi,
                    self.visimp_model, self.proc_fa_dir, self.ratio_list, self.text_spacing, self.exp_prop,
                    self.grid_num, self.sali_coef, self.max_text_area_coef, self.min_text_area_coef, self.min_font_size,
                    self.max_font_size, self.font_inc_unit)

            sample = {
                'imgpath': self._imgpath[idx],
                'image': image,
                'resized_images': resized_images,
                'tbboxes': transformed_bboxes,
                'sourceboxes': source_bboxes,
                'box_list': box_list
            }
        else:
            if self.transform:
                resized_image, transformed_bbox, source_bboxes, box_list = self.transform(
                    image, self.image_size, self._imgpath[idx], self.usr_slogan, self.font_fp, self.is_devi,
                    self.visimp_model, self.proc_fa_dir, self.ratio_list, self.text_spacing, self.grid_num,
                    self.sali_coef, self.max_text_area_coef, self.min_text_area_coef, self.min_font_size,
                    self.max_font_size, self.font_inc_unit)

            sample = {
                'imgpath': self._imgpath[idx],
                'image': image,
                'resized_images': resized_image,
                'tbboxes': transformed_bbox,
                'sourceboxes': source_bboxes,
                'box_list': box_list
            }

        return sample

    def __len__(self):
        return len(self._imgpath)
