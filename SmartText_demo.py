from smtModel import build_smt_model
from smtDataset import setup_test_dataset
import os
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import time
import math

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
from datetime import date

from BASNet.model import BASNet
from cal_color import cal_best_color, RGB_to_Hex
import option
from option import sv_json

import warnings
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt)
opt = option.dict_to_nonedict(opt)

today = date.today().strftime("%Y%m%d")
proc_fa_dir = opt['res_dir'] + opt['model_type'] + '_' + today + '/'
output_dir = proc_fa_dir + 'res/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if torch.cuda.is_available():
    if opt['cuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not opt['cuda']:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# --------- model define ---------
print("...load SMTNet...")
smt_net = build_smt_model(scale='multi', alignsize=9, reddim=8, loadweight=False, model='shufflenetv2', downsample=4)
smt_net.load_state_dict(torch.load(opt['smt_model']))
smt_net.eval()

print("...load BASNet...")
visimp_net = BASNet(3, 1)
visimp_net.load_state_dict(torch.load(opt['visimp_model']))
visimp_net.eval()

if opt['cuda']:
    smt_net = torch.nn.DataParallel(smt_net, device_ids=[0])
    cudnn.benchmark = True
    smt_net = smt_net.cuda()
    visimp_net = visimp_net.cuda()

dataset = setup_test_dataset(usr_slogan=opt['usr_slogan'],
                             font_fp=opt['font_fp'],
                             visimp_model=visimp_net,
                             proc_fa_dir=proc_fa_dir,
                             is_devi=opt['is_devi'],
                             dataset_dir=opt['input_dir'],
                             model_type=opt['model_type'],
                             ratio_list=opt['ratio_list'],
                             text_spacing=opt['text_spacing'],
                             exp_prop=opt['exp_prop'],
                             grid_num=opt['grid_num'],
                             sali_coef=opt['sali_coef'],
                             max_text_area_coef=opt['max_text_area_coef'],
                             min_text_area_coef=opt['min_text_area_coef'],
                             min_font_size=opt['min_font_size'],
                             max_font_size=opt['max_font_size'],
                             font_inc_unit=opt['font_inc_unit'])


def naive_collate(batch):
    return batch[0]


data_loader = data.DataLoader(dataset,
                              opt['batch_size'],
                              num_workers=opt['num_workers'],
                              collate_fn=naive_collate,
                              shuffle=False)


def draw_text_imgpath(imgpath, fsz, fontstr, top_box, res_text_loc, text_spacing, fontcolor, font_loc):
    pil_im = Image.open(imgpath)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(font_loc, fsz, encoding="utf-8")
    draw.text((top_box[1], top_box[0]), fontstr, fontcolor, font=font, spacing=text_spacing)
    pil_im.save(res_text_loc)


def draw_text_cont(pil_im, draw, fsz, fontstr, top_box, res_text_loc, text_spacing, fontcolor, font_loc):
    font = ImageFont.truetype(font_loc, fsz, encoding="utf-8")
    draw.text((top_box[1], top_box[0]), fontstr, fontcolor, font=font, spacing=text_spacing)
    # pil_im.save(res_text_loc)


def output_file_name(input_path, sc, idx, dataset_name='SMT', R_type='RoD'):
    name = os.path.basename(input_path)
    segs = name.split('.')
    assert len(segs) >= 2
    return '%s_%s_%s_%d_%s.%s' % ('.'.join(segs[:-1]), dataset_name, R_type, idx, sc, segs[-1])


def test_sep(st_id, ed_id, resized_images, bboxs):
    roi = []
    st_flg = True
    i_cnt = 0
    for idx in range(st_id, ed_id):
        if (st_flg == True):
            in_imgs = torch.unsqueeze(torch.as_tensor(resized_images[idx]), 0)
            st_flg = False

        else:
            tp_img = torch.unsqueeze(torch.as_tensor(resized_images[idx]), 0)
            in_imgs = torch.cat((in_imgs, tp_img), 0)

        roi.append((i_cnt, bboxs['xmin'][idx], bboxs['ymin'][idx], bboxs['xmax'][idx], bboxs['ymax'][idx]))
        i_cnt += 1

    if opt['cuda']:
        in_imgs = Variable(in_imgs.cuda())
        roi = Variable(torch.Tensor(roi))
    else:
        in_imgs = Variable(in_imgs)
        roi = Variable(roi)

    out = smt_net(in_imgs, roi)
    return out


def test():

    for id, sample in enumerate(data_loader):
        st_time = time.time()
        imgpath = sample['imgpath']
        bboxes = sample['sourceboxes']
        resized_images = sample['resized_images']
        tbboxes = sample['tbboxes']
        box_list = sample['box_list']

        len_tbboxes = len(tbboxes['xmin'])
        if (len_tbboxes == 0):
            continue

        if (opt['model_type'] == 'RoE'):
            bat_sz = 16
            te_cnt = math.ceil(len_tbboxes * 1.0 / bat_sz)
            st = 0
            for ite in range(te_cnt):
                ed = min(st + bat_sz, len_tbboxes)
                sep_out = test_sep(st, ed, resized_images, tbboxes)
                if (ite == 0):
                    cat_out = torch.Tensor(sep_out)
                else:
                    cat_out = torch.cat((cat_out, sep_out), 0)

                st = st + bat_sz

            out = torch.Tensor(cat_out)

        else:
            roi = []
            for idx in range(0, len(tbboxes['xmin'])):
                roi.append((0, tbboxes['xmin'][idx], tbboxes['ymin'][idx], tbboxes['xmax'][idx], tbboxes['ymax'][idx]))

            resized_image = torch.unsqueeze(torch.as_tensor(resized_images), 0)
            if opt['cuda']:
                resized_image = Variable(resized_image.cuda())
                roi = Variable(torch.Tensor(roi))
            else:
                resized_image = Variable(resized_image)
                roi = Variable(torch.Tensor(roi))
            out = smt_net(resized_image, roi)

        print('len_out =', len(out))
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)

        #---------------------------------
        # find json file in box_dir
        base_dat_dir = proc_fa_dir
        base_box_dir = base_dat_dir + 'box_dir' + '/'
        img_name = imgpath.split('/')[-1]
        imgpre, _ = os.path.splitext(img_name)
        box_loc = base_box_dir + imgpre + '/' + imgpre + '.json'
        with open(box_loc, encoding="utf-8") as f:
            box_data = json.load(f)
        #---------------------------------

        impre = imgpath.split('/')[-1].split('.')[0]
        len_bboxes = len(bboxes)
        for i in range(len_bboxes):
            tmp_sc = out[i].cpu().data.numpy().squeeze()
            tmp_sc = tmp_sc * MOS_STD + MOS_MEAN
            box_data[i][0]['score'] = tmp_sc

        sv_json(box_data, box_loc)

        candi_res = min(opt['candi_res'], len(id_out))
        for id in range(0, candi_res):
            top_box = bboxes[id_out[id]]
            tmp_sc = str(box_data[id_out[id]][0]['score'])

            # draw each res in sep dir
            res_sep_dir = output_dir + impre + '/'
            os.makedirs(res_sep_dir, exist_ok=True)
            res_text_loc = os.path.join(
                res_sep_dir,
                output_file_name(input_path=imgpath,
                                 sc=tmp_sc,
                                 idx=id + 1,
                                 dataset_name=opt['dataset_name'],
                                 R_type=opt['model_type']))

            pil_im = Image.open(imgpath)
            draw = ImageDraw.Draw(pil_im)
            tl_cnt = box_list[id_out[id]][0]['tl_cnt']

            if (id == 0):
                # im_np = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR) # h, w, c
                im_np = np.array(pil_im) # h, w, c
                xl = box_list[id_out[id]][0]['xl']
                xr = box_list[id_out[id]][0]['xr']
                yl = box_list[id_out[id]][0]['yl']
                yr = box_list[id_out[id]][0]['yr']
                im_crop_np = im_np[xl:xr, yl:yr]

                # select text color
                color_candi = cal_best_color(im_np, im_crop_np, opt['contrast_threshold'])
                fontcolor = RGB_to_Hex(color_candi[0]['color'])
                print("fontcolor = " + fontcolor)

            for tx in range(1, tl_cnt + 1):
                fsz = box_list[id_out[id]][tx]['fsz']
                fontstr = box_list[id_out[id]][tx]['fontstr']
                top_box = [box_list[id_out[id]][tx]['xl'], box_list[id_out[id]][tx]['yl']]

                draw_text_cont(pil_im,
                               draw,
                               fsz=fsz,
                               fontstr=fontstr,
                               top_box=top_box,
                               res_text_loc=res_text_loc,
                               text_spacing=opt['text_spacing'],
                               fontcolor=fontcolor,
                               font_loc=opt['font_fp'])

            pil_im.save(res_text_loc)
            # draw best res
            if (id == 0):
                res_text_loc = os.path.join(
                    output_dir,
                    output_file_name(input_path=imgpath,
                                     sc=tmp_sc,
                                     idx=id + 1,
                                     dataset_name=opt['dataset_name'],
                                     R_type=opt['model_type']))
                pil_im.save(res_text_loc)

        ed_time = time.time()
        print('timer: %.4f sec.' % (ed_time - st_time))


if __name__ == '__main__':
    test()
