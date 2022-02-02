import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from option import sv_json
import util_cal as uc
from BASNet.basnet_test import get_imp


def gen_boxes_multi(img_name,
                    visimp_pred_dir,
                    visimp_pred_dir_ovl,
                    visimp_model,
                    usr_slogan,
                    font_fp,
                    base_dat_dir,
                    is_devi=False,
                    ratio_list=[1, 1, 1, 1, 1],
                    text_spacing=20,
                    grid_num=120,
                    sali_coef=2.6,
                    max_text_area_coef=17,
                    min_text_area_coef=7,
                    min_font_size=10,
                    max_font_size=500,
                    font_inc_unit=5):
    base_box_dir = base_dat_dir + 'box_dir' + '/'
    im_wh_name = img_name.split('/')[-1]
    imgpre, imgext = os.path.splitext(im_wh_name)
    _, ini_visimp = get_imp(img_name=img_name,
                            prediction_dir=visimp_pred_dir,
                            prediction_dir_ovl=visimp_pred_dir_ovl,
                            visimp_model=visimp_model)

    rescaled = np.array(ini_visimp)
    rerow = len(rescaled)
    recol = len(rescaled[0])

    # grid artition
    grid_rsz = int(max(rerow, rerow) * 1.0 / grid_num)
    if (grid_rsz % 2 == 1):
        grid_rsz = grid_rsz - 1
    grid_csz = grid_rsz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.FloatTensor(rescaled).to(device)
    h, w = x.shape
    x = F.avg_pool2d(x.view(1, 1, h, w), kernel_size=grid_rsz)
    x = x.cpu().numpy()
    crop_mat = np.squeeze(x * grid_rsz * grid_csz)

    crop_row_num = crop_mat.shape[0]
    crop_col_num = crop_mat.shape[1]
    matrix1D = crop_mat.flatten()
    matrixcal = [[0.0 for i in range(crop_col_num)] for i in range(crop_row_num)]
    matrix1D = np.sort(matrix1D)[::-1]

    # the larger sali_coef, the smaller area defined as important of the image
    Kth = (int)(crop_row_num * crop_col_num / sali_coef)
    tmpval = matrix1D[Kth]

    INF = float(1000000007)
    for i in range(crop_row_num):
        for j in range(crop_col_num):
            if (crop_mat[i][j] > tmpval):
                matrixcal[i][j] = INF
            elif (i <= 3 or j <= 3 or i >= crop_row_num - 4 or j >= crop_col_num - 4):
                matrixcal[i][j] = INF
            else:
                matrixcal[i][j] = crop_mat[i][j]

    ini_tprob_map = np.array(uc.cal_imp_conv(crop_row_num, crop_col_num, crop_mat, matrixcal, matrix1D, INF))

    min_text_area = rerow * recol / max_text_area_coef
    max_text_area = rerow * recol / min_text_area_coef

    slogan_list = usr_slogan.split('\n')
    len_slogan_list = len(slogan_list)
    if (len_slogan_list > len(ratio_list)):
        for i in range(len_slogan_list - len(ratio_list)):
            ratio_list.append(1)

    image_name = img_name
    rect_im = Image.open(image_name)
    draw_rect = ImageDraw.Draw(rect_im)

    box_dir = base_box_dir + imgpre + '/'
    os.makedirs(box_dir, exist_ok=True)

    fsz = min_font_size
    fsz_intv = font_inc_unit
    scnt = 0
    anno_list = []
    now_idx = 0
    while fsz <= max_font_size:
        pil_im = Image.open(image_name)
        draw = ImageDraw.Draw(pil_im)

        txarea_x = -text_spacing
        txarea_y = 0.0
        for tli in range(len_slogan_list):
            tli_fsz = int(fsz * ratio_list[tli])
            font = ImageFont.truetype(font_fp, tli_fsz, encoding="utf-8")
            fontstr = slogan_list[tli]
            tli_txsz = draw.textsize(fontstr, font=font, spacing=text_spacing)
            txarea_x = txarea_x + tli_txsz[1] + text_spacing
            txarea_y = max(txarea_y, tli_txsz[0])

        txarea = txarea_x * txarea_y
        txsz = [txarea_y, txarea_x]
        if ((txarea > max_text_area) or (txarea < min_text_area) or (txarea_y >= recol) or (txarea_x >= rerow)):
            fsz += fsz_intv
            continue

        Kth_rect = 1
        st = uc.get_top_k_submatrix(ini_tprob_map, ((int)(txsz[1] / grid_rsz), (int)(txsz[0] / grid_csz)),
                                    Kth_rect,
                                    desc=False)

        for kth in range(Kth_rect):
            stx = st[kth].rx * grid_rsz
            sty = st[kth].cy * grid_csz
            if ((stx >= rerow) or (stx + txsz[1] >= rerow) or (sty >= recol) or (sty + txsz[0] >= recol)):
                continue

            stcol = sty
            strow = stx
            edcol = sty + txsz[0]
            edrow = stx + txsz[1]
            scnt += 1
            tmp_anno_list = []
            tmp_anno_list.append({
                'idx': now_idx,
                'xl': strow,
                'yl': stcol,
                'xr': edrow,
                'yr': edcol,
                'tl_cnt': len_slogan_list
            })
            now_idx += 1

            stcol = sty
            strow = stx
            for tli in range(len_slogan_list):
                tli_fsz = int(fsz * ratio_list[tli])
                font = ImageFont.truetype(font_fp, tli_fsz, encoding="utf-8")
                fontstr = slogan_list[tli]
                tli_txsz = draw.textsize(fontstr, font=font, spacing=text_spacing)
                edcol = stcol + tli_txsz[0]
                edrow = strow + tli_txsz[1]
                tmp_anno_list.append({
                    'xl': strow,
                    'yl': stcol,
                    'xr': edrow,
                    'yr': edcol,
                    'fsz': tli_fsz,
                    'fontstr': fontstr
                })
                strow = strow + tli_txsz[1] + text_spacing

            anno_list.append(tmp_anno_list)

        fsz += fsz_intv

    new_anno_list = []
    len_anno_list = len(anno_list)
    if (is_devi):
        devi_direc = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    else:
        devi_direc = []

    # deviation unit
    devi_unit = grid_rsz * 10
    each_box_num = len(devi_direc)

    for ai in range(len_anno_list):
        new_anno_list.append(anno_list[ai])

        for gen_i in range(each_box_num):
            new_xl = anno_list[ai][0]['xl'] + devi_direc[gen_i][0] * devi_unit
            new_yl = anno_list[ai][0]['yl'] + devi_direc[gen_i][1] * devi_unit
            new_xr = new_xl + abs(anno_list[ai][0]['xr'] - anno_list[ai][0]['xl'])
            new_yr = new_yl + abs(anno_list[ai][0]['yr'] - anno_list[ai][0]['yl'])
            if (new_xl < 0 or (new_xl >= rerow) or (new_yl < 0) or (new_yl >= recol) or new_xr < 0 or (new_xr >= rerow)
                    or (new_yr < 0) or (new_yr >= recol)):
                continue

            tmp_new_anno_list = []
            tmp_new_anno_list.append({
                'idx': now_idx,
                'xl': new_xl,
                'yl': new_yl,
                'xr': new_xr,
                'yr': new_yr,
                'tl_cnt': anno_list[ai][0]['tl_cnt']
            })
            now_idx += 1

            for tli in range(1, anno_list[ai][0]['tl_cnt'] + 1):
                tli_fsz = anno_list[ai][tli]['fsz']
                fontstr = anno_list[ai][tli]['fontstr']
                strow = anno_list[ai][tli]['xl'] + (new_xl - anno_list[ai][0]['xl'])
                stcol = anno_list[ai][tli]['yl'] + (new_yl - anno_list[ai][0]['yl'])
                edrow = anno_list[ai][tli]['xr'] + (new_xr - anno_list[ai][0]['xr'])
                edcol = anno_list[ai][tli]['yr'] + (new_yr - anno_list[ai][0]['yr'])
                tmp_new_anno_list.append({
                    'xl': strow,
                    'yl': stcol,
                    'xr': edrow,
                    'yr': edcol,
                    'fsz': tli_fsz,
                    'fontstr': fontstr
                })

            new_anno_list.append(tmp_new_anno_list)

    sv_json(new_anno_list, box_dir + imgpre + '.json')

    anno_dict = {
        'img_name': img_name,
        'usr_slogan': usr_slogan,
        'font_loc': font_fp,
        'scnt': scnt,
        'now_idx': now_idx,
        'new_anno_list': new_anno_list
    }

    return anno_dict, ini_visimp
