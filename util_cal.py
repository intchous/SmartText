# encoding:utf-8
import numpy as np
import torch
import torch.nn.functional as F


class candik(object):

    def __init__(self, val, rx, cy):
        self.val = val
        self.rx = rx
        self.cy = cy


def takeVal(candik):
    return candik.val


def bb_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)


def get_top_k_submatrix(matrix, kernel_size, k, desc=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.FloatTensor(matrix).to(device)
    h, w = x.shape
    x = F.avg_pool2d(x.view(1, 1, h, w), kernel_size=kernel_size, stride=(1, 1))
    nh, nw = h - kernel_size[0] + 1, w - kernel_size[1] + 1
    x = x.view(nh, nw) * kernel_size[0] * kernel_size[1]
    x = x.cpu().numpy()
    idx = np.dstack(np.unravel_index(np.argsort(x.ravel()), (nh, nw)))[0]
    idx = idx[::-1] if desc else idx

    top_k = []
    for px, py in idx:
        if len(top_k) >= k:
            break
        conflict = False
        for qx, qy, v in top_k:
            if bb_intersection((px, py, px + kernel_size[0], py + kernel_size[1]),
                               (qx, qy, qx + kernel_size[0], qy + kernel_size[1])):
                conflict = True
                break
        if not conflict:
            # top_k.append((px, py, x[px][py]))
            top_k.append(candik(x[px][py], px, py))
    return top_k


def cal_imp_conv(n, m, matrix, matrix_cal, matrix1D, INF):
    comp_flg = False
    cal_num = 0
    fcnt = 0
    while (comp_flg == False):
        fcnt += 1
        for i in range(n):
            for j in range(m):
                tmp_ave = matrix_cal[i][j]
                if ((i - 1) >= 0 and (i - 1) < n and (j - 1) >= 0 and (j - 1) < m):
                    tmp_ave += matrix_cal[i - 1][j - 1]
                else:
                    tmp_ave += INF
                if ((i - 1) >= 0 and (i - 1) < n and j >= 0 and j < m):
                    tmp_ave += matrix_cal[i - 1][j]
                else:
                    tmp_ave += INF
                if ((i - 1) >= 0 and (i - 1) < n and (j + 1) >= 0 and (j + 1) < m):
                    tmp_ave += matrix_cal[i - 1][j + 1]
                else:
                    tmp_ave += INF
                if (i >= 0 and i < n and (j - 1) >= 0 and (j - 1) < m):
                    tmp_ave += matrix_cal[i][j - 1]
                else:
                    tmp_ave += INF
                if (i >= 0 and i < n and (j + 1) >= 0 and (j + 1) < m):
                    tmp_ave += matrix_cal[i][j + 1]
                else:
                    tmp_ave += INF
                if ((i + 1) >= 0 and (i + 1) < n and (j - 1) >= 0 and (j - 1) < m):
                    tmp_ave += matrix_cal[i + 1][j - 1]
                else:
                    tmp_ave += INF
                if ((i + 1) >= 0 and (i + 1) < n and j >= 0 and j < m):
                    tmp_ave += matrix_cal[i + 1][j]
                else:
                    tmp_ave += INF
                if ((i + 1) >= 0 and (i + 1) < n and (j + 1) >= 0 and (j + 1) < m):
                    tmp_ave += matrix_cal[i + 1][j + 1]
                else:
                    tmp_ave += INF
                matrix_cal[i][j] = tmp_ave / 9.0
                if ((matrix1D[i * m + j] != -1) and (matrix_cal[i][j] - matrix[i][j]) > 0.5):
                    matrix1D[i * m + j] = -1
                    cal_num += 1

        if (cal_num == n * m):
            comp_flg = True
            break

    # print("fcnt =", fcnt)
    return matrix_cal
