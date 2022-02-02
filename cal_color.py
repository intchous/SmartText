from sklearn.cluster import KMeans
import numpy as np
import time
import cv2


def cal_domcolor(img_np, k):
    st = time.time()
    img_km = img_np.reshape((img_np.shape[0] * img_np.shape[1], img_np.shape[2]))
    estimator = KMeans(n_clusters=k, max_iter=300, n_init=2)
    estimator.fit(img_km)
    centroids = estimator.cluster_centers_
    centroids = sorted(centroids, key=lambda x: (x[0], x[1], x[2]))
    ed = time.time()
    # print("KMeans.time = " + ed - st)
    return centroids


def draw_domcolor(centroids, n_channels, sv_fp):
    result = []
    res_width = 200
    res_height_per = 80
    k = len(centroids)
    for center_index in range(k):
        result.append(np.full((res_width * res_height_per, n_channels), centroids[center_index], dtype=int))
    result = np.array(result, dtype=np.uint8)
    result = result.reshape((res_height_per * k, res_width, n_channels))

    result_bgr = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite(sv_fp, result_bgr)


def rgb_distance(rgb):
    return abs(rgb[0] - rgb[1]) + abs(rgb[0] - rgb[2]) + abs(rgb[2] - rgb[1])


def RGB_to_Hex(rgb):
    color_str = '#'
    for i in rgb:
        num = int(i)
        color_str += str(hex(num))[-2:].replace('x', '0').upper()
    return color_str


def cal_luminance(rgb):
    for i in range(0, len(rgb)):
        if rgb[i] <= 0.03928:
            rgb[i] = rgb[i] / 12.92
        else:
            rgb[i] = pow(((rgb[i] + 0.055) / 1.055), 2.4)
    l = (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2])
    return l


def cal_contrast_rate(rgbA, rgbB):
    ratio = 1
    l1 = cal_luminance([rgbA[0] / 255, rgbA[1] / 255, rgbA[2] / 255])
    l2 = cal_luminance([rgbB[0] / 255, rgbB[1] / 255, rgbB[2] / 255])
    if l1 >= l2:
        ratio = (l1 + .05) / (l2 + .05)
    else:
        ratio = (l2 + .05) / (l1 + .05)
    ratio = round(ratio * 100) / 100
    return ratio


def cal_best_color(img, img_crop, contrast_threshold=5.5):
    color_candidates = cal_domcolor(img, 6)
    crop_color = cal_domcolor(img_crop, 1)[0]
    color_choose = []
    grey_flag = False
    for color in color_candidates:
        tmp_cr = cal_contrast_rate(color, crop_color)
        if tmp_cr > contrast_threshold:
            color_choose.append({"color": color, "contrast_rate": tmp_cr})

    if len(color_choose) == 0:
        grey_flag = True
        grey_candidates = []
        for i in range(0, 256, 50):
            grey_candidates.append([i, i, i])

        for grey_color in grey_candidates:
            tmp_cr = cal_contrast_rate(grey_color, crop_color)
            if tmp_cr > contrast_threshold:
                color_choose.append({"color": grey_color, "contrast_rate": tmp_cr})

    if len(color_choose) == 0:
        black_cr = cal_contrast_rate([0, 0, 0], crop_color)
        white_cr = cal_contrast_rate([255, 255, 255], crop_color)
        if (black_cr > white_cr):
            color_choose.append({"color": [0, 0, 0], "contrast_rate": black_cr})
        else:
            color_choose.append({"color": [255, 255, 255], "contrast_rate": white_cr})

    if grey_flag:
        color_choose_sorted = sorted(color_choose, key=lambda x: x["contrast_rate"], reverse=True)

    else:
        color_choose_sorted = sorted(color_choose, key=lambda x: rgb_distance(x["color"]), reverse=True)
    return color_choose_sorted
