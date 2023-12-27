import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fusion_dir = 'fusion_result'
if not os.path.exists(fusion_dir):
    os.mkdir(fusion_dir)
    os.mkdir(fusion_dir + '/face')
    os.mkdir(fusion_dir + '/hair')
    os.mkdir(fusion_dir + '/fusion')

for n in range(10):
    # 画像の読み込み
    img = cv2.imread(f"./train_result/images/{n+1}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(f'train_result/masks/{n+1}.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # 画像の切り抜き
    face = cv2.bitwise_and(img, mask)
    plt.axis("off")
    plt.imshow(face)
    plt.savefig(f'{fusion_dir}/face/{n+1}.png')

    hair = cv2.bitwise_or(img, mask)
    # 指定色
    target_color = (255, 255, 255)
    # 変更後の色
    change_color = (0, 0, 0)
    # 画像の縦横
    h, w = hair.shape[:2]
    # 色の変更
    for i in range(h):
        for j in range(w):
            b, g, r = hair[i, j]
            if (b, g, r) == target_color:
                hair[i, j] = change_color
    plt.axis("off")
    plt.imshow(hair)
    plt.savefig(f'{fusion_dir}/hair/{n+1}.png')

for m in range(10):
# 画像の重ね合わせ
    f = cv2.imread(f'{fusion_dir}/face/{m+1}.png')
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    h = cv2.imread(f'{fusion_dir}/hair/{10-m}.png')
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    fusion = cv2.bitwise_or(f, h)
    plt.axis("off")
    plt.imshow(fusion)
    plt.savefig(f'{fusion_dir}/fusion/{m+1}.png')

