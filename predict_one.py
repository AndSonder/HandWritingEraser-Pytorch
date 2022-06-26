import torchvision.transforms.functional
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import argparse
from torchvision import transforms as T
import network
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from tqdm import tqdm

"""裁剪img"""
CROP_SIZE = 1024


def img_crop(img):
    '''
    input: ndarray, [C, H, W]
    output: 2048x2048, img_list(np.array)
    '''
    h, w = img.shape[1:]
    x_number = w // CROP_SIZE + 1
    y_number = h // CROP_SIZE + 1

    if (x_number, y_number) == (1, 1):
        return [[img]]

    else:
        img_list = []
        for i in range(y_number):
            img_list.append([])

        for i in range(1, y_number + 1):
            for j in range(1, x_number + 1):
                if i == 1:
                    center_y = int(CROP_SIZE / 2)
                elif i == y_number:
                    center_y = h - 1024
                else:
                    center_y = (h - CROP_SIZE) // (y_number - 1) * (i - 1) + int(CROP_SIZE / 2)

                if j == 1:
                    center_x = int(CROP_SIZE / 2)
                elif j == x_number:
                    center_x = w - int(CROP_SIZE / 2)
                else:
                    center_x = (w - CROP_SIZE) // (x_number - 1) * (j - 1) + int(CROP_SIZE / 2)

                img1 = img[:, center_y - int(CROP_SIZE / 2):center_y + int(CROP_SIZE / 2),
                       center_x - int(CROP_SIZE / 2):center_x + int(CROP_SIZE / 2)]

                img_list[i - 1].append(img1)  # [C, H, W]

        return img_list


def predict(img_path, model, device, transform):
    image_name = img_path
    ori_img = Image.open(image_name)  # [H, W, C]
    ori_img = np.array(ori_img).transpose(2, 0, 1)  # [C, H, W]
    ori_img0, ori_img1, ori_img2 = ori_img[0], ori_img[1], ori_img[2]  # R, G, B
    ori_img00, ori_img11, ori_img22 = ori_img0.copy(), ori_img1.copy(), ori_img2.copy()

    img = Image.open(img_path).convert('RGB')
    img = torchvision.transforms.functional.pil_to_tensor(img).unsqueeze(0) / 255.0
    img = img.to(device)
    img = model(img).detach().max(dim=1)[1].cpu().numpy()[0]

    # 去除手写
    tmp_img = img.copy().astype(np.uint8)  # 去除手写
    tmp_img2 = img.copy().astype(np.uint8)  # 保留印刷
    tmp_img[tmp_img == 0] = 0
    tmp_img[tmp_img == 1] = 255  # 手写
    tmp_img[tmp_img == 2] = 0
    kernel = np.ones((2, 2), np.uint8)
    tmp_img = cv2.dilate(tmp_img, kernel, iterations=1).astype(np.uint8)
    tmp_img[tmp_img >= 150] = 255

    # 保留印刷
    img[img == 0] = 0
    img[img == 2] = 255  # 印刷
    img[img == 1] = 0
    img = img.astype(np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    kernel2 = np.ones((2, 4), np.uint8)
    img2 = cv2.dilate(img, kernel2, iterations=1).astype(np.uint8)
    img = cv2.dilate(img, kernel, iterations=4).astype(np.uint8)

    img[img >= 150] = 255
    img2[img2 >= 150] = 255

    retain_area = img - tmp_img
    retain_area[retain_area >= 255] = 255
    retain_area[img2 == 255] = 255

    ori_img0[retain_area == 255] = ori_img00[retain_area == 255]
    ori_img1[retain_area == 255] = ori_img11[retain_area == 255]
    ori_img2[retain_area == 255] = ori_img22[retain_area == 255]
    ori_img0[retain_area != 255] = 255
    ori_img1[retain_area != 255] = 255
    ori_img2[retain_area != 255] = 255
    ori_img = np.array([ori_img0, ori_img1, ori_img2]).transpose(1, 2, 0).astype(np.uint8)  # [H, W, C]

    # 去除红色
    ori_img = ori_img.transpose(2, 0, 1)
    ori_img0, ori_img1, ori_img2 = ori_img[0], ori_img[1], ori_img[2]  # R, G, B
    tmp_img = ori_img.copy().astype(np.float32)
    tmp_img1 = tmp_img[0] - tmp_img[1] + tmp_img[0] - tmp_img[2]

    ori_img0[tmp_img1 >= 100] = 255
    ori_img1[tmp_img1 >= 100] = 255
    ori_img2[tmp_img1 >= 100] = 255
    ori_img = np.array([ori_img0, ori_img1, ori_img2]).transpose(1, 2, 0).astype(np.uint8)  # [H, W, C]

    return ori_img


def erase_hand_write(img_path, model1, opt):
    # 读取图像，并切割
    img = Image.open(img_path)  # [H, W, C]
    img = np.array(img)
    img = img.transpose(2, 0, 1)  # [C, H, W]

    width, height = img.shape[2], img.shape[1]

    crop_img_list = img_crop(img)  # [C, H, W]
    process_img_list = []

    # 使用模型输出每个切割图像的预测
    for i in range(len(crop_img_list)):
        process_img_list.append([])
        for j in range(len(crop_img_list[i])):
            img_path = Image.fromarray(crop_img_list[i][j].transpose(1, 2, 0).astype(np.uint8))
            img_path.save('tmp_img.png')
            predict_image = predict('tmp_img.png', model1, opt.device, opt.transform)  # [H, W, C]
            process_img_list[i].append(predict_image)

    shape_y = len(process_img_list)
    shape_x = len(process_img_list[0])

    # 拼接所有图像
    if shape_x == 1 and shape_y == 1:
        output_img = process_img_list[0][0]

    elif shape_x == 2 and shape_y == 1:
        output_img = np.zeros_like(img)  # [C, H, W]
        output_img = output_img.transpose([0, 2, 1])  # [C, W, H]

        output_img[:, 0:CROP_SIZE, :] = process_img_list[0][0].transpose(2, 1, 0)  # [C, W, H]
        output_img[:, width - CROP_SIZE:width, :] = process_img_list[0][1].transpose(2, 1, 0)  # [C, W, H]
        output_img = output_img.transpose(2, 1, 0).astype(np.uint8)

    elif shape_x == 1 and shape_y == 2:
        output_img = np.zeros_like(img)
        output_img = output_img.transpose([0, 2, 1])  # [C, W, H]
        output_img[:, :, :CROP_SIZE] = process_img_list[0][0].transpose(2, 1, 0)  # [C, W, H]
        output_img[:, :, height - CROP_SIZE:height] = process_img_list[1][0].transpose(2, 1, 0)  # [C, W, H]
        output_img = output_img.transpose(2, 1, 0).astype(np.uint8)

    else:
        output_img = np.zeros_like(img)
        output_img = output_img.transpose([0, 2, 1])  # [C, W, H]

        for i in range(shape_y):  # (0, 1, 2)
            for j in range(shape_x):  # (0, 1)
                if i == 0:
                    if j == 0:
                        center_x, center_y = int(CROP_SIZE / 2), int(CROP_SIZE / 2)
                    elif j == shape_x - 1:
                        center_x, center_y = width - int(CROP_SIZE / 2), int(CROP_SIZE / 2)
                    else:
                        center_x, center_y = width // 2, int(CROP_SIZE / 2)

                elif i == shape_y - 1:
                    if j == 0:
                        center_x, center_y = int(CROP_SIZE / 2), height - int(CROP_SIZE / 2)
                    elif j == shape_x - 1:
                        center_x, center_y = width - int(CROP_SIZE / 2), height - int(CROP_SIZE / 2)
                    else:
                        center_x, center_y = width // 2, height - int(CROP_SIZE / 2)
                else:
                    if j == 0:
                        center_x, center_y = int(CROP_SIZE / 2), height // 2
                    elif j == shape_x - 1:
                        center_x, center_y = width - int(CROP_SIZE / 2), height // 2
                    else:
                        center_x, center_y = width // 2, height // 2

                output_img[:, center_x - int(CROP_SIZE / 2):center_x + int(CROP_SIZE / 2),
                center_y - int(CROP_SIZE / 2):center_y + int(CROP_SIZE / 2)] = process_img_list[i][
                    j].transpose(2, 1, 0)

        output_img = output_img.transpose(2, 1, 0).astype(np.uint8)
    return output_img


def main():
    opts = argparse.ArgumentParser()
    opts.add_argument("--mode_path", type=str, default="checkpoints/best_deeplabv3plus_resnet50_os16.pth")
    opts.add_argument("--device", type=str, default='1')
    opts.add_argument("--data_path", type=str, default='/home/disk2/ray/datasets/HandWriting/dehw_testA_dataset/images')
    opts.add_argument('--test_one', type=str,
                      default='/home/disk2/ray/datasets/HandWriting/dehw_testA_dataset/images/dehw_testA_00015.jpg')
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    opts = opts.parse_args()
    opts.transform = transform

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.device
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = network.modeling.deeplabv3plus_resnet101(num_classes=3, output_stride=16)
    # network.convert_to_separable_conv(model.classifier)
    checkpoint = torch.load(opts.mode_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(opts.device)
    model.eval()

    img_list = os.listdir(opts.data_path)
    for i, item in tqdm(enumerate(img_list)):
        path = os.path.join(opts.data_path, item)
        res = erase_hand_write(path, model, opts)
        save_path = 'results/normal_result/' + str(i) + '.png'
        cv2.imwrite(save_path, res)


if __name__ == '__main__':
    main()
