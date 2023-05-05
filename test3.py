import copy
import os
import pickle
from typing import Tuple

import cv2  # 4.5.4
import numpy as np  # 1.24.2
import pandas as pd

# 导入preprocess文件夹下的preprocess.py文件中的ChipRecognition类
from preprocess.preprocess import ChipRecognition


# 图像的空洞阴影面积检测
class Shadow():
    def __init__(self):
        # self.thres_dict = {
        #     'IGBT': (80, 79),
        #     'FRD': (80, 79),
        #     'SIC': (80, 79),
        # }
        self.thres_dict = {
            'IGBT': (100, 150),
            'FRD': (100, 150),
            'SIC': (100, 150),
        }
        self.shadow_chip_size_dict = {
            'IGBT': (92, 92),  # 8464
            'FRD': (51, 71),  # 3621
            'SIC': (34, 34),  # 1156
        }

        # 存储芯片空洞信息的字典
        self.shadow_dict = {}

        # 存储可视化参数
        # 没问题芯片为绿色边框
        # 有问题芯片为红色边框
        self.good_mask = (0, 255, 0)
        self.bad_mask = (0, 0, 255)

    def cut_standard_chip(self, img: np.ndarray, chip_bbox: np.ndarray, centroid: Tuple[int, int], img_label: str) -> \
    Tuple[
        np.ndarray, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        '''
        从输入的原始图像中截取标准的芯片图像，根据不同的芯片类型设置不同的芯片大小。

        :param img: 输入的原始图像
        :type img: numpy.ndarray
        :param centroid: 输入的芯片中心坐标  eg: (x, y)
        :type centroid: tuple[int, int]
        :param img_label: 输入的芯片类型   eg: 'IGBT'
        :type img_label: str
        :return: 返回的标准芯片图像和芯片的四个顶点坐标
        :rtype: Tuple[numpy.ndarray, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]
        '''
        # 根据不同的芯片类型，设置不同的芯片大小
        # 并且要根据传入的芯片bbox
        # 判断芯片是横着还是竖着的
        if chip_bbox[2] - chip_bbox[0] > chip_bbox[3] - chip_bbox[1]:
            height, width = self.shadow_chip_size_dict[img_label]
        else:
            width, height = self.shadow_chip_size_dict[img_label]

        # 将 centroid 坐标转换为整数类型，以便使用它作为切片索引
        centroid = (int(centroid[0]), int(centroid[1]))

        # 计算芯片的4个顶点坐标
        # 左上角坐标
        left_top = (centroid[0] - width // 2, centroid[1] - height // 2)
        # 右上角坐标
        right_top = (centroid[0] + width // 2, centroid[1] - height // 2)
        # 左下角坐标
        left_bottom = (centroid[0] - width // 2, centroid[1] + height // 2)
        # 右下角坐标
        right_bottom = (centroid[0] + width // 2, centroid[1] + height // 2)

        # 以 centroid 坐标为中心，以 img_label 为芯片类型确定芯片大小，从原始图像中截取芯片
        img = img[left_top[1]: right_bottom[1], left_top[0]: right_bottom[0]]

        # 返回截取的芯片图像和芯片的四个顶点坐标
        return img, (left_top, right_top, left_bottom, right_bottom)

    # 固定阈值的双阈值二值化处理

    def double_threshold(self, img: np.ndarray, img_label: str) -> np.ndarray:
        '''
        :param img: 要进行二值化的图像
        :param low: 低阈值
        :param high: 高阈值
        :return img: 返回的二值化图像
        '''
        low_threshold, high_threshold = self.thres_dict[img_label]

        # Apply the high and low threshold filters
        _, bright_mask = cv2.threshold(img, high_threshold, 255, cv2.THRESH_BINARY)
        _, dark_mask = cv2.threshold(img, low_threshold, 0, cv2.THRESH_BINARY_INV)

        # Combine the bright and dark masks
        thresholded_image = cv2.bitwise_or(bright_mask, dark_mask)

        # Return the binary mask
        return thresholded_image

    # 对于一些细小的像素，效果不如直接计算黑色像素点的个数
    def shadow_area(self, img: np.ndarray) -> int:
        '''
        :param img: 输入的二值化图像
        :return: img_shadow_area: 返回的空洞阴影面积
        '''
        # 计算轮廓
        contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 计算轮廓面积和轮廓边界面积
        shadow_area = 0
        edge_area = 0
        for i in range(len(contours)):
            # 如果轮廓没有父轮廓，则说明是最外层轮廓
            if hierarchy[0][i][3] == -1:
                # 计算轮廓面积并加入空洞面积
                shadow_area += cv2.contourArea(contours[i])
                # 计算轮廓边界面积并加入边界面积
                perimeter = cv2.arcLength(contours[i], True)
                edge_area += cv2.contourArea(cv2.approxPolyDP(contours[i], 0.02 * perimeter, True))
        # 返回空洞阴影面积和轮廓边界面积之和
        return shadow_area + edge_area

    def black_pixel_sum(self, img: np.ndarray) -> int:
        '''
        :param img: 输入的二值化图像
        :return: 返回的黑色像素总数
        '''
        # 反转图像，将黑色区域变为白色
        # img = cv2.bitwise_not(img)
        # 计算黑色像素总数
        black_pixel_sum = np.count_nonzero(img, axis=None)
        # 返回黑色像素总数
        return black_pixel_sum

    # 根据输入的roi图像，可视化空洞阴影，在原图上绘点
    def shadow_roi_visualization(self, ori_img, roi_img, centroid, chip_info=None):
        '''
        :param ori_img: 输入的原始图像
        :param roi_img: 输入的二值化图像
        :param centroid: roi区域的中心点坐标，(x,y)
        :return: 返回在原始图像中绘制roi区域及其顶点坐标的图像
        '''

        roi_img_copy = copy.deepcopy(roi_img)

        # 判断ori_img的通道数
        # 如果为三通道，则不需要进行转换   eg: len(ori_img.shape) == 3
        # 如果为单通道，则需要转换为三通道  eg: len(ori_img.shape) == 2
        if len(ori_img.shape) == 2:
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)

        # 导入芯片坐标信息
        if chip_info is not None:
            (left_top, right_top, left_bottom, right_bottom) = chip_info['point_position']


        # 将 centroid 坐标转换为整数类型，以便使用它作为切片索引
        centroid = (int(centroid[0]), int(centroid[1]))
        #
        # ########################    边框绘制    ########################
        # # 计算芯片的2个顶点坐标
        # # 左上角坐标
        # left_top = (centroid[0] - chip_size[0] // 2, centroid[1] - chip_size[1] // 2)
        # # 右下角坐标
        # right_bottom = (centroid[0] + chip_size[0] // 2, centroid[1] + chip_size[1] // 2)

        # 根据2个顶点坐标，在ori_img中画出红色边框
        cv2.rectangle(ori_img, left_top, right_bottom, (0, 0, 255), 2)

        ########################    空洞阴影绘制    ########################
        # centroid是roi_img在ori_img中的中心点坐标
        # 将roi_img中的白色区域坐标转换为ori_img中的坐标
        # 并在ori_img中用红色绘制出来
        for i in range(roi_img.shape[0]):
            for j in range(roi_img.shape[1]):
                if roi_img[i][j] == 255:
                    ori_img[centroid[1] - roi_img.shape[0] // 2 + i][centroid[0] - roi_img.shape[1] // 2 + j] = (
                    0, 0, 255)

        ########################    绘制文字信息    ########################

        return ori_img

    # 推理函数
    # def inference(self, ):

    @classmethod
    def filter_detection_res(cls, img_info, keywords=['IGBT', 'FRD', 'SiC']):
        """
        该方法过滤输入的 `img_info` 列表中每个元素的 'detection_res' 部分，将其中包含指定关键词的部分保留下来。
        如果没有指定关键词，则默认使用 'IGBT'，'FRD' 和 'SiC'。

        :param img_info: 包含检测结果的列表，其中每个元素应该是一个字典，包含 'detection_res' 键。
        :type img_info: list[dict[str, Any]]
        :param keywords: 要用于过滤检测结果的关键词列表，默认为 ['IGBT', 'FRD', 'SiC']
        :type keywords: list[str]
        :return: 返回一个新的列表，其中每个元素包含过滤后的 'detection_res' 部分。
        :rtype: list[dict[str, Any]]
        """
        filtered_img_info = []

        for item in img_info:
            filtered_detections = [detection for detection in item['detection_res'] if
                                   any(keyword in detection['id'] for keyword in keywords)]
            item['detection_res'] = filtered_detections
            filtered_img_info.append(item)

        return filtered_img_info

    @classmethod
    def get_unique_file_path(cls, output_dir: str, file_name: str) -> str:
        """
        获取唯一的文件路径和名称
        """
        # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 组合文件路径和名称
        file_path = os.path.join(output_dir, file_name)

        # 判断文件是否存在，存在则添加后缀，直到文件名唯一
        if os.path.isfile(file_path):
            file_dir, file_name = os.path.split(file_path)
            file_name, ext = os.path.splitext(file_name)
            suffix = 1
            while True:
                new_file_name = f"{file_name}_{suffix}{ext}"
                new_file_path = os.path.join(file_dir, new_file_name)
                if os.path.isfile(new_file_path):
                    suffix += 1
                else:
                    file_path = new_file_path
                    break

        return file_path


if __name__ == '__main__':
    ChipRecognition = ChipRecognition()
    shadow = Shadow()

    ## 读取CSV文件，假设文件中有tiff_path列
    positive_df = pd.read_csv('./test_data/csv/void_test_positive.csv')
    negative_df = pd.read_csv('./test_data/csv/void_test_negative.csv')

    # 合并两个dataframe
    test_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # 读取tiff_path列
    tiff_paths = test_df['tiff_path'].tolist()

    # 取前3个tiff_path用于测试
    tiff_paths = tiff_paths[:3]

    # ChipRecognition.inference(tiff_paths)
    # img_info = ChipRecognition.batch_module_info
    #
    #
    # filtered_detection_res = Shadow.filter_detection_res(img_info)
    #
    # # 持久化test变量，用于后续分析
    # with open('./test_data/img_info.pkl', 'wb') as f:
    #     pickle.dump(filtered_detection_res, f)

    # unpickle
    with open('./test_data/img_info.pkl', 'rb') as f:
        img_info = pickle.load(f)

    for i in range(len(img_info)):
        # 读取tiff图像
        _, img_tiff = cv2.imreadmulti(tiff_paths[i], flags=cv2.IMREAD_ANYDEPTH)

        # 读取第二层tiff图
        img = img_tiff[1]

        # 深拷贝一份原始图像，用于可视化
        visual_img = copy.deepcopy(img)

        # 检测"detection_res"
        # for j in range(3):
        for j in range(len(img_info[i]['detection_res'])):
            # 获取roi的信息
            id = img_info[i]['detection_res'][j]['id']
            img_label = id.split('_')[0]
            roi_centroid = img_info[i]['detection_res'][j]['centroid']
            chip_bbox = img_info[i]['detection_res'][j]['bbox']

            # 二值化roi图像
            test_ori, test_ori_point_position = shadow.cut_standard_chip(img=img, chip_bbox=chip_bbox, centroid=roi_centroid,
                                                                         img_label=img_label)
            test_ori = shadow.double_threshold(img=test_ori, img_label=img_label)

            ################    对roi图像进行空洞检测    ################
            # 轮廓法
            test_area = shadow.shadow_area(img=test_ori)
            # 黑色像素法
            test_black_area = shadow.black_pixel_sum(img=test_ori)
            # 打印对比
            print(f'id: {id}, shadow_area: {test_area},  black_area: {test_black_area}')

            output_dir = "./test_data/server_output/"
            file_name = f"{id}.png"
            file_path = Shadow.get_unique_file_path(output_dir, file_name)
            # # 保存图像
            cv2.imwrite(file_path, test_ori)

            ############################## 可视化图像  ##############################
            # 打包可视化中的文字信息
            chip_info = {
                'id': id,
                'shadow_area': test_black_area,
                'point_position': test_ori_point_position,
            }
            # 可视化
            visual_img = shadow.shadow_roi_visualization(ori_img=visual_img, roi_img=test_ori, centroid=roi_centroid,
                                                         chip_info=chip_info)

        output_dir = "./test_data/server_output/visualization"
        file_name = f"{os.path.basename(tiff_paths[i])}.png"
        file_path = Shadow.get_unique_file_path(output_dir, file_name)
        # 将可视化图像保存到指定路径
        cv2.imwrite(file_path, visual_img)
