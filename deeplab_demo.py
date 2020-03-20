import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import datetime
import time
from keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf

#---------------------------------------------------#
#   windows系统设置程序执行环境目录:
#   等同于linux中的设置环境变量：
#   export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#---------------------------------------------------#
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    # 
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    
    # 模型初始化操作：创建图，加载模型
    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model.
        Args:
        tarball_path: retrained model type of tar.gz  or pb.

        Returns:

        """
        self.graph = tf.Graph()

        graph_def = None
        if ".pb" == os.path.splitext(tarball_path)[1]:
            print("传入pb模型文件…………………………")
            # 传入文件为pb模型文件
            # with open(tarball_path, 'rb') as fhandle:
            #     graph_def = tf.GraphDef.FromString(fhandle.read())
            with tf.gfile.FastGFile(tarball_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        
        # 传入文件为tar压缩文件
        else:
            print("传入tar压缩模型文件…………………………")
            # Extract frozen graph from tar archive.
            tar_file = tarfile.open(tarball_path)
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    print("超找到pb文件………………………………………………")
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.GraphDef.FromString(file_handle.read())
                    break
            # 关闭文件
            tar_file.close()
        print("模型文件读取成功…………………………")
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def run(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: A PIL.Image.object, RGB image resized from original input image.
        seg_map: numpy array object, Segmentation map of `resized_image`.
        """
        width, height = image.size

        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

        seg_map = batch_seg_map[0]
        return resized_image, seg_map

# 创建PASCALVOC分割数据集的类别标签的颜色映射label colormap
def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        创建PASCALVOC分割数据集的类别标签的颜色映射label colormap
    Returns:
        A Colormap for visualizing segmentation results.
        可视化分割结果的颜色映射Colormap
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    # reversed反转0~7迭代器后为：（7，6，5，4，3，2，1，0）
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

    return colormap

# 根据数据集标签的颜色映射label colormap，将数据集标签的颜色映射添加到图片。
def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
        将数据集标签的颜色映射，添加到图片。根据数据集标签的颜色映射 label colormap
    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    # 判断label为2维矩阵
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    # 创建类别标签的颜色映射label colormap
    colormap = create_pascal_label_colormap()
    
    # 正常情况：np.max(label) = len(colormap) - 1
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

# 可视化：1.原图；2.推理出的mask图；3.原图+mask
def vis_segmentation(image, seg_map, output_pathname=None):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))

    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    # seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = seg_map
    plt.imshow(seg_image)
    if output_pathname is not None:
        seg_path_split = os.path.splitext(output_pathname)
        seg_pathname = seg_path_split[0] + "_seg" + seg_path_split[1]
        # print("输出预测结果到指定目录…………………………")
        # print("输出路径名：%s" % out_pathname)
        plt.savefig(seg_pathname)

    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
    if output_pathname is not None:
        all_path_split = os.path.splitext(output_pathname)
        all_pathname = all_path_split[0] + "_all" + all_path_split[1]
        # print("输出预测结果到指定目录…………………………")
        # print("输出路径名：%s" % out_pathname)
        plt.savefig(all_pathname)

    # 获取识别矩阵中物体编号
    # np.unique(seg_map)表示去除重复元素，并获取seg_map矩阵中的非零数元素并进行排序
    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    
    # 绘制右侧类别标签对应的标签图
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    # 在标签图边上添加类别标签
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


#---------------------------------------------------#
#   保存原图和预测结果图片到指定目录
# 
#---------------------------------------------------#
# 可视化：1.原图；2.推理出的mask图；3.原图+mask
def run_save_segmentation(original_img_dir, prediction_img_dir):
    
    total_time = 0
    #---------------------------------------------------#
    #    读取网络上某台服务器jpb图片(https://…………)
    #---------------------------------------------------#
    # try:
    #     f = urllib.request.urlopen(url)
    #     jpeg_str = f.read()
    #     original_im = Image.open(BytesIO(jpeg_str))
    # except IOError:
    #     print('Cannot retrieve image. Please check url: ' + url)
    #     return
    

    #---------------------------------------------------#
    #    读取本地磁盘指定文件夹下的jpg图片(disk file)
    #---------------------------------------------------#
    # 获取文件夹图片数量
    imgs_list = os.listdir(original_img_dir)
    num_imgs = len(imgs_list)
    print("Images num:"+str(num_imgs))
    start_time=datetime.datetime.now()

    print("STARTING ...")

    # 读取mg_path目录下所有图像
    for filename in imgs_list:
        print('running deeplab on image %s...' % filename)
        #预测输出保存为 .png格式
        seg_prename = filename[0:-4] + "_predict" + ".png"   
        original_prename = filename[0:-4] + "_original" + ".png"   
        # 模型输入图像文件路径名
        file_path = original_img_dir + "/" + filename
        # 模型预测之后，原图像保存路径名
        seg_save_path = prediction_img_dir + '/' + seg_prename
        original_save_path = prediction_img_dir + '/' + original_prename
        
  
        #---------------------------------------------------#
        #   加载图片 -- 实现方式一
        #   采用keras.preprocessing.image 方法
        # 根据图像路径URL载入图像数据,并转化为numpy格式
        # 将图片转换为uint8，shape为[1, None, None, 3]格式；
        #---------------------------------------------------#
        # img = load_img(file_path)
        # original_img_np = img_to_array(img)
        # original_img_expanded = np.expand_dims(original_img_np, axis=0).astype(np.uint8)


        #---------------------------------------------------#
        #   加载图片 -- 实现方式二
        #   采用PIL方法
        #---------------------------------------------------#
        original_img = Image.open(file_path)
        # img_np_expanded = np.expand_dims(img_np, axis=0).astype(np.uint8)  
      

        #---------------------------------------------------#
        #   运行模型 -- 实现方式一
        #   模型图输入为INPUT_TENSOR_NAME
        #   输出为OUTPUT_TENSOR_NAME
        #---------------------------------------------------#
        time1 = time.time()
        # 模型推理
        resized_im, seg_map = MODEL.run(original_img)

        # seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_image = Image.fromarray(seg_map.astype('uint8')).convert('RGB')        
        # 保存结果
        seg_image.save(seg_save_path)
        resized_im.save(original_save_path)

        time2 = time.time()
        total_time += float(time2-time1)

    end_time=datetime.datetime.now()

    print("START TIME :"+str(start_time))
    print("END TIME :"+str(end_time))
    print("THE TOTAL TIME COST IS:"+str(total_time))
    print("THE average TIME COST IS:"+str(float(total_time)/float(num_imgs)))
    

# PASCAL VOC 2012数据集
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


# 自定义数据集
# LABEL_NAMES = np.asarray([
#     'background', 'pothole', 'car', 'midlane', 'rightlane', 'dashedline'
#     ])


# 生成[21, 1]矩阵，值为0~20
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
# 生成[21, 1, 3]
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


#---------------------------------------------------#
#   加载指定预训练模型:
#   1.创建存放模型的临时目录
#   2.下载模型文件
#   3.加载模型
#---------------------------------------------------#
# 选择所使用的模型
MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
# @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 
# 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    'xception_mydata':
        'deeplab_model.tar.gz',
}


#---------------------------------------------------#
#   加载远程模型文件--tar.gz格式:
#---------------------------------------------------#
# _TARBALL_NAME = 'deeplab_model.tar.gz'
# # 创建临时目录
# model_dir = tempfile.mkdtemp()
# tf.gfile.MakeDirs(model_dir)
# download_path = os.path.join(model_dir, _TARBALL_NAME)
# print('downloading model, this might take a while...')
# urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
#                 download_path)
# print('download completed! loading DeepLab model...')


#---------------------------------------------------#
#   加载本地模型文件--tar.gz格式:
#---------------------------------------------------#
# model_dir = 'E:/PycharmProject/models_zoo/deeplab/PASCAL_VOC_2012/'
# model_dir = 'E:/PycharmProject/DeepLab_model/datasets/RoadScene/exp/train_on_train_set/export/'
# download_path = os.path.join(model_dir, _MODEL_URLS[MODEL_NAME])


#---------------------------------------------------#
#   加载本地模型文件--pb格式:
#---------------------------------------------------#
# download_path = "E:/PycharmProject/DeepLab_model/datasets/RoadScene/exp/train_on_train_set/export/deeplab_model.tar.gz"
# download_path = 'E:/PycharmProject/models_zoo/deeplab/PASCAL_VOC_2012/deeplabv3_pascal_trainval_2018_01_04/deeplabv3_pascal_trainval/frozen_inference_graph.pb'
# download_path = "E:/PycharmProject/DeepLab_model/deeplabv3_cityscapes_train/frozen_inference_graph.pb"
download_path = "E:/PycharmProject/DeepLab_model/models/research/deeplab/datasets/pascal_voc_seg/exp/train_on_trainval_set/export/frozen_inference_graph.pb"

# 根据路径加载模型
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')



#---------------------------------------------------#
#   windows系统设置程序执行环境目录:
#   等同于linux中的设置环境变量：
#   export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#---------------------------------------------------#
SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ''  #@param {type:"string"}

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
            'deeplab/g3doc/img/%s.jpg?raw=true')


# 进行推理，并展示结果
def run_visualization(input_pathname, output_pathname = None):
    """Inferences DeepLab model and visualizes result."""
    #---------------------------------------------------#
    #    读取网络上某台服务器jpb图片(https://…………)
    #---------------------------------------------------#
    # try:
    #     f = urllib.request.urlopen(url)
    #     jpeg_str = f.read()
    #     original_im = Image.open(BytesIO(jpeg_str))
    # except IOError:
    #     print('Cannot retrieve image. Please check url: ' + url)
    #     return
    

    #---------------------------------------------------#
    #    读取本地磁盘上jpg图片(disk file)
    #---------------------------------------------------#
    original_im = Image.open(input_pathname)
    print('running deeplab on image %s...' % input_pathname)
    

    # 模型推理
    resized_im, seg_map = MODEL.run(original_im)
    
    # 可视化显示
    vis_segmentation(resized_im, seg_map, output_pathname=output_pathname)



if __name__ == '__main__':
    # image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
    # run_visualization(image_url)


#---------------------------------------------------#
#    对指定文件夹内图片进行分割，并保存结果到指定文件夹 
#---------------------------------------------------#
    # images_dir = "E:/PycharmProject/DeepLab_model/datasets/RoadScene/predict_test/"
    # output_images_dir = "E:/PycharmProject/DeepLab_model/datasets/RoadScene/output_predict_test/"
    # run_save_segmentation(original_img_dir=images_dir, prediction_img_dir=output_images_dir)


#---------------------------------------------------#
#    基于deeplab源码实现，指定文件夹下图像分割，及mask可视化 
#---------------------------------------------------#
    # images_dir = 'E:/PycharmProject/datasets/images/'
    images_dir = "E:/PycharmProject/DeepLab_model/datasets/RoadScene/predict_test/"
    output_images_dir = "E:/PycharmProject/DeepLab_model/datasets/RoadScene/output_predict_test/"
    images = sorted(os.listdir(images_dir))
    i = 1
    for imgfile in images:
        print("Image：" + str(i))
        run_visualization(input_pathname = os.path.join(images_dir, imgfile),
         output_pathname = os.path.join(output_images_dir, imgfile))
        i += 1
    print('Done.')


#---------------------------------------------------#
#    基于opencv 实现指定mask文件可视化
#---------------------------------------------------#
    # imgfile = 'image.jpg'
    # pngfile = 'mask.png'

    # img = cv2.imread(imgfile, 1)
    # mask = cv2.imread(pngfile, 0)

    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

    # img = img[:, :, ::-1]
    # img[..., 2] = np.where(mask == 1, 255, img[..., 2])

    # plt.imshow(img)
    # plt.show()
