import tensorflow as tf
import numpy as np
# import cv2 as cv
from PIL import Image 
import os
import time
import datetime
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt



def predict_images(original_img_dir, model_path, prediction_img_dir):
    
    # 判断输入文件夹是否存在
    print("判断输入是单张图片，还是图片集文件夹")


    # 判断输出文件夹是否存在
    # prediction_result_dir="./prediction_dir_name"
    if not os.path.exists(prediction_img_dir):
        os.mkdir(prediction_img_dir)
    
    
    # 模型输入图像名称
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    # 模型输出结果名称
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    graph_def = None
    total_time = 0

    # 获取文件夹图片数量
    imgs_list = os.listdir(original_img_dir)
    num_imgs = len(imgs_list)
    print("Images num:"+str(num_imgs))

    #---------------------------------------------------#
    #   加载模型--实现方式一：
    #   读取模型图结构为graph_def,并设为当前会话默认图
    #---------------------------------------------------#
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive.')
    # 构建图实例
    detection_graph = tf.Graph()
    # 将读取的模型graph_def图设为session会话图graph
    with detection_graph.as_default():
        tf.import_graph_def(graph_def, name='')
    
    #---------------------------------------------------#
    #   加载模型--实现方式二：
    #   读取模型图结构为graph_def,并设为当前会话默认图
    #---------------------------------------------------#
    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(model_path, 'rb') as fid:      #加载模型
    #         serialized_graph = fid.read()
    # graph_def.ParseFromString(serialized_graph)
    # if graph_def is None:
    #     raise RuntimeError('Cannot find inference graph in tar archive.')
    # tf.import_graph_def(graph_def, name='')


    # 依据模型图结构创建会话
    sess = tf.Session(graph=detection_graph)
    start_time=datetime.datetime.now()
    print("STARTING ...")

    # 读取mg_path目录下所有图像
    for filename in imgs_list:
        #预测输出保存为 .png格式
        prename = filename[0:-4] + ".png"   
        # 模型输入图像文件路径名
        file_path = original_img_dir + "/" + filename
        # 模型预测之后，原图像保存路径名
        save_path = prediction_img_dir + '/' + prename
  
        #---------------------------------------------------#
        #   加载图片 -- 实现方式一
        #   采用keras.preprocessing.image 方法
        #---------------------------------------------------#
        # 根据图像路径URL载入图像数据,并转化为numpy格式
        img = load_img(file_path)
        img_np = img_to_array(img)
        # 将图片转换为uint8，shape为[1, None, None, 3]格式；
        img_np_expanded = np.expand_dims(img_np, axis=0).astype(np.uint8)
        
        #---------------------------------------------------#
        #   加载图片 -- 实现方式二
        #   采用PIL方法
        #---------------------------------------------------#
        # img_np = Image.open(image_path)
        # img_np_expanded = np.expand_dims(img_np, axis=0).astype(np.uint8)  
      
        #---------------------------------------------------#
        #   运行模型 -- 实现方式一
        #   模型图输入为INPUT_TENSOR_NAME
        #   输出为OUTPUT_TENSOR_NAME
        #---------------------------------------------------#
        time1 = time.time()
        result = sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: img_np_expanded})
        
        # result = sess.run(output)
        # print(type(result))
        #result.save('aaa.png')
        #cv.imshow(result[0])
        # print(result[0].shape)  # (1, height, width)
        
        # print(result)
        # print(result[0])
        #result[0].save('aaa.png')
        # img_ = img[:,:,::-1].transpose((2,0,1))
        
        # cv.imwrite('aaa.jpg',result.transpose((1,2,0)))
        # plt.imshow(result[0])
        # plt.show()

        # cv.imwrite(save_path, result.transpose((1, 2, 0)))
        result.transpose((1, 2, 0)).save(save_path)
        time2 = time.time()
        total_time += float(time2-time1)

        #---------------------------------------------------#
        #   运行模型 -- 实现方式二
        #   input_map：指出输入是什么
        #   return_elements:指明输出的是什么
        #---------------------------------------------------#
        # time1 = time.time()
        # output = tf.import_graph_def(graph_def,
        #                              input_map={"ImageTensor:0": img_expanded}, 
        #                              return_elements=["SemanticPredictions:0"])
        # result = sess.run(output)
        # cv.imwrite(save_path, result.transpose((1, 2, 0)))
        # time2 = time.time()
        # total_time += float(time2-time1)

    end_time=datetime.datetime.now()

    print("START TIME :"+str(start_time))
    print("END TIME :"+str(end_time))
    print("THE TOTAL TIME COST IS:"+str(total_time))
    print("THE average TIME COST IS:"+str(float(total_time)/float(num_imgs)))



if __name__ == '__main__':
  
  model_path =  "datasets/RoadScene/exp/train_on_train_set/export/frozen_inference_graph.pb"
  pre_img_dir = "E:\PycharmProject\DeepLab_model\datasets\RoadScene\predict_test"
  ori_img_dir = "E:\PycharmProject\DeepLab_model\datasets\RoadScene\predict_test"
  predict_images(model_path=model_path, original_img_dir=ori_img_dir, prediction_img_dir=pre_img_dir)
