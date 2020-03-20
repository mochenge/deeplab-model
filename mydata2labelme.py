
# -*- coding:utf-8 -*-
import argparse
import json
import os
import os.path as osp
import warnings
import copy
 
import numpy as np
import PIL.Image
from skimage import io
import yaml
import cv2
import copy
 
def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
#https://blog.csdn.net/u010103202/article/details/81635436
 
 
#labelme标注的数据分析
#https://blog.csdn.net/wc781708249/article/details/79595174
 
#数据先转换成lableme，然后调用labelme的函数转换为VOC
 
#https://github.com/wkentaro/labelme/tree/master/examples/semantic_segmentation
 
ori_path = "./luosi_phone_seg_20191008"
save_path = "./seg_groundtruth"
json_path = "./labelme_json"
gray_path = "./JPEGImages_gray"
mkdir_os(save_path)
mkdir_os(json_path)
mkdir_os(gray_path)
#path = "luosi_phone_seg_20191008_20191012_185541.json"
path = "luosi_phone_seg_20191008_20191014_194416.json"
data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))
 
num = 0
lendata_num = 0
count = len(data)
file_error = open('error_null_json_biaozhu.txt',"wr")
train_txt = open('train.txt',"wr")
test_txt = open('test.txt',"wr")
 
for lab in range(count):
    num += 1
    #print num,"/",count
    onedate = data[lab]
    name = onedate["url_image"]
    name = str(name).split("/")[-1]
 
    print(name)
 
    # if len(onedate["result"])<2:
    #     print name
    #     continue
    # X1 = onedate["result"][0]["data"]
    # Y1 = onedate["result"][1]["data"]
    # assert len(X1)==len(Y1),"error len"
    img = cv2.imread(os.path.join(ori_path,name))
    tempimg = copy.deepcopy(img)
    grayimg = copy.deepcopy(img)
    hh,ww,c = img.shape
    point_size = 1
    point_color = (0, 0, 255)
    thickness = 4
 
    # VID_20190924_143239_000511_img_good_931_735_1111_895
    if(len(onedate["result"])==0):
        print(name)
        file_error.write(name)
        file_error.write("\n")
        continue
    if name == "VID_20190924_145531_000961_img_bad_791_480_1006_675.jpg":
        a = 1
    if 'data' in onedate["result"] or 'data' in onedate["result"][0]:
        for key in range(len(onedate["result"])):
            ndata = onedate["result"][key]["data"]
            for k in range(len(ndata)/2):
                cv2.circle(img, (ndata[2*k],ndata[2*k+1]), point_size, point_color, thickness)
        grayimg=cv2.cvtColor(grayimg,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(gray_path,name),grayimg)
        cv2.imwrite(os.path.join(save_path,name),img)
    else:
        print(name)
        file_error.write(name)
        file_error.write("\n")
        continue
 
    json_jpg={}
    json_jpg["imagePath"] = str(os.path.join(ori_path,name))
    json_jpg["imageData"] = None
    shapes=[]
    #if name=="VID_20190924_142716_000261_img_good_670_716_772_813.jpg":
    #    sss=1
    #VID_20190924_145903
    #VID_20190924_145950
    for key in range(len(onedate["result"])):
        ndata = onedate["result"][key]["data"]
        points=[]
        if len(ndata)< 6:
            lendata_num += 1
            continue
        else:
            for k in range(len(ndata)/2):
                cv2.circle(img, (ndata[2*k],ndata[2*k+1]), point_size, point_color, thickness)
                points.append([ndata[2*k],ndata[2*k+1]])
        one_shape = {}
        one_shape["line_color"] = None
        one_shape["shape_type"] = "polygon"
        one_shape["points"] = points
        one_shape["flags"] = {}
        one_shape["fill_color"] = None
        one_shape["label"] = "luosi"
        shapes.append(one_shape)
 
    json_jpg["shapes"] = shapes
    json_jpg["version"] = "3.16.7"
    json_jpg["flags"] = {}
    json_jpg["fillColor"] = [
                                255, 
                                0, 
                                0, 
                                128
                            ]
    json_jpg["lineColor"] = [
                                0, 
                                255, 
                                0, 
                                128
                            ]
    json_jpg["imageWidth"] = ww
    json_jpg["imageHeight"] = hh
 
    jsonData = json.dumps(json_jpg, indent=4)
    jsonname = name.split(".")[0]
    jsonname = jsonname+".json"
    fileObject = open(os.path.join(json_path,jsonname), 'w')
    #cv2.imwrite(os.path.join(json_path,name),tempimg)
    fileObject.write(jsonData)
    fileObject.close()
 
    txtname = name.split(".")[0]
    if "VID_20190924_145903" in txtname or "VID_20190924_145950" in txtname:
        test_txt.write(txtname)
        test_txt.write("\n")
    else:
        train_txt.write(txtname)
        train_txt.write("\n")
 
print("lendata_num:",lendata_num)
file_error.close()
test_txt.close()
train_txt.close()
