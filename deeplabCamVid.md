**注意：因为Deeplab于2019年11月20日左右在github上进行了更新，文档中部分地方区分了旧版（课程使用的版本）和新版（更新后的github版本）.**

学员可使用百度网盘旧版deeplab目录下载旧版models.tar.gz或者使用从github上克隆新版deeplab进行项目实践。

### Deeplab项目安装及测试

#### 升级tensorflow版本至1.10以上：

例如：使用1.15.0

```
sudo pip install --upgrade --ignore-installed tensorflow-gpu==1.15.0  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Anaconda环境（例如环境名为tfgpu）下安装命令：

```
conda install tensorflow-gpu==1.15.0
```

#### 克隆deeplab项目

官方代码github地址：<https://github.com/tensorflow/models/tree/master/research/deeplab>

```
git clone https://github.com/tensorflow/models.git
```

####  添加项目依赖路径
**分window系统和linux系统部署两种情况：**
window系统：在train.py, vis.py, eval.py, export_model.py   等程序最前端添加如下代码：

```python
import sys
sys.path.append(r'E:\PycharmProject\DeepLab_model\models\research')
sys.path.append(r'E:\PycharmProject\DeepLab_model\models\research\slim')  
# 以上两处的路径请按照真实的路径添加
```
linux系统：分两种情况，终端添加和环境变量添加
1.在终端添加如下命令：一次性买卖，仅当前终端有效

`export PYTHONPATH=/home/bai/models/research/slim:/home/bai/models/research:$PYTHONPATH`

2.在环境变量中添加：当前用户都有效
编辑文件~/.bashrc

```
sudo gedit ~/.bashrc
```

末尾加上一句:
```python
export PYTHONPATH=/home/bai/models/research/slim:/home/bai/models/research:$PYTHONPATH
```
执行

```
source ~/.bashrc
```

#### 测试deeplab

```
cd /home/bai/models/research/deeplab
```

执行：

```
python model_test.py
```

最后输出：OK

---------------------

### 数据集处理

#### 数据集处理分成三步：

- 标注数据（对CamVid省略）

- 制作指引文件

- 将数据转换成TFRecord

#### CamVid数据集下载

下载地址：百度网盘链接

```
链接：https://pan.baidu.com/s/1YDwnIpyh-X4TY31teKMDfg 
提取码：nhrp 
```

数据集的文件夹结构为：

├── test
├── testannot
├── train
├── trainannot
├── val
└── valannot

由上到下分别是测试集、测试集标签、训练集、训练集标签、验证集、验证集标签。

对于CamVid, 其中训练集、测试集、验证集中的图片数目分别为train 367， test 233，val 101。

#### 制作指引文件

TF 提供了一种统一输入数据的格式—— TFRecord
它有两个优点：

1. 可以将一个样本的所有信息统一起来存储，这些信息可以是不同的数据类型。其内部使用“Protocol Buffer”二进制数据编码方案。

2. 利用文件队列的多线程操作，使得数据的读取和批量处理更加方便快捷。



在制作TFRecord之前，需要有文件指引将数据集分类成训练集、测试集、验证集，故需要创建指引文件。

将所有图片和mask分在两个文件夹下，设置如下：

/home/bai/dataset/CamVid/image: 存放所有的输入图片，共有701张，这其中包括训练集、测试集、验证集的图片。

/home/bai/dataset/CamVid/mask：存放所有的标签图片，共有701张，和image文件夹下的图片是一一对应的。

对于CamVid数据集，创建了一个目录/home/bai/dataset/CamVid/index，该目录下包含三个.txt文件：

- train.txt：所有训练集的文件名称
- trainval.txt：所有验证集的文件名称
- val.txt：所有测试集的文件名称

#### 将数据转换成TFRecord

创建文件夹tfrecord:

```
mkdir tfrecord
```

将上述制作的数据集打包成TFRecord，使用的是build_voc2012_data.py

在目录/home/bai/models/research/deeplab/datasets下执行

```python
python build_voc2012_data.py \
  --image_folder="/home/bai/dataset/CamVid/image" \
  --semantic_segmentation_folder="/home/bai/dataset/CamVid/mask" \
  --list_folder="/home/bai/dataset/CamVid/index" \
  --image_format="png" \
  --output_dir="/home/bai/dataset/CamVid/tfrecord"
```

- image_folder ：数据集中原输入数据的文件目录地址
- semantic_segmentation_folder：数据集中标签的文件目录地址
- list_folder : 将数据集分类成训练集、验证集等的指引目录文件目录
- image_format : 输入图片数据的格式，CamVid的是png格式

- output_dir：制作的TFRecord存放的目录地址(自己创建)



### 网络训练

#### 修改训练脚本

在DeepLabv3+模型的基础上，主要需要修改以下两个文件

- `data_generator.py`
- `train_utils.py`

##### 添加数据集描述

在datasets/data_generator.py文件中，添加camvid数据集描述：


```python
_CAMVID_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 367,  # num of samples in images/training
        'val': 101,  # num of samples in images/validation
    },
    num_classes=12,
    ignore_label=255,
)
```

因为CamVid共有11个classes，所以加上`ignore_label`共12个物体类别。

##### 注册数据集

同时在datasets/data_generator.py文件，添加对应数据集的名称：

```python
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'camvid':_CAMVID_INFORMATION, #camvid示例
    'mydata':_MYDATA_INFORMATION, #自己的数据集
}
```

##### 修改train_utils.py

对应的utils/train_utils.py中，将159行关于`exclude_list`的设置修改，作用是在使用预训练权重时候，不加载该`logit`层：

```python
exclude_list = ['global_step','logits']
if not initialize_last_layer:
exclude_list.extend(last_layers)
```

---------------------

#### 修改train.py

如果想在DeepLab的基础上fine-tune其他数据集， 可在deeplab/train.py中修改输入参数。

其中有一些选项：

- 使用预训练的所有权重，设置initialize_last_layer=True
- 只使用网络的backbone，设置initialize_last_layer=False和last_layers_contain_logits_only=False
- 使用所有的预训练权重，除了logits以外。因为如果是自己的数据集，对应的classes不同(这个我们前面已经设置不加载logits), 可设置initialize_last_layer=False和last_layers_contain_logits_only=True

使用的设置是：

initialize_last_layer=False  #133行 (旧版)

last_layers_contain_logits_only=True  #136行 （旧版）



initialize_last_layer=False  #157行 (新版)

last_layers_contain_logits_only=True  #160行 （新版）

#### 下载预训练模型

在model_zoo上下载预训练模型：

下载地址:<https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md>

下载的预训练权重为xception_cityscapes_trainfine

文件大小为439M



下载到deeplab目录下，然后解压：

tar -zxvf deeplabv3_cityscapes_train_2018_02_06.tar.gz

需要注意对应的解压文件目录为：

```
/home/bai/models/research/deeplab/deeplabv3_cityscapes_train
```

注意：在训练CamVid时，没有考虑类别之间的的平衡问题，所以没有做imblance的修正。如果你是二分类或者存在严重的类别不平衡情况，可以参考后面roadscene分割项目案例中的类别不平衡修正方法。

注意如下几个参数：

- tf_initial_checkpoint：预训练的权重，因为CamVid和自己的RoadScene数据集都和CityScapes类似，所以使用的是CityScapes的预训练权重

- train_logdir: 训练产生的文件存放位置

- dataset_dir: 数据集的TFRecord文件
- dataset：设置为在data_generator.py文件设置的数据集名称

#### 在CamVid上的训练指令如下：

在目录 ~/models/research/deeplab下执行

```python
 python train.py \
    --logtostderr \
    --training_number_of_steps=300 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=321,321 \
    --train_batch_size=4 \
    --dataset="camvid" \
    --tf_initial_checkpoint='/home/bai/models/research/deeplab/deeplabv3_cityscapes_train/model.ckpt' \
    --train_logdir='/home/bai/models/research/deeplab/exp/camvid_train/train' \
    --dataset_dir='/home/bai/dataset/CamVid/tfrecord'
```

Q: 如何设置train_crop_size的值?

A: output_stride * k + 1, where k is an integer. For example, we have 321x321，513x513 。

#### 测试结果可视化：

在目录 ~/models/research/deeplab下执行

```python
python vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=360,480 \
    --dataset="camvid" \
    --colormap_type="pascal" \
    --checkpoint_dir='/home/bai/models/research/deeplab/exp/camvid_train/train' \
    --vis_logdir='/home/bai/models/research/deeplab/exp/camvid_train/vis' \
    --dataset_dir='/home/bai/dataset/CamVid/tfrecord'
```

- vis_split：设置为测试集

- vis_crop_size:设置360,480为图片的大小

- dataset：设置为我们在data_generator.py文件设置的数据集名称

- dataset_dir：设置为创建的TFRecord

- colormap_type：可视化标注的颜色

  可到目录deeplab/exp/camvid_train/vis下查看可视化结果

#### 性能评估：

在目录 ~/models/research/deeplab下执行

```python
python eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=360,480 \
    --dataset="camvid" \
    --checkpoint_dir='/home/bai/models/research/deeplab/exp/camvid_train/train' \
    --eval_logdir='/home/bai/models/research/deeplab/exp/camvid_train/eval' \
    --dataset_dir='/home/bai/dataset/CamVid/tfrecord' \
    --max_number_of_evaluations=1
```

- eval_split：设置为测试集
- crop_size：同样设置为360和480
- dataset：设置为camvid
- dataset_dir：设置为我们创建的数据集

##### 查看mIoU值：

tensorboard --logdir /home/bai/models/research/deeplab/exp/camvid_train/eval

##### 查看训练过程的loss：

tensorboard --logdir /home/bai/models/research/deeplab/exp/camvid_train/train







