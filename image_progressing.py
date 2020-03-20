#---------------------------------------------------#
#   对指定文件夹图像进行压缩:
#   
#---------------------------------------------------#


import os
from PIL import Image

image_dir = 'JPEGImages/'             # 输入RGB图像文件夹
label_dir = 'SegmentationClassRaw/'   # 输入label图像文件

image_output_dir = 'JPEGImages_resized/'             # 输出RGB图像文件夹
label_output_dir = 'SegmentationClassRaw_resized/'   # 输出label图像文件夹

if not os.path.exists(image_output_dir):
  os.mkdir(image_output_dir)
if not os.path.exists(label_output_dir):
  os.mkdir(label_output_dir)

image_format = '.jpg'
label_format = '.png'
image_names = open('ImageSets/Segmentation/trainval.txt', 'r').readlines()
image_names = list(map(lambda x: x.strip(), image_names))

backup_file = open('ImageSizeBackUP.txt', 'w')      # 图像原始尺寸记录下来，备份

for name in image_names:
  # Open an image file and print the size
  image = Image.open(image_dir+name+image_format)
  label = Image.open(label_dir+name+label_format).convert('L')

  # Check image size
  assert image.size == label.size

  # Check image mode
  assert image.mode == 'RGB'

  print('>> Now checking image file: %s', name+image_format)
  print('   Origin size: ', image.size)
  width, height = image.size

  # Log the original size of the image
  backup_file.write('%d,%d\n' % (width, height))

  # Resize a image if it's too large
  if width > 512 or height > 512:
    print('   Resizing...')
    scale = 512.0 / max(image.size)
    resized_w = int(width*scale)
    resized_h = int(height*scale)
    image = image.resize((resized_w, resized_h), Image.BILINEAR)
    label = label.resize((resized_w, resized_h), Image.NEAREST)

    # Save new image and label
    image.save(image_output_dir+name+image_format)
    label.save(label_output_dir+name+label_format)

    print('   Done.')
  else:
    # Save new image and label
    image.save(image_output_dir+name+image_format)
    label.save(label_output_dir+name+label_format)
    print('   Pass.')
backup_file.close()