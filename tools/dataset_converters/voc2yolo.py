import argparse
import os
import time
import shutil
import numpy as np
import os.path as osp
from pathlib import Path
import xml.dom.minidom

dataset_class_names_dic = {"safety belt" : 0, "not safety belt" : 1, "person" : 2, "wheel" : 3, 
                     "dark phone" : 4, "bright phone" : 5, "hand" : 6} # "ignore"

create_class_names_dic  = {"safety_belt" : 0, "not_safety_belt" : 1, "person" : 2, "wheel" : 3, 
                     "dark_phone" : 4, "bright_phone" : 5, "hand" : 6} # "ignore"
def convert_to_center(size, box):
    '''
    size: (w,h)
    box: xmin,xmax,ymin,ymax
    '''
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    # x = (box[0] + box[1]) / 2.0 - 1
    # y = (box[2] + box[3]) / 2.0 - 1
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def get_box_from_voc_label(label_path):
    '''
    return [[class_name, xmin, ymin, xmax, ymax]...], img_w,img_h,img_chn
    '''
    DOMTree = xml.dom.minidom.parse(label_path)
    collection = DOMTree.documentElement

    img_w = collection.getElementsByTagName("width")[0].childNodes[0].data
    img_h = collection.getElementsByTagName("height")[0].childNodes[0].data
    img_chn = 3
    img_w, img_h, img_chn = int(img_w), int(img_h), int(img_chn)

    objs = collection.getElementsByTagName("object")

    obj_boxes = []

    for curr_obj in objs:
        class_name = curr_obj.getElementsByTagName('name')[0].childNodes[0].data
        if class_name not in dataset_class_names_dic:
             continue
        if class_name in dataset_class_names_dic:
            class_name = class_name.replace(' ', '_')
        box_coord = curr_obj.getElementsByTagName('bndbox')[0]
        xmin = box_coord.getElementsByTagName('xmin')[0].childNodes[0].data
        ymin = box_coord.getElementsByTagName('ymin')[0].childNodes[0].data
        xmax = box_coord.getElementsByTagName('xmax')[0].childNodes[0].data
        ymax = box_coord.getElementsByTagName('ymax')[0].childNodes[0].data
        obj_boxes.append([class_name, xmin, ymin, xmax, ymax])

    return obj_boxes, img_w,img_h,img_chn

def check_existence(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')

def convert_voc_to_yolo(image_dir: str, out_dir: str):
    """
        Convert annotations from voc style to yolo style.
    """
    print(f'Start to load existing images and annotations from {image_dir}')
    check_existence(image_dir)

    out_dir = osp.join(out_dir, 'MMYOLO_yoloFromat_%s'%(time.strftime("%Y-%m-%d", time.localtime())))
    if not osp.exists(out_dir):
        print(f'Output Save Path {out_dir}')
        os.makedirs(out_dir, exist_ok=True)

    classesfile = osp.join(out_dir, 'classes.txt')
    classes_content = []
    for classes in create_class_names_dic.keys():
         c_content = '{}\n'.format(classes)
         classes_content.append(c_content)
    with open(classesfile, mode='w', encoding='utf-8') as f:
            f.writelines(classes_content)

    # check local environment
    voc_label_dir = osp.join(image_dir, 'Annotations')
    voc_image_dir = osp.join(image_dir, 'JPEGImages')
    check_existence(voc_label_dir)
    check_existence(voc_image_dir)
    yolo_labels_dir = osp.join(out_dir, 'labels')
    yolo_images_dir = osp.join(out_dir, 'images')
    if not osp.exists(yolo_labels_dir):
        os.makedirs(yolo_labels_dir, exist_ok=True)
    if not osp.exists(yolo_images_dir):
        os.makedirs(yolo_images_dir, exist_ok=True)

    print(f'All necessary files are located at {image_dir}')

    # start the convert procedure
    xmliteration = Path(voc_label_dir).rglob('*.xml')

    train_img_txt = open(osp.join(out_dir, 'train.txt'), 'w', encoding="utf-8")
    val_img_txt   = open(osp.join(out_dir, 'val.txt'), 'w', encoding="utf-8")
    # test_img_txt  = open(osp.join(out_dir, 'test.txt'), 'w', encoding="utf-8")

    i=0
    for xmlfile in xmliteration:
        i+=1
        if i == 1000:
             break
        yolofilename = xmlfile.name.replace('xml', 'txt')
        yolofilesavepath = osp.join(yolo_labels_dir, yolofilename)

        xmlfile = str(xmlfile)
        imgfile = xmlfile.replace('Annotations', 
                                'JPEGImages').replace('xml', 
                                'jpg').replace('xml', 'png')
        
        check_existence(xmlfile)
        check_existence(imgfile)

        shutil.copy(imgfile, yolo_images_dir)

        imgfilename = osp.join(yolo_images_dir, osp.basename(imgfile))
        check_existence(imgfilename)
        if imgfile.find('/train/') != -1:
             train_img_txt.write('%s\n'%(imgfilename))
        elif imgfile.find('/val/') != -1:
             val_img_txt.write('%s\n'%(imgfilename))
        else:
            raise FileNotFoundError(f'{imgfilename} does not exist!')

        voc_box, img_w, img_h, _ = get_box_from_voc_label(xmlfile)
        save_lines = []
        for curr_box in voc_box:
            curr_class_id = create_class_names_dic[curr_box[0]]
            if curr_box[1] == curr_box[3] or curr_box[2] == curr_box[4]:
                print('err, zero img width or height: %s'%(xmlfile))
            xmin = int(float(curr_box[1]))
            ymin = int(float(curr_box[2]))
            xmax = int(float(curr_box[3]))
            ymax = int(float(curr_box[4]))
            box_coord = [xmin,xmax,ymin,ymax]
            x, y, w, h = convert_to_center((img_w, img_h), box_coord)
            line = "{} {} {} {} {}\n".format(curr_class_id, x, y, w, h)
            save_lines.append(line)

        with open(yolofilesavepath, mode='w', encoding='utf-8') as f:
                f.writelines(save_lines)
    train_img_txt.close()
    val_img_txt.close()
    # test_img_text.clone()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'image_dir',
        type=str,
        help='dataser directory'
    )
    parser.add_argument(
        'out_dir',
        type=str,
        help='save dataset path'
    )
    arg = parser.parse_args()
    convert_voc_to_yolo(arg.image_dir, arg.out_dir)