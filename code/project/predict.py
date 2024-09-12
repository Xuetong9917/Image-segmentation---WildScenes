import numpy as np
from PIL import Image

from models.deeplab import DeeplabV3
from sklearn.metrics import jaccard_score
from prettytable import PrettyTable

name_classes    = ["unlabelled",
        "asphalt",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush",
        "fence",
        "structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass"]

mIOU_dict = {}

def calculate_jsc(y_true, y_pred, num_classes):
    class_jsc = []
    # Make sure the input is a numpy array
    y_true = np.asarray(y_true, dtype=np.int32)
    
    y_true[y_true == 1] = 6
    y_pred[y_pred == 1] = 6
    y_true[y_true == 12] = 16
    y_pred[y_pred == 12] = 16

    for cls in range(num_classes):
        if cls != 13:
            true_class = (y_true == cls)
            pred_class = (y_pred == cls)
            if true_class.sum() != 0:
                if pred_class.sum() != 0:
                    jsc = jaccard_score(true_class.flatten(), pred_class.flatten(), average='binary', zero_division=0)        
                    if name_classes[cls] in mIOU_dict.keys():
                        mIOU_dict[name_classes[cls]][0] += jsc
                        mIOU_dict[name_classes[cls]][1] += 1
                    else:
                        mIOU_dict[name_classes[cls]] = [jsc, 1]
                    class_jsc.append(jsc)
                else:
                    class_jsc.append(0)
                    if name_classes[cls] in mIOU_dict.keys():
                        mIOU_dict[name_classes[cls]][0] += 0
                        mIOU_dict[name_classes[cls]][1] += 1
                    else:
                        mIOU_dict[name_classes[cls]] = [0, 1]

    return np.mean(class_jsc), class_jsc

if __name__ == "__main__":
    print('Starting prediction...')
    deeplab = DeeplabV3()
    img_org_path = 'D:/jue/Data/JPEGImages/'
    img_true_path = 'D:/jue/Data/SegmentationClass/'
    img_path_txt = 'C:/Users/Asus/Desktop/COMP9517/project/code/project/split/test_1.txt'

    line_count = 0
    total_jsc = 0
    with open(img_path_txt, 'r') as file:
        for line in file:
            line = line.strip()
            img_pre = img_org_path + line + '.jpg'
            img_true = img_true_path + line + '.png'
            line_count += 1

            image_true = Image.open(img_true)
            image_true = np.array(image_true)
            image = Image.open(img_pre)
            r_image = deeplab.detect_image(image)
            # r_image_array = np.array(r_image.convert('L'))
            # r_image.show()

            mean_jsc, class_jsc = calculate_jsc(image_true, r_image, 19)
            print(f'count: {line_count} jsc: {mean_jsc, class_jsc}')
            total_jsc += mean_jsc

    table = PrettyTable()
    table.field_names = ["Label", "IOU"]

    true_label_num = 0
    sum_IOU = 0
    
    for i in range(19):
        if name_classes[i] in mIOU_dict.keys():
            true_label_num += 1
            value = mIOU_dict[name_classes[i]]
            iou = value[0] / value[1] if value[1] != 0 else 0
            sum_IOU += iou
            table.add_row([name_classes[i], iou])
    print('------------------Table---------------------')
    print(table)
    print('--------------------------------------------')
    average_jsc = total_jsc / line_count
    print(f'Average Jaccard Score: {average_jsc}')
    print(f'mIOU Score: {sum_IOU / true_label_num}')
