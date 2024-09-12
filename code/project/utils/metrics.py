import csv
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Calculate F-Score
def f_score(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), dim=-1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = (temp_inputs > threshold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return torch.mean(score)

# Calculate confusion matrix
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# Calculate each type of IoU
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

# Calculate PA/Recall for each type
def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

# Calculate Precision for each type
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

# Calculate overall accuracy
def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

# Calculate indicators such as mIoU
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind, gt_img in enumerate(gt_imgs):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_img))

        if len(label.flatten()) != len(pred.flatten()):
            print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_img}, {pred_imgs[ind]}')
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if name_classes and ind > 0 and ind % 10 == 0:
            print(f'{ind} / {len(gt_imgs)}: mIoU-{np.nanmean(per_class_iu(hist)) * 100:.2f}%; mPA-{np.nanmean(per_class_PA_Recall(hist)) * 100:.2f}%; Accuracy-{per_Accuracy(hist) * 100:.2f}%')

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    if name_classes:
        for ind_class in range(num_classes):
            print(f'===> {name_classes[ind_class]}: Iou-{IoUs[ind_class] * 100:.2f}; Recall-{PA_Recall[ind_class] * 100:.2f}; Precision-{Precision[ind_class] * 100:.2f}')

    print(f'===> mIoU: {np.nanmean(IoUs) * 100:.2f}; mPA: {np.nanmean(PA_Recall) * 100:.2f}; Accuracy: {per_Accuracy(hist) * 100:.2f}')
    return hist, IoUs, PA_Recall, Precision

# Adjust chart axes
def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

# Draw a bar chart
def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = f" {val:.2f}" if val < 1.0 else f" {val}"
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == len(values) - 1:
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

# Display results and save charts and confusion matrices
def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, f"mIoU = {np.nanmean(IoUs) * 100:.2f}%", "Intersection over Union", os.path.join(miou_out_path, "mIoU.png"), tick_font_size, True)
    print(f"Save mIoU out to {os.path.join(miou_out_path, 'mIoU.png')}")

    draw_plot_func(PA_Recall, name_classes, f"mPA = {np.nanmean(PA_Recall) * 100:.2f}%", "Pixel Accuracy", os.path.join(miou_out_path, "mPA.png"), tick_font_size, False)
    print(f"Save mPA out to {os.path.join(miou_out_path, 'mPA.png')}")

    draw_plot_func(PA_Recall, name_classes, f"mRecall = {np.nanmean(PA_Recall) * 100:.2f}%", "Recall", os.path.join(miou_out_path, "Recall.png"), tick_font_size, False)
    print(f"Save Recall out to {os.path.join(miou_out_path, 'Recall.png')}")

    draw_plot_func(Precision, name_classes, f"mPrecision = {np.nanmean(Precision) * 100:.2f}%", "Precision", os.path.join(miou_out_path, "Precision.png"), tick_font_size, False)
    print(f"Save Precision out to {os.path.join(miou_out_path, 'Precision.png')}")

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + name_classes)
        for i, row in enumerate(hist):
            writer.writerow([name_classes[i]] + row.tolist())
    print(f"Save confusion_matrix out to {os.path.join(miou_out_path, 'confusion_matrix.csv')}")
