import json
import os

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from glob import glob
from collections import defaultdict


def get_acc_p_r_f1(file_path, is_loc=False):
    acc, p_sum, r_sum, f1_sum = 0.0, 0.0, 0.0, 0.0
    # 加载数据（假设从文件读取）
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取标签
    y_true = [item["ground_truth"] for item in data]
    y_pred = [item["output"] for item in data]

    # 允许1-2个样本点误差
    if is_loc:
        for idx, (a, b) in enumerate(zip(y_true, y_pred)):
            if abs(a - b) < 3 and a != 0:
                y_pred[idx] = a

    class_num = defaultdict(int)
    all_class = defaultdict(dict)
    for k in y_true:
        class_num[k] += 1
    len_class = len(class_num.keys())
    for o in class_num.keys():
        one_class = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == o and p == o:
                one_class['TP'] += 1
            elif p == o and t != o:
                one_class['FP'] += 1
            elif t == o and p != o:
                one_class['FN'] += 1
            elif t != o and p != o:
                one_class['TN'] += 1
        all_class[o] = one_class
    for n in all_class.values():
        p = n['TP'] / (n['TP'] + n['FP'])
        r = n['TP'] / (n['TP'] + n['FN'])
        p_sum += p
        r_sum += r
        f1_sum += (2 * p * r) / (p + r)
        acc += n['TP'] / sum(n.values())
    print(all_class)
    return acc, p_sum/len_class, r_sum/len_class, f1_sum/len_class


def get_metrics(file_path, is_loc=False):
    # 加载数据（假设从文件读取）
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取标签
    y_true = [item["ground_truth"] for item in data]
    y_pred = [item["output"] for item in data]
    infer_times = [float(item["infer_time"]) for item in data]

    # 允许1-2个样本点误差
    if is_loc:
        for idx, (a, b) in enumerate(zip(y_true, y_pred)):
            if abs(a - b) < 3 and a != 0:
                y_pred[idx] = a

    # 计算分类指标
    acc = accuracy_score(y_true, y_pred)
    # acc = calculate_accuracy(y_true, y_pred, is_loc=is_loc)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)

    # 计算ROC-AUC（需要数值化标签）
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")

    return acc, precision, recall, f1, mcc, roc_auc, sum(infer_times)/len(infer_times)


if __name__ == '__main__':
    is_loc = False
    if is_loc:
        l_res = glob('results_loc/*.json')
    else:
        l_res = glob('results/*.json')
    # print(l_res)
    for i in l_res:
        acc, precision, recall, f1, mcc, roc_auc, times = get_metrics(i, is_loc=is_loc)
        filename = os.path.split(os.path.dirname(i))[-1] + '/' + os.path.basename(i)
        # 打印结果
        print('#'*10, filename, '#'*10)

        print(f"ACC: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Times: {times:.6f}")

        # print(get_acc_p_r_f1(i, is_loc=is_loc))
        print('#' * len('#'*10+os.path.basename(i)+'#'*10))
