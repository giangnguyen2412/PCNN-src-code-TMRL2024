import numpy as np

result_file = 'infer_results/best_model_gentle-shadow-1025.pt'
kbc = np.load('{}.npy'.format(result_file), allow_pickle=True, ).item()
T_list = list(np.arange(0.0, 1.05, 0.05))
very_confidence = 0
acc_T_dict = []
for T in T_list:
    T = int(T*100)
    T_sample_cnt = 0
    T_correct_cnt = 0
    for key in kbc.keys():
        confidence1 = kbc[key]['confidence1']
        if confidence1 >= 95:
            very_confidence += 1
        if confidence1 >= T:
            continue
        T_sample_cnt += 1
        result = kbc[key]['result']
        if result is True:
            T_correct_cnt += 1
    if T_sample_cnt == 0:
        print("Acc is N/A at T: {}".format(T))
        # acc_T_dict[T] = 'N/A'
        acc_dict = dict()
        acc_dict['Threshold'] = T
        acc_dict['Accuracy'] = 'N/A'
        acc_T_dict.append(acc_dict)
    else:
        acc = (T_correct_cnt/T_sample_cnt)
        # acc = '{:.2f}'.format(acc)
        print("Accuracy at T: {} is {}%".format(T, acc))
        print(T_correct_cnt, T_sample_cnt)
        acc_dict = dict()
        acc_dict['Threshold'] = T
        acc_dict['Accuracy'] = acc
        acc_T_dict.append(acc_dict)

print(very_confidence)
import csv

field_names = ['Threshold', 'Accuracy']
with open('{}.csv'.format(result_file), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(acc_T_dict)








