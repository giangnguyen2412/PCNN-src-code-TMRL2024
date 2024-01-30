import numpy as np
import os

cnt = 0
overlap = 0
crt_cnt = 0
filename = '/home/anonymous/Downloads/advising_network/faiss/cub/ViT_top10_k1_enriched_train_NN1th.npy'
cnt_dict = {}

kbc = np.load(filename, allow_pickle=True, ).item()
new_kbc = dict()

for k, v in kbc.items():
    NN_class = os.path.basename(os.path.dirname(v['NNs'][0])).split('.')[1]
    # print(k)
    # print(NN_class)
    # print(v['label'])
    if v['label'] == 1:
        if NN_class.lower() in k.lower():
            continue
        else:
            print('Error!')
            exit(-1)
    else:
        if NN_class.lower() not in k.lower():
            continue
        else:
            print('Error!')
            exit(-1)

for k, v in kbc.items():

    if v['label'] == 1:
        cnt+=1

    if '_0_0_' in k:
        if v['label'] == 1:
            crt_cnt += 1

    k_base_name = k.split('_')
    k_base_name = ('_').join(k_base_name[3:])

    for nn in v['NNs']:
        base_name = os.path.basename(nn)
        if base_name in k:
            print("sth wrong")
            print(v)
            print(k)
            overlap += 1
            break
        else:
            new_kbc[k] = v
print(cnt)
print(cnt*100/len(kbc))
print(len(kbc))
print(overlap)
print(len(new_kbc))
# np.save(filename, new_kbc)
#