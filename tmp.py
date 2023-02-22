import numpy as np
from datasets import ImageFolderForNNs

filename = 'faiss/faiss_SDogs_val_RN34_topk.npy'
kbc = np.load(filename, allow_pickle=True, ).item()

cnt = 0
for query, nns in kbc.items():
    if 'train' in filename:
        nn_label = nns[1].split('/')[-2]
    else:
        nn_label = nns[0].split('/')[-2]

    nn_label = nn_label.split('.')[1]
    nn_label = nn_label.upper()
    query = query.upper()
    if nn_label in query:
        cnt += 1
    else:
        print(nn_label, query)


print(cnt*100/len(kbc))
print(cnt)
print(len(kbc))



