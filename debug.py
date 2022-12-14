import numpy as np

kbc = np.load('faiss/faiss_CUB_200way_train_topk_HP_INAT.npy', allow_pickle=True, ).item()

print('hi')
pass

import os
def visualize_model2_decision_with_prototypes(query: str,
                                              gt_label: str,
                                              pred_label: str,
                                              model2_decision: str,
                                              save_path: str,
                                              save_dir: str,
                                              confidence1: int,
                                              confidence2: int,
                                              prototypes: list,
                                              ):
    save_dir = 'tmp'
    cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/query.jpeg".format(
        query, save_dir
    )
    os.system(cmd)
    annotation = gt_label
    cmd = 'convert {}/query.jpeg -font aakar -pointsize 10 -gravity North -background White -splice 0x40 -annotate +0+4 "{}" {}/query.jpeg'.format(
        save_dir, annotation, save_dir
    )
    os.system(cmd)
    for idx, prototype in enumerate(prototypes):
        if idx == 0 or idx > 3:  # skip the query and only print out the first 3 prototypes
            continue
        cmd = "convert '{}' -resize 256x256^ -gravity Center -extent 224x224 {}/{}.jpeg".format(
            prototype, save_dir, idx)
        os.system(cmd)
        annotation = prototype.split('/')[-2]
        cmd = 'convert {}/{}.jpeg -font aakar -pointsize 10 -gravity North -background White -splice 0x40 -annotate +0+4 "{}" {}/{}.jpeg'.format(
            save_dir, idx, annotation, save_dir, idx
        )
        os.system(cmd)

    cmd = 'montage {}/[1-3].jpeg -tile 3x1 -geometry +0+0 {}/aggregate.jpeg'.format(save_dir, save_dir)
    os.system(cmd)
    cmd = 'montage {}/query.jpeg {}/aggregate.jpeg -tile 2x -geometry +10+0 {}/{}.JPEG'.format(save_dir, save_dir, save_dir, gt_label)
    os.system(cmd)

cnt = 0
for query, nns in kbc.items():
    if cnt >= 10:
        break
    gt_label = nns[0].split('/')[-2]
    query = nns[0]
    visualize_model2_decision_with_prototypes(query=query,
                                              gt_label=gt_label,
                                              pred_label=None,
                                              model2_decision=None,
                                              save_path=None,
                                              save_dir=None,
                                              confidence1=None,
                                              confidence2=None,
                                              prototypes=nns)
    cnt += 1

cmd = 'img2pdf -o tmp/topk.pdf --pagesize A4^T tmp/*.JPEG'
os.system(cmd)


