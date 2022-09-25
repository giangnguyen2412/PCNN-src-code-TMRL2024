from modelvshuman.models.pytorch.simclr import simclr_resnet50x1
resnet = simclr_resnet50x1_supervised_baseline(pretrained=True, use_data_parallel=False)
print(resnet)
pass