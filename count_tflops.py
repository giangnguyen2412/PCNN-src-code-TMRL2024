import torch
import torchvision.models as models
from thop import profile
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    ################################ INAT - RN50
    from iNat_resnet import ResNet_AvgPool_classifier, Bottleneck

    model = ResNet_AvgPool_classifier(Bottleneck, [3, 4, 6, 4])
    my_model_state_dict = torch.load(
        'pretrained_models/iNaturalist_pretrained_RN50_85.83.pth',
        map_location=torch.device('cuda:0')  # Load directly onto GPU
    )
    model.load_state_dict(my_model_state_dict, strict=True)
    model = model.cuda()  # Ensure model is on GPU

    # Use a dummy input tensor of shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1000, 3, 224, 224).cuda()

    tflops, _ = profile(model, inputs=(input_tensor,))
    tflops = tflops / 1e12
    print(f"TFLOPs for ResNet50: {tflops:.2f}")

    ################################
    from transformer import Transformer_AdvisingNetwork
    import torch.nn as nn

    model_path = 'best_models/best_model_decent-mountain-3215.pt'

    model = Transformer_AdvisingNetwork()
    # model = nn.DataParallel(model).cuda()
    #
    # checkpoint = torch.load(model_path, map_location=torch.device('cuda:0'))  # Load directly onto GPU
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()  # Ensure model is on GPU

    explanation = torch.randn(1000, 3, 224, 224).cuda()
    model1_score = torch.randn(1000, 1).cuda()

    tflops, _ = profile(model, inputs=(input_tensor, explanation, model1_score))
    tflops = tflops / 1e12
    print(f"TFLOPs for Transformer_AdvisingNetwork: {tflops:.2f}")
