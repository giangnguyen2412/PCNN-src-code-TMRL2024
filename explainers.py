from torchray.attribution.grad_cam import grad_cam


class ModelExplainer(object):
    def __init__(self):
        pass

    @staticmethod
    def grad_cam(model, images, target_ids, saliency_layer, resize):
        explanations = grad_cam(model, images, target_ids, saliency_layer=saliency_layer, resize=resize)
        return explanations

    @staticmethod
    def nearest_neighbors():
        # Tensor of three NNS here
        explanations = None
        return explanations
