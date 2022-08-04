from torchray.attribution.grad_cam import grad_cam
from params import RunningParams

import torch

RunningParams = RunningParams()


class ModelExplainer(object):
    def __init__(self):
        # TODO: pass XAI params here. e.g. k in kNNs or layer in GradCAM
        pass

    @staticmethod
    def grad_cam(model, images, target_ids, saliency_layer, resize):
        explanations = grad_cam(model, images, target_ids, saliency_layer=saliency_layer, resize=resize)
        return explanations

    @staticmethod
    def faiss_nearest_neighbors(model1, embeddings, phase, faiss_gpu_index, faiss_data_loader):
        # TODO: later I may need the path of NNs here for visualization
        distance, indices = faiss_gpu_index.search(embeddings, RunningParams.k_value + 1)

        # Check here if in validation we retrieved the wrong exemplar --> making model always agree

        # Running two loops to get the NNs' processed tensors
        q_list = []
        for q_idx in range(embeddings.shape[0]):
            n_list = []
            for n_idx in range(RunningParams.k_value):
                if phase == 'train':
                    input, _ = faiss_data_loader.dataset[indices[q_idx][n_idx+1]]
                else:
                    input, _ = faiss_data_loader.dataset[indices[q_idx][n_idx]]
                n_list.append(input)
            n_tensors = torch.stack(n_list)
            q_list.append(n_tensors)

        explanations = torch.stack(q_list)
        return explanations