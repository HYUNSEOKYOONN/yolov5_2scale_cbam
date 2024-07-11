import torch
import torch_pruning as tp
from yolov5 import YOLOv5

# Load pre-trained YOLOv5 model
model = YOLOv5('yolov5s.pt')

# Define a function to prune the model
def prune_model(model, amount=0.2):
    # Select parameters to prune (you can choose other layers as well)
    parameters_to_prune = []
    for module_name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    # Create a pruner object
    pruner = tp.pruner.MagnitudePruner(parameters_to_prune)

    # Prune the model
    pruner.step(amount)

    return model

# Prune the model
pruned_model = prune_model(model)

# Save the pruned model
torch.save(pruned_model.state_dict(), 'yolov5s_pruned.pt')
