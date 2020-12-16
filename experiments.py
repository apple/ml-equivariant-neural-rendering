import json
import os
import sys
import time
import torch
from misc.dataloaders import scene_render_dataloader
from models.neural_renderer import NeuralRenderer
from training.training import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get path to data from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python experiments.py <config>"))
path_to_config = sys.argv[1]

# Open config file
with open(path_to_config) as file:
    config = json.load(file)

# Set up directory to store experiments
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, config["id"])
if not os.path.exists(directory):
    os.makedirs(directory)

# Save config file in directory
with open(directory + '/config.json', 'w') as file:
    json.dump(config, file)

# Set up renderer
model = NeuralRenderer(
    img_shape=config["img_shape"],
    channels_2d=config["channels_2d"],
    strides_2d=config["strides_2d"],
    channels_3d=config["channels_3d"],
    strides_3d=config["strides_3d"],
    num_channels_inv_projection=config["num_channels_inv_projection"],
    num_channels_projection=config["num_channels_projection"],
    mode=config["mode"]
)

model.print_model_info()

model = model.to(device)

if config["multi_gpu"]:
    model = torch.nn.DataParallel(model)

# Set up trainer for renderer
trainer = Trainer(device, model, lr=config["lr"],
                  rendering_loss_type=config["loss_type"],
                  ssim_loss_weight=config["ssim_loss_weight"])

dataloader = scene_render_dataloader(path_to_data=config["path_to_data"],
                                     batch_size=config["batch_size"],
                                     img_size=config["img_shape"],
                                     crop_size=128)

# Optionally set up test_dataloader
if config["path_to_test_data"]:
    test_dataloader = scene_render_dataloader(path_to_data=config["path_to_test_data"],
                                              batch_size=config["batch_size"],
                                              img_size=config["img_shape"],
                                              crop_size=128)
else:
    test_dataloader = None

print("PID: {}".format(os.getpid()))

# Train renderer, save generated images, losses and model
trainer.train(dataloader, config["epochs"], save_dir=directory,
              save_freq=config["save_freq"], test_dataloader=test_dataloader)

# Print best losses
print("Model id: {}".format(config["id"]))
print("Best train loss: {:.4f}".format(min(trainer.epoch_loss_history["total"])))
print("Best validation loss: {:.4f}".format(min(trainer.val_loss_history["total"])))
