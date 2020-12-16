import sys
import torch
from misc.dataloaders import scene_render_dataset
from misc.quantitative_evaluation import get_dataset_psnr
from models.neural_renderer import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get path to experiment folder from command line arguments
if len(sys.argv) != 3:
    raise(RuntimeError("Wrong arguments, use python experiments_psnr.py <model_path> <dataset_folder>"))
model_path = sys.argv[1]
data_dir = sys.argv[2]  # This is usually one of "chairs-test" and "cars-test"

# Load model
model = load_model(model_path)
model = model.to(device)

# Initialize dataset
dataset = scene_render_dataset(path_to_data=data_dir, img_size=(3, 128, 128),
                               crop_size=128, allow_odd_num_imgs=True)

# Calculate PSNR
with torch.no_grad():
    psnrs = get_dataset_psnr(device, model, dataset, source_img_idx_shift=64,
                             batch_size=125, max_num_scenes=None)
