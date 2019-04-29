import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import argparse
import json
import utilities
from network import Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser_sec = argparse.ArgumentParser()
parser_sec.add_argument('b_img_path', action = 'store') # image path
parser_sec.add_argument('checkpoint', action = 'store') # checkpoint file
parser_sec.add_argument('--top_k', nargs = '?', type = int)
parser_sec.add_argument('--category_names', nargs = '?')
parser_sec.add_argument('--gpu', action = 'store_true')

args = parser_sec.parse_args()
args = vars(args)

arg_sec_inputs = []
for key, value in args.items():
    arg_sec_inputs.append(value)
   
if arg_sec_inputs[3] is None:
    arg_sec_inputs[3] = 'cat_to_name.json'
    
with open(arg_sec_inputs[3], 'r') as f:
    cat_to_name = json.load(f)
    
Network.model = utilities.load_checkpoint('saves/mycheckpoint.pth')

image = utilities.process_image(arg_sec_inputs[0])
    
utilities.predict_image(image, Network.model, True, cat_to_name, arg_sec_inputs[2])






