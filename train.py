import torch
import argparse
import utilities
from network import Network

parser = argparse.ArgumentParser()
parser.add_argument('a_data', action = 'store') # data folder
parser.add_argument('--save_dir', nargs = '?')
parser.add_argument('--arch', nargs = '?')
parser.add_argument('--learning_rate', nargs = '?', type = float)
parser.add_argument('--hidden_layers', nargs = '+', type = int)
parser.add_argument('--epochs', nargs = '?', type = int)
parser.add_argument('--gpu', action = 'store_true')

args = parser.parse_args()
args = vars(args)

arg_inputs = []
for key, value in args.items():
    arg_inputs.append(value)

Network.arch = arg_inputs[2]
Network.output = int(input('What is the output size? '))

def training(data_dir, arg_inputs, output):
   
    Network.model, Network.criterion, Network.optimizer = Network.build_network(Network.output, 0.2, arg_inputs[4], arg_inputs[3])
        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    Network.trainloader, Network.testloader, Network.validloader, Network.train_datasets = Network.transforms_function(train_dir,test_dir, valid_dir)
    Network.trainer(arg_inputs[6], arg_inputs[5], 60)     
    Network.network_save(0.2, arg_inputs[1], arg_inputs[5], arg_inputs[3], arg_inputs[4])

training(arg_inputs[0], arg_inputs, Network.output)
