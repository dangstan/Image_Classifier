import torch
from torch import nn
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import numpy as np
from network import Network

def load_checkpoint(filepath): # retirei o modell 
                                  
    checkpoint = torch.load(filepath)
    Network.model, Network.criterion, Network.optimizer = Network.build_network(checkpoint['output_size'], checkpoint['dropouts'], checkpoint['hidden_layers'], checkpoint['lr'])
    
    
    Network.model.fc1 = nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])
    Network.model.drop1 = nn.Dropout(checkpoint['dropouts'])
    Network.model.fc2 = nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])
    Network.model.drop2 = nn.Dropout(checkpoint['dropouts'])
    Network.model.fc3 = nn.Linear(checkpoint['hidden_layers'][1], checkpoint['hidden_layers'][2])
    Network.model.drop3 = nn.Dropout(checkpoint['dropouts'])
    Network.model.fc4 = nn.Linear(checkpoint['hidden_layers'][2], checkpoint['output_size'])
    Network.model.classifier.load_state_dict(checkpoint['state_dict'])
    
    Network.model.arg = checkpoint['arch']
    Network.model.class_to_idx = checkpoint['class_labels']
    Network.criterion.load_state_dict(checkpoint['criterion_state'])
    Network.optimizer.load_state_dict(checkpoint['optimizer_state'])
    Network.optimizer.parameters = Network.model.classifier.parameters()
    
    return Network.model

def process_image(image):
    import numpy as np
    from PIL import Image
    image = Image.open(image)
    imageloader = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])  
    image = imageloader(image)
    image = np.array(image)
    return image
                                  
def predict_image(image_path, model, to_gpu, cat_to_name,  topk=5):
    if to_gpu is True and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.cuda.is_available() is False:
        return "gpu isn't available"
    Network.model = Network.model.to(Network.device)
    vd = False
    if topk is None:
        topk = 3
    else:
        vd = True
    image_path = image_path.transpose(1,2,0)
    imageloader = transforms.ToTensor()
    image_test = imageloader(image_path)
    image_test = torch.unsqueeze(image_test,0)
    image_test = image_test.to(device)
    logps = Network.model.forward(image_test)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim = 1)
    classes = []
    for unit in top_class[0]:
        for key, values in Network.model.class_to_idx.items():
            if unit == values:
                classes.append(key)
    top_p = top_p.cpu().detach().numpy()
    
    flower_names = []
    for unit in classes:
        for keys, values in cat_to_name.items():
            if unit == keys:
                flower_names.append(values)
    if vd:
        print('The top 3 most probable types of the flower displayed on the picture are: {0}, {1} and {2}.'.format(flower_names[0].title(), flower_names[1].title(), flower_names[2].title()))
 
    print("The most probable type of the flower displayed on the picture is the {}.".format(flower_names[0].title()))
    print("The resulting probability of it being a {0} is {1:.2f} %.".format(flower_names[0].title(), top_p[0][0]*100))