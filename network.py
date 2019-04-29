import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utilities

class Network:
    model = None
    arch = None
    optimizer = None
    criterion = None
    trainloader = None
    validloader = None 
    testloader = None
    train_datasets = None
    output = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def transforms_function(train_dir,test_dir, valid_dir):

        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder
        Network.train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
        test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
        valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        Network.trainloader = torch.utils.data.DataLoader(Network.train_datasets, batch_size=64, shuffle = True)
        Network.testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
        Network.validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
        return Network.trainloader, Network.testloader, Network.validloader, Network.train_datasets
       

    def build_network(output, dropouts = 0.2, hidden_layers = [768,512,256], lr = 0.001):

        if Network.arch == None:
            Network.arch = 'vgg16'
        if hidden_layers == None:
            hidden_layers = [768,512,256]
        if lr == None:
            lr = 0.001
        if Network.arch == 'vgg16':
            Network.model = models.vgg16(pretrained = True)
        elif Network.arch == 'vgg13':
            Network.model = models.vgg13(pretrained = True)    
        elif Network.arch == 'alexnet':
            Network.model = models.alexnet(pretrained = True)
        elif Network.arch == 'densenet161':
            Network.model = models.densenet161(pretrained = True)
        elif Network.arch =='inception_v3':
            Network.model = models.inception_v3(pretrained = True)
        else:
            print("The chosen model isn't supported. Try another Torchvison Model.")
            return

        for param in Network.model.parameters():
            param.requires_grad = False

        try:
            input_unit = Network.model.classifier.in_features
        except:
            try:
                input_unit = Network.model.classifier[0].in_features
            except:
                try:
                    input_unit = Network.model.classifier[1].in_features
                except:
                    try:
                        input_unit = Network.model.fc.in_features
                    except:
                        try:
                            input_unit = Network.model.fc[0].in_features
                        except:
                            try:
                                input_unit = Network.model.fc[1].in_features
                            except:
                                print('Model chosen not possible for this function. Try another Torchvision Model.')
                                return

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_unit, hidden_layers[0])),
                                                ('relu1', nn.ReLU()),
                                                ('drop1', nn.Dropout(dropouts)),
                                                ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                                                ('relu2', nn.ReLU()),
                                                ('drop2', nn.Dropout(dropouts)),
                                                ('fc3', nn.Linear(hidden_layers[1], hidden_layers[2])),
                                                ('relu3', nn.ReLU()),
                                                ('drop3', nn.Dropout(dropouts)),  
                                                ('fc4', nn.Linear(hidden_layers[2], output)),
                                                ('output', nn.LogSoftmax(dim=1))]))

        try:
            Network.model.classifier = classifier
        except:
            Network.model.fc = classifier


        Network.criterion = nn.NLLLoss()

        Network.optimizer = optim.Adam(Network.model.classifier.parameters(), lr = lr)

        return Network.model, Network.criterion, Network.optimizer


    def trainer(to_gpu, epochs = 5, print_each = 60):

        if to_gpu is True and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.cuda.is_available() is False:
            return "gpu isn't available"

        Network.model.to(Network.device)
        if epochs == None:
            epochs = 5
        steps = 0
        running_loss = 0

        for e in range(epochs):
            for images, labels in Network.trainloader:
                steps += 1
                images, labels = images.to(Network.device), labels.to(Network.device)

                Network.optimizer.zero_grad()

                logps = Network.model.forward(images)
                loss = Network.criterion(logps, labels)
                loss.backward()
                Network.optimizer.step()

                running_loss += loss.item()

                if steps % print_each == 0:
                    valid_loss = 0
                    accuracy = 0
                    Network.model.eval()
                    with torch.no_grad():
                        for images, labels in Network.validloader:
                            images, labels = images.to(Network.device), labels.to(Network.device)

                            logps = Network.model.forward(images)
                            batch_loss = Network.criterion(logps, labels)

                            valid_loss = batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim = 1)
                            equalizers = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equalizers.type(torch.FloatTensor)).item()

                    print(f"Epochs: {e+1}/{epochs}.. "
                          f"Train Loss: {running_loss/print_each:.3f}.. "
                          f"Validation Loss: {valid_loss/len(Network.validloader):.3f}.. "
                          f"Accuracy: {accuracy/len(Network.validloader):.3f}.. ")
                    running_loss = 0
                    Network.model.train()            

    def network_save(dropouts = 0.2, saving_file = 'saves/mycheckpoint.pth', epochs = 5, lr = 0.001, hidden_layers = [756,512,256]):

        Network.model.class_to_idx = Network.train_datasets.class_to_idx
        import os
        if not os.path.exists(saving_file):
            os.makedirs(saving_file)
        if saving_file == None:
            saving_file = 'saves/mycheckpoint.pth'
        else:
            saving_file = saving_file + '/mycheckpoint.pth'
        if lr == None:
            lr = 0.001
        if hidden_layers == None:
            hidden_layers = [768,512,256]
        if epochs == None:
            epochs = 5
        try:
            input_unit = Network.model.classifier.in_features
        except:
            try:
                input_unit = Network.model.classifier[0].in_features
            except:
                try:
                    input_unit = Network.model.classifier[1].in_features
                except:
                    try:
                        input_unit = Network.model.fc.in_features
                    except:
                        try:
                            input_unit = Network.model.fc[0].in_features
                        except:
                            try:
                                input_unit = Network.model.fc[1].in_features
                            except:
                                print("Input size wasn't found.")
                                return      

        checkpoint = {'input_size': input_unit,
                      'arch': Network.arch,
                      'output_size': Network.output,
                      'hidden_layers': hidden_layers,
                      'state_dict': Network.model.classifier.state_dict(),
                      'optimizer_state': Network.optimizer.state_dict(),
                      'criterion_state': Network.criterion.state_dict(),
                      'epochs': epochs,
                      'dropouts':dropouts,
                      'lr': lr,
                      'class_labels': Network.model.class_to_idx}

        torch.save(checkpoint, saving_file)

    def tester(model, testloader):

        modell = load_checkpoint('mycheckpoint.pth', Network.model)

        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in Network.testloader:

                logps = modell.forward(images)
                batch_loss = Network.criterion(logps, labels)
                test_loss = batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim = 1)
                equalizers = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equalizers.type(torch.FloatTensor)).item()

            print(f"Test Loss: {test_loss/len(Network.testloader):.3f}.. "
                  f"Accuracy: {accuracy/len(Network.testloader):.3f}.. ")