import torch

import argparse

from torch import nn, optim

from torchvision import datasets, models, transforms

from collections import OrderedDict
import matplotlib.pyplot as plt

import torch.nn.functional as F




# %matplotlib inline



# Setting up command line arguments



parser = argparse.ArgumentParser()



parser.add_argument('data_directory', action='store',default='flowers',

                    help='Store a Data Directory location')



parser.add_argument('--save_dir', action='store',

                    dest='save_dir',

                    help='Specify a directory to save checkpoints')



parser.add_argument('--arch', action='store',

                    default='vgg',

                    dest='arch',

                    help='Specify architecture. Select from vgg(default) or densenet.')



parser.add_argument('--learning_rate', action='store',

                    default=0.001,

                    dest='learning_rate',

                    type=float,

                    help='Specify learning rate. Default is 0.001.')



parser.add_argument('--hidden_units', action='store',

                    default=512,

                    dest='hidden_units',

                    type=int,

                    help='Specify hidden units. Default is 512.')



parser.add_argument('--epochs', action='store',

                    default=3,

                    dest='epochs',

                    type=int,

                    help='Specify epochs. Default is 3.')



parser.add_argument('--gpu', action='store_true',

                    default=False,

                    dest='gpu',

                    help='Set mode to GPU. Default is FALSE.')



parser.add_argument('--version', action='version',

                    version='%(prog)s 1.0')



arguments = parser.parse_args()



# Checking if GPU processing has been selected

device = torch.device("cuda:0" if arguments.gpu else "cpu")



# ## Load the data

data_dir = arguments.data_directory

train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir = data_dir + '/test'



# Define your transforms for the training, validation, and testing sets
data_transforms = {'testing': transforms.Compose([transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])]),
                   'validation': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])]),
                   'training': transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
                  }

image_datasets = {'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
                  'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
                  'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
                 }

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
               'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64),
               'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64)
              }

class_to_idx = image_datasets['training'].class_to_idx









# # Building and training the classifier



# Build and train your network. Consider adding options to select an algorithm

def build_model(modelName, hidden_units):

    if modelName == 'vgg':

        model = models.vgg11(pretrained=True) #instantiate the model



        for param in model.parameters():

            param.requires_grad = False

        #Replace the classifier

        classifier_fl = nn.Sequential(OrderedDict([

            ('cfl1', nn.Linear(25088,hidden_units)),

            ('relu1', nn.ReLU()),

            ('drp1', nn.Dropout(0.2)),

            ('cfl3', nn.Linear(hidden_units, 102)),

            ('out', nn.LogSoftmax(dim=1))

            ]))



        model.classifier = classifier_fl

        arch = 'vgg'



    elif modelName == 'densenet':

        model = models.densenet201(pretrained=True) #instantiate the model



        for param in model.parameters():

            param.requires_grad = False

        #Replace the classifier

        classifier_fl = nn.Sequential(OrderedDict([

            ('cfl1', nn.Linear(1920,hidden_units)),

            ('relu1', nn.ReLU()),

            ('drp1', nn.Dropout(0.3)),

            ('cfl3', nn.Linear(hidden_units, 102)),

            ('out', nn.LogSoftmax(dim=1))

            ]))



        model.classifier = classifier_fl

        arch = 'densenet'



    else:

        model = models.vgg11(pretrained=True)



        for param in model.parameters():

            param.requires_grad = False

        #Replace the classifier

        classifier_fl = nn.Sequential(OrderedDict([

            ('cfl1', nn.Linear(25088,hidden_units)),

            ('relu1', nn.ReLU()),

            ('drp1', nn.Dropout(0.3)),

            ('cfl3', nn.Linear(hidden_units, 102)),

            ('out', nn.LogSoftmax(dim=1))

            ]))



        model.classifier = classifier_fl

        arch = 'vgg'

        print("Unknown architecture, continuing with default settings")


    return model, arch



model, arch = build_model(arguments.arch ,arguments.hidden_units)



# Define Loss function and optimizer

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)



#function to validate while training



def validate(model, valloader, criterion):

    #setting validation loss and accuracy as zero

    val_loss = 0

    accuracy = 0

    #iterating through validation loader to feed forward and to track the error/accuracy

    for img, lbl in valloader:

        img, lbl = img.to(device), lbl.to(device)

        op = model.forward(img)

        val_loss += criterion(op, lbl).item()

        ps = torch.exp(op)

        eq = (lbl.data == ps.max(dim=1)[1])

        accuracy += eq.type(torch.FloatTensor).mean()



    return val_loss, accuracy



# Training the network. Convert this to a function

epochs = arguments.epochs

print_every = 40

steps = 0



#model to cuda

model.to(device)



#iterating for given number of epochs

epochs = 3
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in dataloaders['training']:
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['validation']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(dataloaders['validation']):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders['validation']):.3f}")
            running_loss = 0
            model.train()


print('End of training')




# ## Testing your network



# Validation on the test set

def chk_test_accuracy(testloader):

    optimizer.zero_grad()
test_loss1=0
accuracy1=0
model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['testing']:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss1 += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy1 += torch.mean(equals.type(torch.FloatTensor)).item()

        #print(f"Test loss: {test_loss1/len(dataloaders['testing']):.3f}.. "
         #     f"Test accuracy: {accuracy1/len(dataloaders['testing']):.3f}")

    print(f"Test accuracy: {accuracy1/len(dataloaders['testing']):.3f}")


# Calling the function

chk_test_accuracy(dataloaders["testing"])







# ## Save the checkpoint

model.class_to_idx = image_datasets['training'].class_to_idx
model.cpu()
torch.save({'epochs': arguments.epochs,
            'classifier': model.classifier, 
            'state_dict': model.state_dict(), 
            'arch': arch,
            'class_to_idx': model.class_to_idx},
            'classifier.pth')
