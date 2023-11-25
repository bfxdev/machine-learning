# Collection of results of assignments for the Coursera training "PyTorch Deep Learning in 7 Days"
# Best viewed in VSCode with interactive Jupyter notebook

#%%

import torch

print("PyTorch version: " + torch.__version__)


#%% Creation of tensors 1

empty = torch.empty(2,2)
print("Empty tensor:\n", empty)

# Random
rand = torch.rand(2,2)
print("Random tensor:\n", rand)

zeros = torch.zeros(2,2)
print("Zeros tensor:\n", zeros)

# Constants
full = torch.full((2,3,4), 10)
print("Full tensor:\n", full)

#%% Creation of tensors 2

print("Tensor from list of lists:\n", torch.tensor([[1,2], [3,4]]))

import numpy
n = numpy.linspace(0, 5, 5)
nn = torch.tensor(n)
print("Tensor from numpy linspace ", n, ":\n", nn)
print("Back to numpy: ", nn.numpy())

print("Shape: ", nn.shape)
print("Slice nn[1:] : ", nn[1:])
print("Slice nn[0] : ", nn[0])

print("With specific data type:\n", torch.ones(3, 3, dtype=torch.int64))

# %% Tensor operations

eye = torch.eye(3, 3)
print("Eye tensor (diagonal matrix):\n", eye)
print("Tensors addition eye+ones:\n", eye + torch.ones(eye.shape))
print("Multiplication by a scalar:\n", eye*3)
print("Element-wise multiplication eye*rand:\n", eye*torch.rand(eye.shape))
prod = eye@torch.rand(eye.shape)
print("Dot product eye @ rand:\n", prod)

print("Function argmax, i.e. gives the index of the maximum value: ", torch.argmax(prod))

# %%
import torch

# Using gradient functions to solve a basic equation x*x-4=0
# See 1-4-gradient.py
# See https://bytepawn.com/pytorch-basics-solving-the-axb-matrix-equation-with-gradient-descent.html

# Initial random value (between -5 and 5)
X = torch.autograd.Variable(10*(torch.rand(1)-0.5), requires_grad=True)
print("\n----------------------\nInitial X value:", X)

# Rate of adding reversed gradient
learning_rate = 0.01

# Gradient descent loop
for i in range(100):

  # The equation we want to solve, delta needs to be minimized
  delta = X*X - 4

  # The loss function is just a positive measure how far we are from the result
  loss = delta*delta # torch.norm(delta)

  # Computes the gradients i.e. the value of the derivatives of the loss function
  loss.backward()
  print("Current X =", X[0].data, " X.grad =", X.grad.data)
  print("--")

  # Adapts the current value of X
  # Positive derivative --> curve climbs i.e. we move away from zero, X needs to go back
  # Negative derivative --> curves descends i.e. good X direction to reach zero
  X.data -= learning_rate * X.grad.data

  # Value of gradients are added at each backward path, i.e. need to be zeroed
  # See https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
  X.grad.data.zero_()


# %%
# Assignment 1: load MNIST with torchvision, use DataLoader for batches, average pixel values per picture

import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Built-in data sets
print("Built-in datasets in torchvision:", dir(torchvision.datasets))
print("'MNIST' in torchvision datasets?", "MNIST" in dir(torchvision.datasets))

# Downloads MNIST dataset
mnist = torchvision.datasets.MNIST("C:\\temp", download=True)
print("Size of the MNIST data set:", len(mnist))
print("First element:", mnist[0])

# Displays N first images
N = 10
figure, axes = plt.subplots(1, N)
for i in range(N):
  img = mnist[i][0]
  num = mnist[i][1]
  axes[i].set_title(str(num))
  axes[i].imshow(img)
plt.show()

# Defines new dataset with on-the-fly transformation
mnist_as_tensors = torchvision.datasets.MNIST("C:\\temp",
                      transform=torchvision.transforms.ToTensor())

# Prepares DataLoader to read data in batches
# Each element will be a tuple ([list of images as tensors],[list of labels])
loader = DataLoader(mnist_as_tensors, batch_size=10)

# List comprehension to load data and compute average of each batch
batch_averages = torch.Tensor([batch[0].mean() for batch in loader])
print("Size of list of average values for each batch", len(batch_averages))
print(batch_averages)

# Final average:
average = batch_averages.mean()
print("Final average:", average)



#%%

# Assignment 2: read some new CSV data for regression or classification, modify parameters

import torch, pandas
import sklearn.metrics

# Reads input data and shows first row
# From: https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction/
# df = pandas.read_csv("ford.csv")
# print(df.iloc[0])
# Content of first row: model Fiesta, year 2017, price 12000, transmission Automatic, mileage 15944,
# fuelType Petrol, tax 150, mpg 57.7, engineSize 1.0

class DataSetCSV(torch.utils.data.Dataset):

  def encode(self, series_name, value):
    ''' Returns a tensor containing `value` encoded as scalar value or one-hot vector if
        `series_name` is a categorical series '''
    
    if series_name in self.categories:
      # One-hot encoding
      output = torch.zeros(len(self.categories[series_name]))
      output[self.categories[series_name].index(value)] = 1.0
      return output

    else:
      return torch.Tensor([value])


  def __init__(self, filename, scalar_names, category_names, output_name):
    ''' Reads the given CSV filename as Pandas dataframe, then initializes object to output a tensor
        with series given by `scalar_names` encoded as normal values and `category_names` as one-hot
        vectors. The `output_name` series is the target value to optimize in the training
        (`output_name` needs to be already mentioned in `scalar_names` or `category_names`).'''

    # Reads and keeps input data as Pandas DataFrame
    self.dataframe = pandas.read_csv(filename)

    # Stores constructor arguments
    self.scalar_names = scalar_names
    self.output_name = output_name

    # Prepares data for one-hot encoding, `self.categories` is a dict of lists of possible values
    self.categories = dict()
    for name in category_names:
      # Gets the unique values in this series, then convert form NumPy array to list
      self.categories[name] = self.dataframe[name].unique().tolist()

    # Computes sizes
    input,output = self[0]
    self.input_size = input.shape[0]
    self.output_size = output.shape[0]

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, index):
    ''' Returns an (input, output) tensor tuple with all categories one hot encoded '''

    # Index may be passed as a tensor
    if type(index) is torch.Tensor:
      index = index.item()

    # Retrieves row in dataframe
    sample = self.dataframe.iloc[index]

    # Encodes input in the order of columns found in input dataframe
    input_components = []
    for name, value in sample.items():
      if (name in self.categories.keys() or name in self.scalar_names) and name != self.output_name:
        input_components.append(self.encode(name, value))
    # This function is the only and fastest way to bring the components together in a tensor
    # print(input_components)
    input = torch.cat(input_components)

    # Encodes output
    output = self.encode(self.output_name, sample[self.output_name])

    return (input, output)

class Model(torch.nn.Module):

  def __init__(self, input_size, output_size, depth=2, size=128):
    ''' Creates `depth` layers of `size` neurons, `input_size` values entering the NN and
        `output_size` values to optimize '''

    # Calls constructor of the parent class
    super().__init__()

    # Stores input arguments
    self.size = size
    self.depth = depth
    self.input_size = input_size
    self.output_size = output_size

    # Creates NN layers as members of the object
    self.layers = torch.nn.ModuleList()
    for i in range(depth):
      self.layers.add_module("Linear "+str(i), torch.nn.Linear(input_size if i==0 else size, size))
      self.layers.add_module("ReLU "+str(i), torch.nn.ReLU())
    self.layers.add_module("Last layer", torch.nn.Linear(size, output_size))

  def forward(self, inputs):

    # Computes inference through the layers
    buffer = inputs
    for layer in self.layers:
      buffer = layer(buffer)

    # Applies a softmax function for classification
    if self.output_size > 1:
      buffer = torch.nn.functional.softmax(buffer, dim=-1)

    return buffer

def training(training_set, model, optimizer, loss_function, iterations=16):

  # Defines DataLoader
  dataloader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)

  # Training loop
  for epoch in range(iterations):
    for inputs, outputs in dataloader:
      optimizer.zero_grad()
      results = model(inputs)
      loss = loss_function(results, outputs)
      loss.backward()
      optimizer.step()
    print("Loss: {0}".format(loss))

def testing(testing_set):
  dataloader = torch.utils.data.DataLoader(testing_set, batch_size=len(testing_set), shuffle=False)
  for inputs, outputs in dataloader:
    results = model(inputs).argmax(dim=1).numpy()
    actual = outputs.argmax(dim=1).numpy()
    accuracy = sklearn.metrics.accuracy_score(actual, results)
    print("Accuracy: {0:.2f}%".format(100*accuracy))


# -------------------- CLASSIFICATION --------------------

# Defines Data Set for Fuel Type classification
ford_classification = DataSetCSV("ford.csv", ['year','price','mileage','tax','mpg','engineSize'],
                                 ['model', 'transmission', 'fuelType'], 'transmission')

print("\nCategories found in CSV:", ford_classification.categories)
print("\nCLASSIFICATION")
print("Input size:", ford_classification.input_size)
print("Output size:", ford_classification.output_size)
#print("\n100th row in DataFrame\n", df.iloc[100])
#print("\n100th row in DataSet\n", ford_fuel_type[100])

# Splits training and testing sets
ntesting = int(len(ford_classification) * 0.05)
ntraining = len(ford_classification) - ntesting
train, test = torch.utils.data.random_split(ford_classification, [ntraining, ntesting])

# Trains model
model = Model(ford_classification.input_size, ford_classification.output_size, 10, 200)
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCELoss()
training(train, model, optimizer, loss_function, 15)
testing(test)

# -------------------- REGRESSION --------------------

ford_regression = DataSetCSV("ford.csv", ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize'],
                        ['model', 'transmission', 'fuelType'], 'price')
print("\nREGRESSION")
print("Input size:", ford_regression.input_size)
print("Output size:", ford_regression.output_size)
train,test = torch.utils.data.random_split(ford_regression, [ntraining, ntesting])
model = Model(ford_regression.input_size, ford_regression.output_size, 10, 200)
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.MSELoss()
training(train, model, optimizer, loss_function, 15)
testing(test)


#%%

# Assignment 3: convolutional neural network

import torch, torchvision, math, random
# import torchvision.transforms as transforms
# import torch.nn as nn
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

# Converts images to tensors
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# DataSet and DataLoader for training
# CIFAR10 consists of 60000 32x32 color images (3 channels) in 10 classes
train_set = torchvision.datasets.CIFAR10('C:\\temp', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
#print("First training element:", train_set[0], train_set[0][0].shape)

# DataSet and DataLoader for testing
test_set = torchvision.datasets.CIFAR10('C:\\temp', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

# Model to classify input pictures after convolution layers
class ConvolutionNetwork(torch.nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()

    self.convolution = torch.nn.Sequential(

      # Initial convolution layer, CIFAR10 inputs are 3x32x32 tensors, output 32x16x16
      torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Output 32x32x32
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      # Convolution layer, output 64x8x8
      torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output 64x16x16
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      # Convolution layer, no pooling, output 128x8x8
      torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=True),

      # Convolution layer, no pooling, output 256x8x8
      torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=True),

      # Convolution layer, output 128x4x4
      torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(kernel_size=2, stride=2))

    self.flatten = torch.nn.Sequential(

      # Flatten layer, output 2048 = 128*4*4
      # After this layer, the size of the input picture matters
      torch.nn.Flatten(start_dim=1))
    
    self.classification = torch.nn.Sequential(

      # Initial classification layer
      # If the input size is not correct from the flatten operation, this error appears:
      # RuntimeError: mat1 and mat2 shapes cannot be multiplied ( batch_size x flatten_size ...
      torch.nn.Dropout(),
      torch.nn.Linear(2048, 1024),
      torch.nn.ReLU(inplace=True),

      # Classification layer
      torch.nn.Dropout(),
      torch.nn.Linear(1024, 1024),
      torch.nn.ReLU(inplace=True),

      # Final classification layer
      torch.nn.Linear(1024, num_classes))

  def forward(self, x):
    x = self.convolution(x)
    x = self.flatten(x)
    x = self.classification(x)
    return x
  
  def evaluate(self, input, classes):
    """ Displays the result of the Conv2d filters on the given input picture and shows the result
        of the classification """

    # Displays input picture
    print("Input shape in display_filters:", input.shape)
    fig = plt.figure(figsize=(1,1), frameon=False)
    ax = fig.add_subplot(1,1,1)
    image = input.clone().to('cpu').numpy().swapaxes(0,2).swapaxes(0,1)
    ax.imshow(image, interpolation='bicubic')
    ax.axis("off")
    plt.show(fig)

    # Applies the layers of the Convolution part one-by-one
    output = input
    for layer in self.convolution:
      output = layer(output)

      # Displays results of convolution
      if type(layer) == torch.nn.Conv2d:
        print("Layer:", layer)
        print("Output shape:", output.shape)

        # Arranges pictures
        num = output.shape[0]
        columns = 10
        rows = math.ceil(num/columns)
        print(f"Number of filters: {num}  Columns: {columns}  Rows: {rows}")

        # Displays all filters in one figure
        fig = plt.figure(figsize=(columns,rows), frameon=False, dpi=80)
        for i in range(num):
          ax = fig.add_subplot(rows, columns, i+1)
          image = output[i].clone().to('cpu').detach().numpy() #.swapaxes(0,2).swapaxes(0,1)
          ax.imshow(image, interpolation='bicubic')
          ax.axis("off")
        plt.show(fig)

    # Shows the input picture again
    fig = plt.figure(figsize=(1,1), frameon=False)
    ax = fig.add_subplot(1,1,1)
    image = input.clone().to('cpu').numpy().swapaxes(0,2).swapaxes(0,1)
    ax.imshow(image, interpolation='bicubic')
    ax.axis("off")
    plt.show(fig)

    # Evaluates the classification, flattens all dimensions as we are not in a batch
    output = output.flatten(start_dim=0)
    output = self.classification(output)

    # Prints the results
    result = {name: score for name,score in zip(classes, output.to('cpu').tolist())}
    print("Classification result:", result)
    print("\n*** CLASS: ", classes[output.argmax().item()], "***")


# Instantiation of the network
net = ConvolutionNetwork(num_classes=10)
print(net)

# Reasonable settings for a Classification
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# Checks for GPU support
if torch.cuda.is_available():
  device = torch.device('cuda')
  print("GPU supported!")
else:
  device = torch.device('cpu')
net.to(device)

# Main training loop
loss = 1e3
while loss > 0.2:

  # Training Step: trains network on all inputs
  total_loss = 0
  for inputs, outputs in train_loader:
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    optimizer.zero_grad()
    results = net(inputs)
    loss = loss_function(results, outputs)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
  loss = total_loss / len(train_loader)

  # Displays current status during training with one element of the testing set
  net.evaluate(test_set[int(random.random()*len(test_set))][0].to(device), test_set.classes)

  print(f"Loss: {loss}")

# Displays the accuracy of the trained network
for inputs, actual in test_loader:
  inputs = inputs.to(device)
  results = net(inputs).argmax(dim=1).to('cpu').numpy()
  accuracy = sklearn.metrics.accuracy_score(actual, results)
  print(f"Accuracy: {accuracy}")
print(sklearn.metrics.classification_report(actual, results))



#%%

# Assignment 4: Transfer learning with ResNet

import numpy as np
import sklearn.metrics
import torch, torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Pre-trained model
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
print(model)
print("Final classifier:", model.fc)

# Resizes input images to 224x224 pixels
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(224,224)),
                                            torchvision.transforms.ToTensor()])

# DataSet and DataLoader for training
# CIFAR10 consists of 60000 32x32 color images (3 channels) in 10 classes
train_set = torchvision.datasets.CIFAR10('C:\\temp', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
#print("First training element:", train_set[0], train_set[0][0].shape)

# DataSet and DataLoader for testing
test_set = torchvision.datasets.CIFAR10('C:\\temp', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

# Displays first image of the training set
image = train_set[0][0].numpy().swapaxes(0,2).swapaxes(0,1)
fig = plt.figure(figsize=(3,3), frameon=False)
ax = fig.add_subplot(1,1,1)
ax.imshow(image, interpolation='bicubic')
ax.axis("off")
plt.show(fig)

# Replaces the final classifier with a simpler one for the 10 classes of CFAR10
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Checks if GPU is available
if torch.cuda.is_available():
  device = torch.device('cuda')
  print("GPU supported!")
else:
  device = torch.device('cpu')

# Parameters for training
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
model.to(device)
model.train()
for epoch in range(1):
  for inputs, outputs in tqdm(train_loader):
    inputs = inputs.to(device, non_blocking=True)
    outputs = outputs.to(device, non_blocking=True)
    optimizer.zero_grad()
    results = model(inputs)
    loss = loss_function(results, outputs)
    loss.backward()
    optimizer.step()
  print("Last loss: {0}".format(loss))

# Displays result
results_buffer = []
actual_buffer = []
with torch.no_grad():
    model.eval()
    for inputs, actual in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        results = model(inputs).argmax(dim=1).to('cpu').numpy()
        results_buffer.append(results)
        actual_buffer.append(actual)

print(sklearn.metrics.classification_report(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))


#%%

# Assignment 5: experiment with LSTM-based network and word embeddings

import numpy as np, pandas, torch, spacy, tqdm, sklearn.metrics

# from torch.utils.data import Dataset
# import torch.utils.data

# Large English spaCy pipeline from blogs/news/comments data from the web
# See https://spacy.io/models/en#en_core_web_lg
# Installed with: python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')

# Example 
for token in nlp('Hello world!'):
    print("Token:", token)
    # print("Token vector:", token.vector)
    print("Token vector shape:", token.vector.shape)

# Definition of the DataSet
class SentimentDataset(torch.utils.data.Dataset):
    
    # Just reads the input data, then groups by SentenceID and takes the first entry in the group,
    # which happens to be the longest sentence in the very redundant "sentiment.tsv"
    def __init__(self):
        self.data = pandas \
            .read_csv('sentiment.tsv', sep='\t', header=0) \
            .groupby('SentenceId') \
            .first()

    def __len__(self):
        return len(self.data)

    # Gets one element with on-the-fly transformation to Tensors
    def __getitem__(self, idx):
        if type(idx) is torch.Tensor:
            idx = idx.item()
        sample = self.data.iloc[idx]
        token_vectors = []

        # Builds a tensor with all the word vectors concatenated in a single long vector
        # --> The size of each returned item is variable reflects the number of words in sentence
        # Switching off NER for a tiny speed boost, see https://spacy.io/api/entityrecognizer
        for token in nlp(sample.Phrase.lower(), disable=['ner']):
            token_vectors.append(token.vector)

        # Tokens and length as inputs, so the length is needed to 'pack' variable length sequences
        # output is the sentiment score 
        return (torch.tensor(token_vectors),
                torch.tensor(len(token_vectors)),
                torch.tensor(sample.Sentiment))

# Example of data item from the dataset
sentiment = SentimentDataset()
print("\nFirst element of the Data Set:\n", sentiment[0])
print("Shape of the data tensor:", sentiment[0][0].shape)
print("\nSecond element of the Data Set:\n", sentiment[1])
print("Shape of the data tensor:", sentiment[1][0].shape)

# Need to collate into fixed width as these will be variable batches
# See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
# See https://discuss.pytorch.org/t/is-padding-and-packing-of-sequences-really-needed/66478
def collate(batch):
    # Sort in descending length order, this is needed for padding sequences in pytorch
    # This is a preparation for pack_padded_sequence in the forward function of the model
    batch.sort(key=lambda x: x[1], reverse=True)

    # Separates items from the dataset in lists with same type of information
    # * is the unpack operator, i.e. provides all elements of the batch as arguments to zip
    # zip here will group the 3 elements of the items of the 3-tuple from the dataset
    sequences, lengths, sentiments = zip(*batch)

    # The variable `sequences` now contains a list of B items, each is a Nx300 tensor, where N is
    # the original length of the sentence from the sentiment data file (N different for each item).
    # The function pad_sequence here will add 1x300 tensors of zeros to each list item, resulting
    # in a BxTx300 tensor, where T is the longest sequence among the values of N.
    # See https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    # Creates tensors back from the lists
    # See https://pytorch.org/docs/stable/generated/torch.stack.html
    sentiments = torch.stack(sentiments)
    lengths = torch.stack(lengths)
    return sequences, lengths, sentiments

# Break this into a training and testing dataset
number_for_testing = int(len(sentiment) * 0.05)
number_for_training = len(sentiment) - number_for_testing
train, test = torch.utils.data.random_split(sentiment, [number_for_training, number_for_testing])

# The collate function is used on-the-fly for padding
train_loader = torch.utils.data.DataLoader(
    train, batch_size=32, shuffle=True,
    collate_fn=collate)
testloader = torch.utils.data.DataLoader(
    test, batch_size=32, shuffle=True,
    collate_fn=collate)

# Take a peek at first element and see what we are collating
for b in train_loader: break
print("\nShapes of first 3 data items of first batch:", b[0][0].shape, b[0][1].shape, b[0][2].shape)
# what is the max length?
print("Longest sentence from padding function:", b[1][0])

class Model(torch.nn.Module):

    def __init__(self, input_dimensions, size=128, layers=1):
        super().__init__()
        self.seq = torch.nn.GRU(input_dimensions, size, layers)
        self.layer_one = torch.nn.Linear(size * layers, size)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size, size)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size, 5)

    def forward(self, inputs, lengths):
        
        # Packs the sequences of word vectors, necessary for all RNN modules for performance
        # See https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        # Need to sort the sequences for pytorch, which we did in our collation above
        number_of_batches = lengths.shape[0]
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths,
            batch_first=True)
        # buffer, (hidden, cell) = self.seq(packed_inputs)  # for LSTM
        buffer, hidden = self.seq(packed_inputs)            # for GRU
        # batch first...
        buffer = hidden.permute(1, 0, 2)
        # flatten out the last hidden state -- this will
        # be the tensor representing each batch
        buffer = buffer.contiguous().view(number_of_batches, -1)
        # and feed along to a simple output network with
        # a single output cell for regression
        buffer = self.layer_one(buffer)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        return buffer

# Get the input dimensions from the first sample encodings are word, vectors - so index 1 at the end
model = Model(sentiment[0][0].shape[1])

# Training loop
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(5):
    losses = []
    for sequences, lengths, sentiments in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        results = model(sequences, lengths)
        loss = loss_function(results, sentiments)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss: {0}".format(torch.tensor(losses).mean()))

# Displays statistics
results_buffer=[]
actual_buffer=[]
with torch.no_grad():
    model.eval()
    for test_seq, test_len, test_sentiment in testloader:
        results=model(test_seq, test_len).argmax(dim=1).numpy()
        results_buffer.append(results)
        actual_buffer.append(test_sentiment)

print(sklearn.metrics.classification_report(
    np.concatenate(actual_buffer),
    np.concatenate(results_buffer)))
