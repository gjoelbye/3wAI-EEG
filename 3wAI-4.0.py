import numpy as np
import torch, mne, warnings, glob
from datetime import datetime

# Uses pytorchs nn.Module class
class Model(torch.nn.Module):
    # Defines all the layers of the model
    def __init__(self, num_channels, num_classes, sample_length):
        super(Model, self).__init__()
        self.sample_length = sample_length
    
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (1, num_channels), padding = 0),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(16, False),
            torch.nn.Dropout(0.5)
            )
        
        self.block2 = torch.nn.Sequential(
            torch.nn.ZeroPad2d((16, 17, 0, 1)), 
            torch.nn.Conv2d(1, 4, (2, 32)),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(4, False),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d((2, 4))
            )
        
        self.block3 = torch.nn.Sequential(
            torch.nn.ZeroPad2d((2, 1, 4, 3)),
            torch.nn.Conv2d(4, 4, (8, 4)),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(4, False),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d((2, 4))
            )
        
        # Dense/fully connected layer
        self.fc = torch.nn.Linear(4*2*8*sample_length, num_classes)
        
    def forward(self, x):
        x = self.block1(x).permute(0, 3, 1, 2)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(-1, 4*2*8*self.sample_length)
        x = self.fc(x)
        return x

# EEG signal with 14 channels and recorded at 128Hz
# (Needs to be optimized for larger amounts of data) (upcomming)
def dataload(subjects, sample_length, seed=None, hz=128):
    x_data = []
    y_data = []
    
    # Loop through every person (subject)
    for index, subject in enumerate(subjects):
        
        # Loop through every EEG Data file for every persons
        for filename in subject:
        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw = mne.io.read_raw_edf(filename, preload=True, verbose=False) 
                
            # Apply a simple filter to remove the most basic noise
            raw.filter(1, 50., fir_design='firwin')
            
            
            # Assert if the data uses the wrong channels or are recorded at the wrong frequency.
            assert int(raw.info["sfreq"]) == hz, "EEG data is not recorded at 128Hz. Filename: {}".format(filename)
            assert raw.ch_names == ['TP10', 'Fz', 'P3', 'Cz', 'C4', 'TP9', 'Pz', 'P4', 'FT7', 'C3', 'O1', 'FT8', 'Fpz', 'O2'], "EEG data have different channels. Filename: {}".format(filename)
            
            data = raw.get_data().T
            
            # Splits the data in to samples of the correct length and format
            for i in range(hz*sample_length, len(data), hz*sample_length):
                x_data.append(np.array([data[i - hz*sample_length:i , :]]))
                y_data.append(index)
            
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Shuffle the x-data and y-data the same way
    np.random.seed(seed)    
    rnd_state = np.random.get_state()
    
    np.random.shuffle(x_data)
    np.random.set_state(rnd_state)
    np.random.shuffle(y_data)
    
    assert len(x_data) == len(y_data), "X-data and Y-data does not match."
    
    return x_data, y_data

# Calculates accuracy according to the higest value in output vector
# in other words, it does not take into account "how sure" the model is
# but only of it is correct.
def accuracy(model, device, batch_size, x_data, y_data):
    if mode: model.eval()
    
    corrects = 0
    with torch.no_grad():
        for i in range((len(x_data) // batch_size)):
            start = i * batch_size
            end = start + batch_size
            
            x = torch.tensor(x_data[start:end], dtype=torch.float32, device=device)
            y = torch.tensor(y_data[start:end], dtype=torch.long, device=device)
                    
            output = model(x)
            args = torch.max(output.data, dim=1)[1]
            
            corrects += torch.sum(args == y).item()
            
        if end < (len(x_data) - 1):
            x = torch.tensor(x_data[end:], dtype=torch.float32, device=device)
            y = torch.tensor(y_data[end:], dtype=torch.long, device=device)
                    
            output = model(x)
            args = torch.max(output.data, dim=1)[1]
            
            corrects += torch.sum(args == y).item()             
    
    return corrects/len(x_data)


def train(model, device, batch_size, x_train, y_train):
    if mode: model.train()
    
    running_loss = 0
    
    # Split the training up in batches.
    for i in range((len(x_train) // batch_size)):
        start = i * batch_size
        end = start + batch_size
        
        # Converts the samples to tensors in the GPU
        x = torch.tensor(x_train[start:end], dtype=torch.float32, requires_grad=True, device=device)
        y = torch.tensor(y_train[start:end], dtype=torch.long, device=device)
        
        # Sends the x data through the model
        output = model(x)
        # Calculates lost
        loss = criterion(output, y)
        
        # Optimizes the model using backpropagation and the optimizer algorithm 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Trains on the left over training data if there is any
    if end < (len(x_train) - 1):
        x = torch.tensor(x_train[end:], dtype=torch.float32, requires_grad=True, device=device)
        y = torch.tensor(y_train[end:], dtype=torch.long, device=device)
        
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss

#-----------------------------------------------------------------------------

hz = 128 # Datapoints per second per channel
channels = 14 # Number of channels
sample_length = 30 # Number of seconds per sample


"""
If the data are in a folder named "data" which contains
a folder for each person where their EEG-data is in,
then this should automaticly load the data.
"""

subjects = []
for path in [directory for directory in glob.glob("data/**/")]:
    files = [f for f in glob.glob(path + "/*")]
    subjects.append(files)

inputs, labels = dataload(subjects, sample_length, hz = hz)

# Assert if the data doesn't match up for some reasons.
assert len(inputs) == len(labels), "X-data and Y-data does not match."

split_index = int(len(inputs) * 0.85) # 85% training data and 15% test data

x_train = inputs[0:split_index]
y_train = labels[0:split_index]

x_test = inputs[split_index:]
y_test = labels[split_index:]
    
    
epochs = 50 # Number of epochs
batch_size = 1 # Batch size (significantly better accuracy for lower values)
learning_rate = 0.001 # Default: 0.001
mode = False # Enables or disables the switching between train and eval mode for model

#------------------------------------------------------------------------------

torch.cuda.empty_cache() # Empties the GPU catch
device = torch.device("cuda", 0) # Sets the device to be a CUDA-enabled GPU

# Creates the model
model = Model(channels, len(subjects), sample_length).to(device)

print("Number of classes: {} | Number of epochs: {}".format(len(subjects), epochs))
print("Batch size: {} | Learning rate: {}".format(batch_size, learning_rate))
print("Training data size: {} | Test data size: {}\n".format(len(x_train), len(x_test)))

# Tests the model on a random sample
test_sample = torch.tensor(np.random.rand(1, 1, hz*sample_length, channels), dtype=torch.float32, requires_grad=True, device=device)
test = model.forward(test_sample)

# Defines the loss function and optimizer algorithm

criterion = torch.nn.CrossEntropyLoss() # Loss function (build-in softmax)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # Optimizer algorithm

loss_list = []
test_list = []
train_list = []

test_acc = accuracy(model, device, batch_size, x_test, y_test)*100
train_acc = accuracy(model, device, batch_size, x_train, y_train)*100

print("Epoch: 000 | Test accuracy: {0:0.3f}% | Train accuracy: {1:0.3f}%".format(test_acc, train_acc))

for epoch in range(epochs):
    loss = train(model, device, batch_size, x_train, y_train)
    
    test_acc = accuracy(model, device, batch_size, x_test, y_test)*100
    train_acc = accuracy(model, device, batch_size, x_train, y_train)*100
    
    print("Epoch: {0:03d} | Test accuracy: {1:0.3f}% | Train accuracy: {2:0.3f}% | Loss: {3:0.3f}".format(epoch+1, test_acc, train_acc, loss))
    
    loss_list.append(loss)
    test_list.append(test_acc)
    train_list.append(train_acc)











