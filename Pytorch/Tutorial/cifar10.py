# Image classifier using Cifar10 dataset and Pytorch framework
# By: Ricardo Vergel
# November 10/2020

########## Torch and utilities
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
########################
import matplotlib.pyplot as plt
import numpy as np

#### Useful functions
def imshow(img):
	img = img / 2 + 0.5 #Unnormalize the img
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()

#### Dataset
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
								
trainset = torchvision.datasets.CIFAR10(root = './data', train=True,
										download=True, transform=transform)
										
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
											shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root = './data', train=False,
										download=True, transform=transform)
										
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
											shuffle = False, num_workers=2)										

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')		
           
dataiter = iter(trainloader)									
images, labels = dataiter.next()

#show images

imshow(torchvision.utils.make_grid(images))

# Print labels

print(' '.join('%5s' % classes[labels[j]]for j in range(4)))

### Network architecture

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
		
net = Net()

### Loss function and optimizer

Loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		
		optimizer.zero_grad()
		
		# Forward
		outputs = net(inputs)
		
		#backward + optimize
		
		loss = Loss_fn(outputs,labels)
		loss.backward()
		optimizer.step()
		
		#print statistics
		
		running_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss:%.3f' %
					(epoch + 1, i+1, running_loss/2000))
			running_loss = 0.0
print('Training finished')

# Save the trained model

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

### Testing the network on the test dataset

dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)










