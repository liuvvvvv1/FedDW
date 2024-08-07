import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./datas/MNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./datas/MNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# Step 2: Define a simple neural network to extract features
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10,bias=False)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def extract_features(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = SimpleNN()

# Step 3: Train the neural network
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(6):  # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')

# Step 4: Extract features using the trained network




print("---------------------------------------------------------------------------------------------------------------")
w_features=[]
w_labels=[]

# print(net.fc3.weight.data)
print(type(net.fc3.weight.data))
wt=(net.fc3.weight.data)
for i in range(10):
    w_labels.append(i)
    # w_features.append(wt[i])
    # print("wt[i] ",wt[i])

    pass
w_features=wt
w_labels=torch.tensor(w_labels)
print(w_labels)
print(w_features.shape)

print("---------------------------------------------------------------------------------------------------------------")

features = []
labels = []
with torch.no_grad():
    for data in testloader:
        inputs, targets = data
        outputs = net.extract_features(inputs)

        # print("[output]  ",targets.shape)
        # print("[output] ",type(targets))

        features.append(outputs)
        labels.append(targets)
        pass
    features.append(w_features)
    # for i in features:
    #     print(i.shape)
    #     pass
    labels.append(w_labels)
    pass

features = torch.cat(features).cpu().numpy()
print("type ",type(features))
# print("ffff ",features[-20:])

s_labels = torch.cat(labels).cpu().numpy()

norms = np.linalg.norm(features, axis=1)
features = features / norms[:, np.newaxis]



# Step 5: Apply T-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
sum_features_2d = tsne.fit_transform(features)


features_2d=sum_features_2d[:-10]
w_features=sum_features_2d[-10:]

labels=s_labels[:-10]
w_labels=s_labels[-10:]
print("w_labels  ",w_labels)


# Step 6: Visualize the T-SNE results
plt.figure(figsize=(10, 8))
# scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.1)

plt.scatter(w_features[:,0],w_features[:,1],s=8000,c=w_labels,cmap='tab10',alpha=1.0,marker='x')
scatter = plt.scatter(w_features[:, 0], w_features[:, 1],s=70, c=w_labels, cmap='tab10', alpha=1.0)



print("w_f ",w_features)


plt.legend(*scatter.legend_elements())
plt.title("T-SNE Visualization",size=15)
plt.xlabel("T-SNE component 1",size=15)
plt.ylabel("T-SNE component 2",size=15)
plt.xticks([])
plt.yticks([])

# plt.rcParams['xtick.labelsize'] = 20
# plt.rcParams['ytick.labelsize'] = 14

# plt.savefig('t-sne.pdf')

plt.show()



