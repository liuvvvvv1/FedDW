import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from keras.src.datasets import imdb
from keras.src.utils import pad_sequences

def Load_Datasets(name_dataset):
    if name_dataset=='emnist':
        #train_dataset_size=697932
        #test_dataset_size=116323
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomVerticalFlip(p=1)
        ])
        train_data = datasets.EMNIST(root="./datas/EMNIST", split="byclass", download=False, train=True,transform=trans)
        test_data = datasets.EMNIST(root="./datas/EMNIST", split="byclass", download=False, train=False,transform=trans)
        return train_data,test_data
        pass
    elif name_dataset=='mnist':
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomVerticalFlip(p=1)
        ])
        train_data = datasets.MNIST(root='./datas/MNIST', train=True, download=True,transform=trans)
        test_data = datasets.MNIST(root='./datas/MNIST', train=False, download=True,transform=trans)
        return train_data,test_data
        pass
    elif name_dataset=="fashion_mnist":
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.RandomVerticalFlip(p=1)
        ])
        train_data = datasets.FashionMNIST(root='./datas/FashionMNIST', train=True, download=True,transform=trans)
        test_data = datasets.FashionMNIST(root='./datas/FashionMNIST', train=False, download=True,transform=trans)
        return train_data,test_data
    elif name_dataset=='cifar10':
        trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_data = torchvision.datasets.CIFAR10(root='./datas/CIFAR10', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.CIFAR10(root='./datas/CIFAR10', train=False, download=True, transform=trans)
        return train_data,test_data
        pass
    elif name_dataset=='cifar100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_data = torchvision.datasets.CIFAR100(root='./datas/CIFAR100', train=True, download=True, transform=trans)
        test_data = torchvision.datasets.CIFAR100(root='./datas/CIFAR100', train=False, download=True, transform=trans)
        return train_data, test_data
        pass
    elif name_dataset=='IMDB':
        MAX_WORDS = 10000
        MAX_LEN = 200
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
        x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
        x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
        print(x_train.shape, x_test.shape)
        train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
        test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
        return y_train,train_data, test_data
        pass
    else:

        pass
    pass











