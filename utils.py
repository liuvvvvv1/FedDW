import copy
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from matplotlib import pyplot


def dirichlet_split_noniid(train_labels, alpha, n_clients):

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idx = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idx = [[] for _ in range(n_clients)]
    for k_idx, fracs in zip(class_idx, label_distribution):
        for i, idx in enumerate(np.split(k_idx, (np.cumsum(fracs)[:-1]*len(k_idx)).astype(int))):
            client_idx[i] += [idx]
            pass
        pass
    client_idx = [np.concatenate(idx) for idx in client_idx]
    return client_idx


def split_iid(train_labels,n_clients):
    idxs = np.random.permutation(len(train_labels))
    client_idx = np.array_split(idxs, n_clients)
    return client_idx
    pass



# def show_division_by_class(train_data,labels,client_idx,n_client,n_class):
#     plt.figure(figsize=(12, 8))
#     plt.hist([labels[idx]for idx in client_idx], stacked=True,bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
#                  label=["Client {}".format(i) for i in range(n_client)],rwidth=0.5)
#
#
#     print(np.arange(n_class))
#     print(train_data.classes)
#     plt.xticks(np.arange(n_class), train_data.classes)
#
#     plt.xlabel("Label type")
#     plt.ylabel("Number of samples")
#     plt.legend(loc="upper right")
#     plt.title("show division by class")
#     plt.show()
#     pass


def show_division_by_client(train_data,labels,client_idx,n_client,n_class,config):
    plt.figure(figsize=(12, 10))
    label_distribution = [[] for _ in range(n_class)]
    for c_id, idc in enumerate(client_idx):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)
            pass
        pass

    # print("[type]  ",type(train_data.classes))
    # print("[data]  ",train_data.classes)
    if config.name_dataset=='IMDB':
        classes=['0','1']
        pass
    else:
        classes=train_data.classes
        pass


    plt.hist(label_distribution, stacked=True,bins=np.arange(-0.5, n_client + 1.5, 1),label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_client), ["client %d" %c_id for c_id in range(n_client)])
    plt.xlabel("client_num")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("show division by client")

    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('d=01.pdf')
    plt.show()
    pass



def show_accuracy_curve(x,y):
    pyplot.plot(x,y)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Harbin Engineering University')
    pyplot.show()
    pass

def show_accury_list(acc,acc_list,ep_list):
    print('accuracy  :{} '.format(acc))
    print("acc = ", acc_list)
    print("epo = ", ep_list)
    pass


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Dataset_By_Client(Dataset):
    def __init__(self,dataset,idx_client):
        self.dataset=dataset
        self.idx_client=[int(i) for i in idx_client]
        pass
    def __len__(self):
        return len(self.idx_client)
        pass
    def __getitem__(self, item):
        img,label=self.dataset[self.idx_client[item]]
        #return torch.tensor(img),torch.tensor(label)
        return img,label
        pass
    pass

##---test--------------------------------------------------------------------------------------------------------------


class Particle:
    def __init__(self, dimension, num_points):
        # Initialize position and velocity
        self.positions = np.random.randn(num_points, dimension)
        self.positions /= np.linalg.norm(self.positions, axis=1)[:, np.newaxis]
        self.velocities = np.random.randn(num_points, dimension) * 0.1
        self.best_positions = np.copy(self.positions)
        self.best_score = float('-inf')

    def update_velocity(self, global_best_positions, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_velocity = cognitive_weight * r1 * (self.best_positions - self.positions)
        social_velocity = social_weight * r2 * (global_best_positions - self.positions)
        self.velocities = inertia_weight * self.velocities + cognitive_velocity + social_velocity

    def update_position(self):
        self.positions += self.velocities
        self.positions /= np.linalg.norm(self.positions, axis=1)[:, np.newaxis]

def pso_sphere(n, m, iterations=100, swarm_size=30):
    swarm = [Particle(n, m) for _ in range(swarm_size)]
    global_best_positions = np.random.randn(m, n)
    global_best_positions /= np.linalg.norm(global_best_positions, axis=1)[:, np.newaxis]
    global_best_score = float('-inf')

    def objective_function(points):
        dist_sum = 0
        num_points = len(points)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist_sum += np.linalg.norm(points[i] - points[j])
        return dist_sum

    for iteration in range(iterations):
        for particle in swarm:
            particle.update_velocity(global_best_positions)
            particle.update_position()

            score = objective_function(particle.positions)
            if score > particle.best_score:
                particle.best_score = score
                particle.best_positions = np.copy(particle.positions)

            if score > global_best_score:
                global_best_score = score
                global_best_positions = np.copy(particle.positions)

        #print(f"Iteration {iteration + 1}/{iterations}, Best Score: {global_best_score}")

    return global_best_positions


def generate_classifier_parameter(n,m):
    iterations = 50
    points = pso_sphere(n, m, iterations)
    # for point in points:
    #     print(point, end='')
    #     print("  ", end='')
    #     print(np.sqrt(point[0] * point[0] + point[1] * point[1]))
    #     pass
    return points
    pass




# t=torch.tensor(points)
# print('ttt = ',t)
# t.requires_grad=False
# # def init_fc(m):
# #     torch.nn.init.constant_(m.weight,t)
# #     pass
# print(net)
# print(net.fc1)
# print('TYPE = ',type(net.fc1.weight.data))
# net.fc1.weight.data=t
# print(net.fc1.weight.shape)
# print("in_features = ",net.fc1.in_features)
# print("out_features = ",net.fc1.out_features)
# print("w ",net.fc1.weight)
# print("[][][][][]")
# # net.fc1.apply(init_fc)
# print(net.fc1.weight)
def init_align(model):
    fc3_in=model.fc3.in_features
    fc3_out=model.fc3.out_features
    points=generate_classifier_parameter(fc3_in,fc3_out)
    #
    t3 = torch.tensor(points).float()
    #t3.requires_grad = False
    model.fc3.weight.data = t3
    model.fc3.bias.data=torch.zeros(fc3_out)
    #model.fc3.weight.requires_grad=False
    model.fc3.bias.requires_grad=False
    # model.fc2.weight.requires_grad=False
    # model.fc2.bias.requires_grad=False
    # model.fc1.weight.requires_grad=False
    # model.fc1.bias.requires_grad=False
    print(model.fc3.weight)
    print(model.fc3.bias)
    return model
    pass


def inference(model,test_loader,device):
    model.eval()
    correct=0.0
    total=0.0
    model = model.to(device)
    with torch.no_grad():
        for data,label in test_loader:
            data=data.to(device)
            label=label.to(device)

            ch,h,outputs=model(data)
            p=torch.max(outputs,dim=1)[1]

            correct+=(p==label).sum().item()
            total+=data.size(0)
            pass
        pass
    acc=100*correct/total
    return acc
    pass


def inference_by_pFL(list_client,test_loader,device):
    list_acc=[]
    sum_acc=0.0
    for inde in range(len(list_client)):
        model=list_client[inde].model
        model=model.eval()
        correct = 0.0
        total = 0.0
        model = model.to(device)
        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                ch, h, outputs = model(data)
                p = torch.max(outputs, dim=1)[1]

                correct += (p == label).sum().item()
                total += data.size(0)
                pass
            pass
        acc = 100 * correct / total
        sum_acc=sum_acc+acc
        list_acc.append(acc)
        pass
    avg_acc=sum_acc/len(list_client)
    return list_acc,avg_acc
    pass

def check_norm(list_client):
    print("==========norm==========")
    for inde in range(len(list_client)):
        print("[inde] ",inde)
        model=list_client[inde].model
        t= model.fc3.weight.data
        # print(model.fc3.bias.data)
        # print("shape = ",t.shape)
        # print("[] ",torch.norm(t,dim=1))

        tt=torch.norm(t,dim=1)

        print(tt / torch.sum(tt))
        pass

    pass




def Draw_heatmap(t,t2):
    data1 = t.detach().to('cpu').numpy()
    data2 = t2.detach().to('cpu').numpy()
    # 创建一个包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))

    # 绘制第一个热力图
    cax1 = ax1.imshow(data1, cmap='viridis', aspect='auto')
    # ax1.set_title('Heatmap 1', size=27)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # fig.colorbar(cax1, ax=ax1)

    # 绘制第二个热力图
    cax2 = ax2.imshow(data2, cmap='viridis', aspect='auto')
    # ax2.set_title('Heatmap 2', size=27)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    # fig.colorbar(cax2, ax=ax2)

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.2)  # 增加宽度方向的间距

    plt.savefig('niid.pdf')

    plt.show()
    pass
