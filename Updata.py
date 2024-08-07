import copy
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from utils import Dataset_By_Client
from utils import Draw_heatmap


class Local_Update_by_Fedavg():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.model=model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self):
        model = self.model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate, momentum=0.9,weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                # print("data ",data.shape)
                # print("label ",label.shape)

                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                # outputs=outputs.reshape(-1,self.config.n_class)
                # print("outputs ",outputs.shape)
                # print("label ",label.shape)
                loss = criterion(outputs, label)

                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ", end_time - start_time)
            pass
        return model.state_dict()
        pass
    pass


class Local_Update_by_moon():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.model=model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,client):
        cos = torch.nn.CosineSimilarity(dim=-1)
        model = self.model.to(self.config.device)

        global_model=copy.deepcopy(model)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False
            pass
        global_model = global_model.to(self.config.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()


                ch,h,outputs = model(data)
                g_ch,g_h,g_outputs=global_model(data)
                loss2=0.0
                if client.pre_nets.qsize()!=0:
                    posi = cos(ch,g_ch)            #  posi = torch.Size([256])
                    logits = posi.reshape(-1, 1)   # logits = torch.Size([256, 1])

                    for pre_net in client.pre_nets.queue:
                        pre_net.to(self.config.device)
                        pre_ch, pre_h, pre_outputs = pre_net(data)
                        nega = cos(ch,pre_ch)
                        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                        pre_net.to('cpu')
                        pass
                    logits = logits / self.config.temperature
                    label_c = torch.zeros(data.size(0)).to(self.config.device).long()
                    loss2 = self.config.moon_coefficient * criterion(logits, label_c)
                    pass
                loss1 = criterion(outputs, label)
                loss=loss1+loss2
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass

        model_c=copy.deepcopy(model)
        model_c.eval()
        for param in model_c.parameters():
            param.requires_grad = False
            pass
        if client.pre_nets.qsize()==self.config.max_pre_net_num:
            client.pre_nets.get()
            client.pre_nets.put(model_c)
            pass
        else:
            client.pre_nets.put(model_c)
            pass
        return model.state_dict()
        pass
    pass


class Local_Update_by_FedUV():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.model=model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        tester = torch.eye(config.n_class)
        self.batch_gamma = tester.std(dim=0).mean().item()
        pass

    def UVReg(self,outputs,label,ch):
        if len(outputs.shape) > 2:
            outputs = torch.reshape(outputs, (outputs.shape[0], np.prod(outputs.shape[1:])))
            label = torch.reshape(label, (label.shape[0], np.prod(label.shape[1:])))
            pass


        pdist_x = torch.pdist(ch, p=2).pow(2)
        sigma_unif_x = torch.median(pdist_x[pdist_x != 0])

        unif_loss = pdist_x.mul(-1/sigma_unif_x).exp().mean()

        logsoft_out = torch.softmax(outputs, dim=1)
        logsoft_out_std = logsoft_out.std(dim=0)
        std_loss = torch.mean(F.relu(self.batch_gamma - logsoft_out_std))

        loss = (self.config.std_coeff * std_loss+ self.config.unif_coeff * unif_loss)

        return loss, np.array([0, np.round(std_loss.item(), 5),0, 0, np.round(unif_loss.item(), 10)])
        pass

    def train(self):
        model = self.model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate, momentum=0.9,weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)
                loss = criterion(outputs, label)

                label = torch.nn.functional.one_hot(label, num_classes=self.config.n_class)
                label = label.float()

                loss_reg, loss_logs = self.UVReg(outputs, label, ch)
                loss+=loss_reg
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass




class Local_Update_by_FedDistill():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset = dataset
        self.idx_client = idx_client
        self.config = config
        self.model = model
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size,shuffle=True, num_workers=4)
        pass
    def train(self,distills,server_epoch):

        model = self.model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        criterion_KL = torch.nn.KLDivLoss(reduction='batchmean').to(self.config.device)
        mse_loss = torch.nn.MSELoss().to(self.config.device)

        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch,h,outputs = model(data)
                loss = criterion(outputs, label)


                p=torch.softmax(outputs,dim=1)
                soft_label=torch.zeros_like(p)
                for i in range(len(soft_label)):
                    soft_label[i]=distills[label[i]]
                    pass
                loss_kl=criterion_KL(p,soft_label)


                loss=loss+loss_kl
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass

        model.eval()
        loacl_distill =torch.zeros(self.config.n_class,self.config.n_class).to(self.config.device)
        counts = {}
        for k in range(self.config.n_class):
            counts[k]=0
            pass
        with torch.no_grad():
            for data, label in self.trainloader:
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch, h, outputs = model(data)

                di=torch.softmax(outputs,dim=1)

                for i, label in enumerate(label):
                    loacl_distill[label.item()]+=di[i]
                    counts[label.item()] += 1
                    pass
                pass
            pass
        return model.state_dict(),loacl_distill, counts
        pass


    pass



class Local_Update_by_Fedprox():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.model=model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self):
        model = self.model.to(self.config.device)

        global_model=copy.deepcopy(model)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False
            pass
        global_model = global_model.to(self.config.device)
        global_weight_list = list(global_model.parameters())

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()

            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss1 = 0.0
                for param_index, param in enumerate(model.parameters()):
                    loss1+= ((self.config.Fedprox_coefficient/2.0)*torch.norm((param - global_weight_list[param_index])) ** 2)
                    pass
                loss=loss+loss1
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass


class Local_Update_by_Fedproto():
    def __init__(self,dataset,idx_client,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,client,prototypes,server_epoch):

        model = client.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch,h,outputs = model(data)
                loss = criterion(outputs, label)

                loss1 = 0.0
                if server_epoch!=1:
                    for i in range(len(h)):
                        k = label[i].item()
                        prototype = prototypes[k].to(self.config.device)
                        loss1 += torch.nn.MSELoss()(h[i], prototype)
                    loss1 /= len(h)
                    loss=loss+loss1
                    pass
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        model.eval()
        loacl_prototypes = {}
        counts = {}
        with torch.no_grad():
            for data, label in self.trainloader:
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch, h, outputs = model(data)
                for i, label in enumerate(label):
                    if label.item() not in loacl_prototypes:
                        loacl_prototypes[label.item()] = []
                        counts[label.item()] = 0
                        pass
                    loacl_prototypes[label.item()].append(h[i].detach().cpu().numpy())
                    counts[label.item()] += 1
                    pass
                pass
            pass
        for k in loacl_prototypes.keys():
            loacl_prototypes[k] = torch.tensor(np.array(loacl_prototypes[k])).mean(0)
        return loacl_prototypes, counts
        pass
    pass




class Local_Update_by_FedBABU():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.model=model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self):
        model = self.model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()

            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass




class Local_Update_by_Fedper():
    def __init__(self,dataset,idx_client,avg_w,config):
        self.dataset=dataset
        self.idx_client=idx_client
        self.avg_w=avg_w
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,client):
        c_avg_w=client.model.state_dict()

        for k in c_avg_w.keys():
            if k!="fc3.bias" and k!="fc3.weight":
                c_avg_w[k]=self.avg_w[k]
                pass
            pass
        client.model.load_state_dict(c_avg_w)
        model = client.model

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass



class Local_Update_by_Fedrep():
    def __init__(self,dataset,idx_client,avg_w,config):
        self.dataset=dataset
        self.idx_client=idx_client
        self.avg_w=avg_w
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,client):
        c_avg_w=client.model.state_dict()

        for k in c_avg_w.keys():
            if k!="fc3.bias" and k!="fc3.weight":
                c_avg_w[k]=self.avg_w[k]
                pass
            pass
        client.model.load_state_dict(c_avg_w)
        model = client.model

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)

        for name, param in model.named_parameters():
            if name=="fc3.weight":
                param.requires_grad=True
                pass
            elif name=="fc3.bias":
                param.requires_grad=True
                pass
            else:
                param.requires_grad=False
                pass
            pass
        for epoch in range(self.config.head_ep):
            # start_time = time.time()

            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass


        for name, param in model.named_parameters():
            if name=="fc3.weight":
                param.requires_grad=False
                pass
            elif name=="fc3.bias":
                param.requires_grad=False
                pass
            else:
                param.requires_grad=True
                pass
            pass
        for epoch in range(self.config.head_ep,self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass





class Local_Update_by_Per_Fedavg():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        self.idx_client=idx_client
        self.model=model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass

    def compute_grad(self,x,y,model,criterion):
        ch, h, outputs = model(x)
        loss = criterion(outputs, y)
        grads = torch.autograd.grad(loss, model.parameters())
        return grads
        pass

    def train(self):
        model = self.model.to(self.config.device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)


        for epoch in range(self.config.client_epoch):
            # start_time = time.time()

            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)

                temp_model = copy.deepcopy(model)
                grads = self.compute_grad(data,label,temp_model,criterion)

                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.config.alpha * grad)
                    pass

                grads = self.compute_grad(data,label,temp_model,criterion)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.config.beta * grad)
                    pass

                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Local_Update_by_FedDW():
    def __init__(self,dataset,idx_client,model,config):
        self.dataset=dataset
        self.idx_client=idx_client
        self.config=config
        self.model=model
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,distills,server_epoch):

        model = self.model.to(self.config.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        mse_loss = torch.nn.MSELoss().to(self.config.device)

        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch,h,outputs = model(data)
                loss = criterion(outputs, label)

                # print("label")
                # print("type ",type(label))
                # print(label)
                # print("----------")
                # print(type(label[0]))
                # print(label[0])

                loss1 = 0.0
                if server_epoch!=-1:
                    mat_w=torch.mm(model.fc3.weight,model.fc3.weight.T)
                    # mat_w=F.relu(mat_w)
                    mat_w=torch.softmax(mat_w,dim=1)
                    loss1 = mse_loss(distills,mat_w)
                    loss=loss+loss1*self.config.FedDW_coefficient
                    #loss=loss
                    pass

                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass

        # print("----mat_w------")
        # mat_w = torch.mm(model.fc3.weight, model.fc3.weight.T)
        # mat_w = F.relu(mat_w)
        # mat_w = torch.softmax(mat_w, dim=1)
        # print("------distills------")
        # Draw_heatmap(distills,mat_w)



        model.eval()
        loacl_distill =torch.zeros(self.config.n_class,self.config.n_class).to(self.config.device)
        counts = {}
        for k in range(self.config.n_class):
            counts[k]=0
            pass
        with torch.no_grad():
            for data, label in self.trainloader:
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch, h, outputs = model(data)

                di=torch.softmax(outputs,dim=1)

                for i, label in enumerate(label):
                    loacl_distill[label.item()]+=di[i]
                    counts[label.item()] += 1
                    pass
                pass
            pass
        return model.state_dict(),loacl_distill, counts
        pass
    pass



class Local_Update_by_FedDW_pFL():
    def __init__(self,dataset,idx_client,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,client,distills,server_epoch):

        model = client.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        mse_loss = torch.nn.MSELoss().to(self.config.device)

        for epoch in range(self.config.client_epoch):
            # start_time = time.time()

            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch,h,outputs = model(data)
                loss = criterion(outputs, label)

                loss1 = 0.0
                if server_epoch!=-1:
                    mat_w=torch.mm(model.fc3.weight,model.fc3.weight.T)
                    mat_w=F.relu(mat_w)
                    mat_w=torch.softmax(mat_w,dim=1)

                    # print("mat_w" ,mat_w.requires_grad)
                    # print(mat_w.shape)
                    # print(distills.shape)
                    # print("a-b  :   ",distills-mat_w)
                    # print("dist_gn : ",distills.requires_grad)
                    loss1 = mse_loss(distills,mat_w)

                    loss=loss+loss1*self.config.FedDW_coefficient
                    pass
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        model.eval()
        loacl_distill =torch.zeros(self.config.n_class,self.config.n_class).to(self.config.device)
        counts = {}
        for k in range(self.config.n_class):
            counts[k]=0
            pass

        with torch.no_grad():
            for data, label in self.trainloader:
                data = data.to(self.config.device)
                label = label.to(self.config.device).long()
                ch, h, outputs = model(data)

                di=torch.softmax(outputs,dim=1)

                for i, label in enumerate(label):
                    loacl_distill[label.item()]+=di[i]
                    counts[label.item()] += 1
                    pass
                pass
            pass
        return loacl_distill, counts
        pass
    pass










class Local_Update_by_FedALA():
    def __init__(self,dataset,idx_client,global_model,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.global_model=global_model
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        self.layer_idx=self.config.layer_idx
        self.trainloader_w = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        self.eta=1.0
        self.rand_percent=80
        pass

    def adaptive_local_aggregation(self,global_model,client) :

        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)

        params_g = list(global_model.parameters())
        params = list(client.model.parameters())
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        model_t = copy.deepcopy(client.model)
        params_t = list(model_t.parameters())

        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False
            pass

        optimizer = torch.optim.SGD(params_tp, lr=0)

        if client.weights == None:
            client.weights = [torch.ones_like(param.data).to(self.config.device) for param in params_p]

        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,client.weights):
            param_t.data = param + (param_g - param) * weight

        for x, y in self.trainloader_w:
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            optimizer.zero_grad()
            ch, h, outputs = model_t(x)
            loss_value =criterion(outputs, y)
            loss_value.backward()

            for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, client.weights):
                weight.data = torch.clamp(weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)
                pass

            for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, client.weights):
                param_t.data = param + (param_g - param) * weight
                pass
            pass

        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
            pass
        pass
    def train(self,client):
        # start_time = time.time()
        self.adaptive_local_aggregation(self.global_model, client)
        # end_time = time.time()
        # print("run time : ", end_time - start_time)

        model=client.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        return model.state_dict()
        pass
    pass



class Local_Update_by_local_only():
    def __init__(self,dataset,idx_client,config):
        self.dataset=dataset
        # self.idx_client=idx_client
        self.config=config
        self.trainloader = DataLoader(Dataset_By_Client(dataset, idx_client), batch_size=config.batch_size, shuffle=True,num_workers=4)
        pass
    def train(self,client):
        model = client.model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.config.device)
        for epoch in range(self.config.client_epoch):
            # start_time = time.time()
            model.train()
            for data, label in self.trainloader:
                model.zero_grad()
                data = data.to(self.config.device)
                label = label.to(self.config.device)
                ch,h,outputs = model(data)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                pass
            # end_time = time.time()
            # print("run time : ",end_time-start_time)
            pass
        pass
    pass
