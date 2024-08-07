import copy
from Initialization import initialization
from Updata import *
from config import Configs
import dataset
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
import party
from models import *
from utils import *
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#-----------------------------------------------------------------------------------------------------------------------------------------------------#




if __name__=='__main__':
    config = Configs()
    labels,train_data, test_data, global_model, client_idx = initialization(config)

    # print("[client_idx]", client_idx)
    for i in range(len(client_idx)):
        if len(client_idx[i])==0:
            client_idx[i]=np.append(client_idx[i],0)
            pass
        print("[client_idx]",i,"  ",len(client_idx[i]))
        pass

    # print("[client_idx]", client_idx)

    # show_division_by_class(train_data,np.array(train_data.targets), client_idx,config.n_client,config.n_class)

    show_division_by_client(train_data, labels, client_idx,config.n_client,config.n_class,config)

    print("------------------------------------Initialization completed------------------------------------")

    sum=len(train_data)
    print("sum = ",sum)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=4)

    list_client = []
    freq_client=[]
    server = party.Server(config)
    for inde, i in enumerate(range(config.n_client)):
        list_client.append(party.Client(config, client_idx[i]))
        # freq_client.append(len(client_idx[i])/sum)
        freq_client.append(len(client_idx[i]))
        pass

    # print("[]][][][][][]")
    # print(freq_client)
    # print("FREQ")
    acc_list = []
    ep_list = []
    if config.name_method == 'Fedavg':
        print("Fedavg")
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            global_model.train()
            sum_choose=0
            freq_client_choose=[]
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_Fedavg(train_data, list_client[inde].idx, copy.deepcopy(global_model), config)
                local_w=local_updata.train()

                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            #--------------------------------------------------------------------------------------------------
            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method == 'moon':
        print("moon")
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_moon(train_data, list_client[inde].idx, copy.deepcopy(global_model), config)
                local_w=local_updata.train(list_client[inde])
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            #--------------------------------------------------------------------------------------------------
            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method == 'FedUV':
        print("Feduv")
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            global_model.train()
            sum_choose=0
            freq_client_choose=[]
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_FedUV(train_data, list_client[inde].idx, copy.deepcopy(global_model), config)
                local_w=local_updata.train()
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            #--------------------------------------------------------------------------------------------------
            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
        pass
    elif config.name_method == 'FedDistill+':
        print("FedDistill+")
        distills = torch.eye(config.n_class).to(config.device)
        eye = torch.eye(config.n_class).to(config.device)
        # distills = torch.zeros(config.n_class,config.n_class).to(config.device)
        client_distill = []
        client_counts = []

        for epoch in range(config.server_epoch):
            print("epoch : ", epoch + 1)
            local_ws = []
            sum_choose = 0
            freq_client_choose = []
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ", participation_client)
            for num, inde in enumerate(participation_client):
                print("num : ", num)
                local_updata = Local_Update_by_FedDistill(train_data, list_client[inde].idx, copy.deepcopy(global_model),
                                                     config)
                local_w, loacl_distill, counts = local_updata.train(distills, epoch + 1)
                local_ws.append(copy.deepcopy(local_w))
                client_distill.append(loacl_distill)
                client_counts.append(counts)
                sum_choose += freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            # --------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i == 0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0] / sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key] * freq_client_choose[i] / sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            # --------------------------------------------------------------------------------------------------
            total_distills = torch.zeros(config.n_class, config.n_class).to(config.device)
            total_counts = {k: 0 for k in range(config.n_class)}

            for inde in range(len(client_distill)):
                total_distills += client_distill[inde]
                for k in client_counts[inde].keys():
                    total_counts[k] += client_counts[inde][k]
                    pass
                pass

            for k in range(config.n_class):
                if total_counts[k] == 0:
                    total_counts[k] = 1
                    total_distills[k] += eye[k]
                    pass
                pass

            for k in range(config.n_class):
                distills[k] = total_distills[k] / total_counts[k] * 0.5 + distills[k] * 0.5
                # distills[k]=total_distills[k]/total_counts[k]*1.0+distills[k]*0.0
                pass

            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch + 1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)

            pass
        pass
    elif config.name_method=="Fedprox":
        print("Fedprox")
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_Fedprox(train_data, list_client[inde].idx, copy.deepcopy(global_model), config)
                local_w=local_updata.train()
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            #--------------------------------------------------------------------------------------------------
            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method=='Fedproto':
        print("Fedproto")
        for client in list_client:
            client.model=copy.deepcopy(global_model).to(config.device)
            pass
        prototypes = {i: torch.zeros(config.hidden_size) for i in range(config.n_class)}
        client_prototypes=[]
        client_counts=[]

        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_Fedproto(train_data, list_client[inde].idx, config)
                loacl_prototypes, counts=local_updata.train(list_client[inde],prototypes,epoch+1)
                client_prototypes.append(loacl_prototypes)
                client_counts.append(counts)

                pass
            #--------------------------------------------------------------------------------------------------
            total_prototypes = {k: torch.zeros_like(prototypes[k]) for k in prototypes.keys()}
            total_counts = {k: 0 for k in prototypes.keys()}

            for inde in range(len(client_prototypes)):
                for k in client_prototypes[inde].keys():
                    total_prototypes[k] += client_prototypes[inde][k] * client_counts[inde][k]
                    total_counts[k] += client_counts[inde][k]
                    pass
                pass
            for k in prototypes.keys():
                if total_counts[k] > 0:
                    prototypes[k] = total_prototypes[k] / total_counts[k]
                    pass
                pass
            list_acc,avg_acc= inference_by_pFL(list_client,test_loader, config.device)
            print("list_acc : ",list_acc)
            acc_list.append(avg_acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(avg_acc, acc_list, ep_list)
        pass
    elif config.name_method=='Fedrep':
        print("Fedrep")
        for client in list_client:
            client.model=copy.deepcopy(global_model).to(config.device)
            pass
        avg_w=global_model.state_dict()

        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_Fedrep(train_data, list_client[inde].idx, copy.deepcopy(avg_w), config)
                local_w=local_updata.train(list_client[inde])
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            #--------------------------------------------------------------------------------------------------
            list_acc,avg_acc= inference_by_pFL(list_client,test_loader, config.device)
            print("list_acc : ",list_acc)
            acc_list.append(avg_acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(avg_acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method=='Fedper':
        print("Fedper")
        for client in list_client:
            client.model=copy.deepcopy(global_model).to(config.device)
            pass
        avg_w=global_model.state_dict()

        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_Fedper(train_data, list_client[inde].idx, copy.deepcopy(avg_w), config)
                local_w=local_updata.train(list_client[inde])
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            #--------------------------------------------------------------------------------------------------
            list_acc,avg_acc= inference_by_pFL(list_client,test_loader, config.device)
            print("list_acc : ",list_acc)
            acc_list.append(avg_acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(avg_acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method=='FedBABU':
        print("FedBABU")
        global_model.fc3.weight.requires_grad=False
        global_model.fc3.bias.requires_grad=False
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch + 1)
            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ", participation_client)
            for num, inde in enumerate(participation_client):
                print("num : ", num)
                local_updata = Local_Update_by_FedBABU(train_data, list_client[inde].idx, copy.deepcopy(global_model),config)
                local_w = local_updata.train()
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            # --------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)

            print("global_model.fc3.weight.requires_grad : ",global_model.fc3.weight.requires_grad)
            print("global_model.fc3.bias.requires_grad : ",global_model.fc3.bias.requires_grad)
            print(global_model.fc3.bias)
            # --------------------------------------------------------------------------------------------------
            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch + 1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method=='FedDW':
        print("FedDW")
        distills = torch.eye(config.n_class).to(config.device)
        eye = torch.eye(config.n_class).to(config.device)
        # distills = torch.zeros(config.n_class,config.n_class).to(config.device)
        client_distill = []
        client_counts = []

        for epoch in range(config.server_epoch):
            print("epoch : ", epoch + 1)

            if (epoch+1)%9==0 and config.coefficient_change==True:
                config.FedDW_coefficient=config.FedDW_coefficient/10.0
                print("FedDW_coefficient  : ",config.FedDW_coefficient)
                pass

            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ", participation_client)
            for num, inde in enumerate(participation_client):
                print("num : ", num)
                local_updata = Local_Update_by_FedDW(train_data, list_client[inde].idx, copy.deepcopy(global_model),config)
                local_w,loacl_distill, counts = local_updata.train(distills, epoch + 1)
                local_ws.append(copy.deepcopy(local_w))
                client_distill.append(loacl_distill)
                client_counts.append(counts)
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            #--------------------------------------------------------------------------------------------------
            total_distills = torch.zeros(config.n_class,config.n_class).to(config.device)
            total_counts = {k: 0 for k in range(config.n_class)}

            for inde in range(len(client_distill)):
                total_distills += client_distill[inde]
                for k in client_counts[inde].keys():
                    total_counts[k] += client_counts[inde][k]
                    pass
                pass

            for k in range(config.n_class):
                if total_counts[k]==0:
                    total_counts[k]=1
                    total_distills[k]+=eye[k]
                    pass
                pass

            for k in range(config.n_class):
                distills[k]=total_distills[k]/total_counts[k]*0.5+distills[k]*0.5
                #distills[k]=total_distills[k]/total_counts[k]*1.0+distills[k]*0.0
                pass

            # print("---------------------distilltion---------------------")
            # print(distills)
            # mat_w = torch.mm(global_model.fc3.weight, global_model.fc3.weight.T)
            # # mat_w=F.relu(mat_w)
            # mat_w = torch.softmax(mat_w, dim=1)
            # print("---------------------mat_w---------------------")
            # print(mat_w)
            # Draw_heatmap(distills, mat_w)


            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method == 'FedDW(pFL)':
        print("FedDW(pFL)")
        distills = torch.eye(config.n_class).to(config.device)
        client_distill = []
        client_counts = []
        for client in list_client:
            client.model=copy.deepcopy(global_model).to(config.device)
            pass
        for epoch in range(config.server_epoch):
            # if (epoch+1)%9==0 and config.coefficient_change==True:
            #     config.FedDW_coefficient=config.FedDW_coefficient/10.0
            #     print("FedDW_coefficient  : ",config.FedDW_coefficient)
            #     pass
            print("epoch : ", epoch + 1)
            local_ws = []
            sum_choose = 0
            freq_client_choose = []
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ", participation_client)
            for num, inde in enumerate(participation_client):
                print("num : ", num)
                local_updata = Local_Update_by_FedDW_pFL(train_data, list_client[inde].idx,config)
                loacl_distill, counts = local_updata.train(list_client[inde],distills, epoch + 1)
                client_distill.append(loacl_distill)
                client_counts.append(counts)
                sum_choose += freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            total_distills = torch.zeros(config.n_class, config.n_class).to(config.device)
            total_counts = {k: 0 for k in range(config.n_class)}

            for inde in range(len(client_distill)):
                total_distills += client_distill[inde]
                for k in client_counts[inde].keys():
                    total_counts[k] += client_counts[inde][k]
                    pass
                pass

            for k in range(config.n_class):
                distills[k]=total_distills[k]/total_counts[k]*0.5+distills[k]*0.5
                pass
            list_acc, avg_acc = inference_by_pFL(list_client, test_loader, config.device)
            print("list_acc : ", list_acc)
            acc_list.append(avg_acc)
            ep_list.append(epoch + 1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(avg_acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method=="Per-Fedavg":
        print("Per-Fedavg")
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch+1)
            local_ws = []
            sum_choose=0
            freq_client_choose=[]
            global_model.train()
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ",participation_client)
            for num,inde in enumerate(participation_client):
                print("num : ",num)
                local_updata=Local_Update_by_Per_Fedavg(train_data, list_client[inde].idx, copy.deepcopy(global_model), config)
                local_w=local_updata.train()
                local_ws.append(copy.deepcopy(local_w))
                sum_choose+=freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            #--------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i==0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0]/sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key]*freq_client_choose[i]/sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            #--------------------------------------------------------------------------------------------------
            acc = inference(global_model, test_loader, config.device)
            acc_list.append(acc)
            ep_list.append(epoch+1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(acc, acc_list, ep_list)
            pass
        pass
    elif config.name_method=="FedALA":
        print("FedALA")
        global_model=global_model.to(config.device)
        for client in list_client:
            client.model = copy.deepcopy(global_model).to(config.device)
            pass
        avg_w = global_model.state_dict()

        for epoch in range(config.server_epoch):
            print("epoch : ", epoch + 1)
            local_ws = []
            sum_choose = 0
            freq_client_choose = []
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ", participation_client)
            for num, inde in enumerate(participation_client):
                print("num : ", num)
                local_updata = Local_Update_by_FedALA(train_data, list_client[inde].idx, copy.deepcopy(global_model), config)
                local_w = local_updata.train(list_client[inde])
                local_ws.append(copy.deepcopy(local_w))
                sum_choose += freq_client[inde]
                freq_client_choose.append(freq_client[inde])
                pass
            # --------------------------------------------------------------------------------------------------
            avg_w = copy.deepcopy(local_ws[0])
            for key in avg_w.keys():
                for i in range(0, len(local_ws)):
                    if i == 0:
                        avg_w[key] = local_ws[0][key] * freq_client_choose[0] / sum_choose
                        continue
                        pass
                    avg_w[key] += local_ws[i][key] * freq_client_choose[i] / sum_choose
                    pass
                pass
            global_model.load_state_dict(avg_w)
            # --------------------------------------------------------------------------------------------------
            list_acc, avg_acc = inference_by_pFL(list_client, test_loader, config.device)
            print("list_acc : ", list_acc)
            acc_list.append(avg_acc)
            ep_list.append(epoch + 1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(avg_acc, acc_list, ep_list)
            pass
        pass
    else: #config.name_method=="local_only"
        print("local_only")
        for client in list_client:
            client.model = copy.deepcopy(global_model).to(config.device)
            pass
        for epoch in range(config.server_epoch):
            print("epoch : ", epoch + 1)
            m = max(int(config.participation_rate * config.n_client), 1)
            participation_client = np.random.choice(range(config.n_client), m, replace=False)
            print("participation_client : ", participation_client)
            for num, inde in enumerate(participation_client):
                print("num : ", num)
                local_updata = Local_Update_by_local_only(train_data, list_client[inde].idx, config)
                local_updata.train(list_client[inde])
                pass

            list_acc, avg_acc = inference_by_pFL(list_client, test_loader, config.device)
            check_norm(list_client)
            print("list_acc : ", list_acc)
            acc_list.append(avg_acc)
            ep_list.append(epoch + 1)
            show_accuracy_curve(ep_list, acc_list)
            show_accury_list(avg_acc, acc_list, ep_list)
            pass
        pass
    pass









