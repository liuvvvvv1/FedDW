import dataset
from models import *
from utils import *


def initialization(config):
    train_data = None
    test_data = None
    client_idx = None
    model=None
    if config.name_dataset == 'emnist':
        train_data, test_data = dataset.Load_Datasets('emnist')
        pass
    elif config.name_dataset == 'mnist':
        train_data, test_data = dataset.Load_Datasets('mnist')
        pass
    elif config.name_dataset == 'fashion_mnist':
        train_data, test_data = dataset.Load_Datasets('fashion_mnist')
        pass
    elif config.name_dataset == 'cifar10':
        train_data, test_data = dataset.Load_Datasets('cifar10')
        pass
    elif config.name_dataset == 'cifar100':
        train_data, test_data = dataset.Load_Datasets('cifar100')
        pass
    elif config.name_dataset=='IMDB':
        y_train,train_data, test_data = dataset.Load_Datasets('IMDB')
        pass

    if config.name_dataset=='IMDB':
        labels =y_train
        pass
    else:
        labels = np.array(train_data.targets)
        pass



    print("The dataset you have selected is [",config.name_dataset,"]")
    print("----------------------------------------------------------")
    if config.name_model=="CNN_emnist":
        model=CNN_emnist()
        pass
    elif config.name_model=='CNN_mnist':
        model=CNN_mnist()
        pass
    elif config.name_model=='CNN_fashion_mnist':
        model=CNN_fashion_mnist()
        pass
    elif config.name_model=='CNN_cifar10':
        model=CNN_cifar10()
        pass
    elif config.name_model=='ResNet18':
        model=ResNet18(config.n_class)
        pass
    elif config.name_model=='ShuffleNet':
        model = ShuffleNet(num_classes=config.n_class)
        pass
    elif config.name_model=='MobileNet':
        model = MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=2, num_classes=config.n_class)
        pass
    elif config.name_model=='LSTM':
        MAX_WORDS = 10000
        EMB_SIZE = 128
        HID_SIZE = 128
        model = LSTM(MAX_WORDS, EMB_SIZE, HID_SIZE)
        pass

    print("The model you have selected is [",config.name_model,"]")
    print("----------------------------------------------------------")
    print("The method you have selected is [",config.name_method,"]")

    # if config.name_method == 'FedDW' or config.name_method=='FedDW(pFL)':
    #     config.init_align =1
    #     pass
    # else:
    #     config.init_align =0
    #     pass

    if config.init_align == 1:
        model = init_align(model)
        print("init_align = true")
        pass
    else:
        print("init_align = false")
        pass

    if config.name_method=="Fedavg":
        pass
    elif config.name_method=='moon':
        print("moon_coefficient : ",config.moon_coefficient)
        print("max_pre_net_num : ",config.max_pre_net_num)
        pass
    elif config.name_method=='FedHNS':
        print("FedHNS_coefficient : ",config.FedHNS_coefficient)
        print("init_align : ",config.init_align)
        print("classifier_update_interval : ",config.classifier_update_interval)
        pass
    elif config.name_method=='Fedavg_add_align_updata':
        print("init_align : ",config.init_align)
        print("classifier_update_interval : ",config.classifier_update_interval)
        pass
    elif config.name_method=='Fedprox':
        print("Fedprox_coefficient : ",config.Fedprox_coefficient)
        pass
    elif config.name_method=='Fedproto':
        print("Fedproto_coefficient : ",config.Fedproto_coefficient)
        pass
    elif config.name_method=='FedDW':
        print("FedDW_coefficient : ",config.FedDW_coefficient)
        print("coefficient_change : ",config.coefficient_change)

        pass
    elif config.name_method=='FedUV':
        print("std_coeff : ",config.std_coeff)
        print("unif_coeff : ",config.unif_coeff)
        pass

    elif config.name_method=='FedALA':

        pass

    print("----------------------------------------------------------")

    if config.idd == 1:
        client_idx = split_iid(labels, n_clients=config.n_client)
        print("The data distribution pattern you have selected is [iid]")
        pass
    else:
        client_idx = dirichlet_split_noniid(labels, alpha=config.dirichlet_alpha, n_clients=config.n_client)
        print("The data distribution pattern you have selected is [niid]")
        print("dirichlet_alpha : ",config.dirichlet_alpha)
        pass
    print("client_epoch : ",config.client_epoch)
    print("server_epoch : ",config.server_epoch)
    print("participation_rate : ",config.participation_rate)
    return labels,train_data,test_data,model,client_idx
    pass



