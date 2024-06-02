from packet import *
from Bert import *
from utils import *


def getPTmodel(args):

    if args.task == "classify":
        if args.pretrained_model_name == "bert-base-uncased":
            model = BertSST2Model(class_size=args.categories, origin_output=768, adapter_hidden=1024)
    return model


# def clusterAndkd(args, pre_trained_model, clients):
def clusterAndkd(args, pre_trained_model, clients):
    groups = cluster(args, clients)
    models = PTmodelKD(args, pre_trained_model)  ##see next part
    for i in range(args.group_number): #range(6) 0,1,2,3,4,5
        print('group_number', i, args.group_number)
        print('groups[0]', len(groups[0]))
        for client in groups[i]:
            client.get_modelAdata(models[i]) ##in client 得到对应模型的参数配置

    print('groups', len(groups))
    return groups


def PTmodelKD(args, pre_trained_model): #use the smaller bert model to mimic the kd
    r = []
    for _ in range(args.group_number):
        # model = KD_BertSST2Model(class_size=args.categories, origin_output=768, adapter_hidden=16)
        model = BertSST2Model(class_size=args.categories, origin_output=768, adapter_hidden=16) #smaller than 1024
        r.append(model)

    return r  #不同的组可能有不同的初始化模型参数
