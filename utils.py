from packet import *
from client import *


def save_pretrained(clients, folder_path):

    os.makedirs(folder_path, exist_ok=True)

    for idx, client in enumerate(clients):
        model_path = os.path.join(folder_path, f"client_{idx}.pt")
        torch.save(client.model.state_dict(), model_path)


def cluster(args, clients): #to group randomly
    clinet_num = len(clients) #set to 6
    print('client_num', clinet_num)
    groups = [[clients[0], clients[1], clients[2], clients[3], clients[4], clients[5]]]
    return groups

def init_client(args, device):

    train_data, test_data, _ = load_sentence_polarity(data_path=args.datapath, train_ratio=args.train_ratio)
    # train_dataset = BertDataset(train_data)
    test_dataset = BertDataset(test_data)

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=lambda batch: coffate_fn(batch, args)
    )
# 这行代码定义了一个 collate_fn 函数，它是一个lambda表达式，这个表达式接收一个批次的数据 batch 和一些额外的参数 args，然后调用 coffate_fn 函数来处理这些数据。
# coffate_fn分别处理标签和输入

    clients = []
    train_dataloader = None
    length = len(train_data) // 6
    for index in range(args.client_number):
        if index < args.client_number:
            client = Client(index, args, device)
            train_dataset = BertDataset(train_data[index * length: (index + 1) * length])
            train_dataloader = DataLoader( # 一样的啊 没错 它们是一样的
                train_dataset,
                batch_size=args.batch_size,
                # sampler=sampler_list[index],
                collate_fn=lambda batch: coffate_fn(batch, args),
                shuffle=False,
                #shuffle=True（默认值）：在每个epoch开始时，数据集中的样本将被随机打乱。这意味着每个epoch中样本的顺序都是不同的。随机打乱数据可以帮助模型训练时避免陷入局部最优解，有助于提高模型的泛化能力。
            )
            # print("++++ train_dataset", len(train_dataset)) 120
            # print("++++++++++train_dataloader, test_dataloader", len(train_dataloader), len(test_dataloader)) 24,6
            client.get_dataloader(train_dataloader, test_dataloader)
            clients.append(client)

    return clients, train_dataloader, test_dataloader


def gEdgeModel(args, groups): #to get the edgemodel which has the average param in a group
    edge_models = []
    for group in groups:
        edge_model = group[0]

        all_parameters = [client.get_parameters() for client in group]
        average_params = {}
        for param_name in all_parameters[0].keys():
            average_params[param_name] = sum(params[param_name] for params in all_parameters) / len(all_parameters)
        edge_model.update_parameters(average_params)
        edge_models.append(edge_model)
    return edge_models
