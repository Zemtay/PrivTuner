from packet import *


def load_sentence_polarity(data_path, train_ratio=0.8): #, client_i;the task is about sentiment polarity sst2 default 0.8
    all_data = []
    categories = set()

    with open(data_path, "r", encoding="utf8") as file:
        for sample in file.readlines():
            polar, sent = sample.strip().split("\t")
            categories.add(polar) ##what's the use: it may be a multi label problem
            all_data.append((polar, sent))
    length = 1800 #len(all_data) ##can this correctly count?
    train_len = 1440 #int(length // 25 * 24)  # 基于0.8，6个client的处理，从0开始不取最后一个
    train_data = all_data[:train_len] # 从索引3的元素开始取到序列 data 的结束
    test_data = all_data[train_len::]#length // 25 + train_len]
    print("train_len, train_data, test_data", train_len, len(train_data), len(test_data))
    return train_data, test_data, len(categories) #process the data


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.dataset[index]


def coffate_fn(examples, args): ##about the way to get data
    inputs, targets = [], []
    for polar, sent in examples:
        inputs.append(sent)
        targets.append(int(polar))

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)

    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets


def load_data(args):
    # 获取训练、测试数据、分类类别总数
    train_data, test_data, categories = load_sentence_polarity(data_path=args.datapath, train_ratio=args.train_ratio)

    # 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
    train_dataset = BertDataset(train_data)
    test_dataset = BertDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=lambda batch: coffate_fn(batch, args), shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=lambda batch: coffate_fn(batch, args)
    )
    return train_dataloader, test_dataloader, categories
