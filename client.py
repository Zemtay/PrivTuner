from packet import *
from dataset import *
import sys
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
print(os.getcwd())


class DPSGD(Optimizer):
    def __init__(self, params, lr, sigma, max_grad_norm=None):
        defaults = dict(lr=lr, sigma=sigma, max_grad_norm=max_grad_norm)
        super(DPSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('DPSGD does not support sparse gradients')
                grad.add_(torch.normal(mean=0, std=group['sigma'], size=grad.size()))
                if group['max_grad_norm'] is not None:
                    clip_grad_norm_(p, group['max_grad_norm'])
                p.data.add_(grad, alpha=-group['lr'])

        return loss


class Client:
    def __init__(self, index, args, device):
        self.index = index
        self.args = args
        self.device = device
        self.group = 1

    def get_dataloader(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_group(self, group):
        self.group = group

    def get_modelAdata(self, model):
        self.model = model.to(self.device)
        # self.optimizer = DPSGD(self.model.parameters(), self.args.lr, sigma=15, max_grad_norm=None)#
        self.optimizer = Adam(self.model.parameters(), self.args.lr)

    def save_model(save_path, iteration, optimizer, model, i, time_used, acc):
        # save_path = "client_check.pt"
        torch.save({'iteration': iteration,
                    'optimizer_dict': optimizer.state_dict(),
                    'model_dict': model.state_dict(),
                    'end_time': time_used,
                    'acc': acc},
                   f"client_check{i+1}.pt")
        print("model save success")

    def train(self, i):
        print('the client is: ', i + 1)

        CE = nn.CrossEntropyLoss()
        time_used = 0
        overall = self.model  # get the initial one

        if os.path.exists(f"client_check{0}.pt"):
            #对比不用原来的基准 速度与正确率
            #讨论聚类的依据 iid
            print("yes")
            checkpoint = torch.load(f"client_check{0}.pt")
            self.model.load_state_dict(checkpoint['model_dict'])
            #overall = self.model #指向了同一个内存，同时更新
            overall = copy.deepcopy(self.model)

        for epoch in range(self.args.epochs):
            time_start = time.time()
            self.model = self.model.to(self.device)
            self.model.train() #not self.train 只是将模型设置为训练模式
            #the output

            for batch in tqdm(self.train_dataloader, desc=f"Training Epoch {epoch} client{self.index}:"): #wrong spelling
                inputs, targets = [x.to(self.device) for x in batch]
                bert_output = F.softmax(self.model(inputs), dim=1)
                loss = CE(bert_output, targets)  #train and update itself
                self.optimizer.zero_grad()
                loss.backward()   #to add the dpsgd
                self.optimizer.step()
                # for (name1, param1) in self.model.named_parameters():
                #     for layer in ["classifier"]:
                #         if layer in name1:
                #             print("batch update")
                #             print(param1)
            ###
            time_end = time.time()
            print('test')
            acc = self.test()
            time_used = time_used + time_end - time_start
            print("time_used", time_used)
            # self.save_model(epoch, self.optimizer, self.model, i, time_used, acc)

        epoch = self.args.epochs
        time_used = 0
        acc = 0
        update_layers = ["adapter1", "adapter2", "classifier"]
        if i == 0:
            print("here i am")
            self.save_model(epoch, self.optimizer, self.model, -1, time_used, acc)
        elif i > 0:
            #把当前的模型与原来的基准求平均
            # overall = self.avg_model(self.model, overall) # no need to add self as a param
            print("get the avg model")
            for (name1, param1), (name2, param2) in zip(overall.named_parameters(), self.model.named_parameters()):
                for layer in update_layers:
                    if layer in name1:
                        # if layer in ["classifier"]:
                        #     print(param1, param2)
                        # print(param1, param2)
                        # print(name1, name2, (param1 - param2) * 100000000)
                        # params[name1] = param1
                        # print('!!!!!!!!', layer, name1)
                        avg_param = (param1.data + param2.data) / 2
                        param1.data = avg_param
            self.save_model(epoch, self.optimizer, overall, -1, time_used, acc)


    def test(self):
        self.model.eval()
        acc = 0
        total = 0
        for batch in tqdm(self.test_dataloader, desc=f"Testing clinet{self.index}:"):
            inputs, targets = [x.to(self.device) for x in batch]
            with torch.no_grad():
                bert_output = self.model(inputs)
                acc += (bert_output.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)
                #print("predict target", bert_output.argmax(dim=1), targets) #全0或者全1
        print(f"Acc: {acc / total :.4f}")
        return acc / total

    def get_parameters(self):
        update_layers = ["adapter1", "adapter2", "classifier"]

        params = {}
        for name, param in self.model.named_parameters():
            for layer in update_layers:
                if layer in name:
                    if name not in params:
                        params[name] = torch.zeros_like(param.data)
                    params[name] += param.data
        return params

    def update_parameters(self, average_params):
        for name, param in self.model.named_parameters(): #layer with name
            if name in average_params: #the layers needed to try bert
                param.data = average_params[name]
