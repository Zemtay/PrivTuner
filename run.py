from packet import *
from Bert import *
from dataset import *
from utils import *
from model import *
from train import *
from test import *


def main():

    pre_trained_model = getPTmodel(args) #small models from bert 没用到pre_trained_model
    clients, train_dataloader, _ = init_client(args, device)
    groups = clusterAndkd(args, pre_trained_model, clients) #get the kd small models and group them randomly
    trainClinetModel(args, groups) #client.train() all the samll models   give the output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("--datapath", default="data/sst2.tsv", help="root directory of data")
    parser.add_argument("--dataset", default="bert_sst2")
    parser.add_argument("--gpu", default="1", help="gpu id")
    parser.add_argument("--pretrained_model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=5) #32
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument("--check_step", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--client_number", type=int, default=6)
    parser.add_argument("--group_number", type=int, default=1)
    parser.add_argument("--categories", type=int, default=2)
    parser.add_argument("--task", default="classify")
    parser.add_argument("--save_folder_path", default="params")
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()
    random.seed(args.random_seed)

    time_start = time.time()
    main()
    time_end = time.time()
    print("=" * 50)
    print("Running time:", (time_end - time_start) / 60, "m")
    print("=" * 50)
# writer.add_scalars(tag,{'Train':value}, time)