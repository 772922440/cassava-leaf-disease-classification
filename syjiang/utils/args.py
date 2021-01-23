import argparse


def parse_args(string=None):
    parser = argparse.ArgumentParser(description='Blood_Base')
    # train args

    parser.add_argument('--train_csv', type = str, default = '../Data/train.csv')
    parser.add_argument('--train_dirs', type = str, default = '../Data/train_images/')
    parser.add_argument('--bsize', type=int, default=32, help = "batch size")
    parser.add_argument('--epochs', type=int, default=50,
                        help="epochs")

    parser.add_argument('--device', type=str, default='cuda:0',
                        help="cpu, cuda:0, cuda:1, .....")
    parser.add_argument('--lr', type=float, default=2e-4, #5e-2
                        help="learning rate")
    
    parser.add_argument('--eta_min', type=float, default=2e-5,
                        help="cosin annealing to lr=eta_min")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="LARS optimizer weight_decay")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help="checkpoints")
    parser.add_argument('--model_path', type=str, default='./checkpoints/bestmodel.pth',
                        help='modelpath')

    if string is not None: args = parser.parse_args(string)  
    else: args = parser.parse_args()

    return args 