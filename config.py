import argparse
args = argparse.ArgumentParser()


args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', type=float, default=0.0000325)  # 0.000001
args.add_argument('--epochs', type=int, default=100)
args.add_argument('--hidden', type=int, default=100)
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=1e-5)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)
args.add_argument('--num_features_nonzero', type=int, default=1145)
args = args.parse_args()
print(args)

