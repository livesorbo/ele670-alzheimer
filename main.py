import argparse
from src.train import train_and_evaluate

def parse_args():
    p = argparse.ArgumentParser(description="ELE670 Alzheimer’s classification (2D CNN)")
    p.add_argument("--train_csv", type=str, default="data/train.csv")
    p.add_argument("--val_csv", type=str, default="data/val.csv")
    p.add_argument("--test_csv", type=str, default="data/test.csv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--multi_slice", action="store_true", help="Stack 3 slices as channels")
    p.add_argument("--num_workers", type=int, default=0) 
    p.add_argument("--device", type=str, default="cpu") 
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
