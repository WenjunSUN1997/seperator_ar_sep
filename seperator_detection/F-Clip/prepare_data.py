from dataset.wireframe import main as prepare_label
from dataset.wireframe_line import main as prepare_line
import argparse

def prepare(args):
    prepare_label(args)
    prepare_line(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='../../seperator_detection/sep_dataset/test/fi')
    args = parser.parse_args()
    prepare(args)