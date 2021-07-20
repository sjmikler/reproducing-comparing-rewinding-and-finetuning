import argparse

import tensorflow as tf

from modules.tf_helper import tf_utils

try:
    from ._initialize import *
except ImportError:
    pass

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--gpu",
                        type=int,
                        nargs='*',
                        help="Which GPUs to use during training, e.g. 0 1 3 or just 1")
arg_parser.add_argument("--no-memory-growth",
                        action="store_true",
                        help="Disables memory growth")
args, unknown_args = arg_parser.parse_known_args()

print(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")
print(f"  KNOWN CMD ARGUMENTS: {args.__dict__}")

if args.gpu:
    gpus = tf.config.get_visible_devices("GPU")
    print(f"SETTING VISIBLE GPUS TO {args.gpu}")
    tf_utils.set_visible_gpu([gpus[idx] for idx in args.gpu])

if not args.no_memory_growth:
    tf_utils.set_memory_growth()


def main(exp):
    print("RUNNING TF-HELPER MODULE")
    tf_utils.set_precision(exp.precision)
