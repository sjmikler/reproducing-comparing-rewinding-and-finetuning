import argparse
import yaml
from pprint import pprint
import training.tools
import training.run

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--exp",
                        default='experiments/resnet-20-one-shot.yaml',
                        type=str,
                        help="Path to .yaml file with experiments")
arg_parser.add_argument("--gpu",
                        default=0,
                        type=int,
                        help="Which GPU to use in case of multiple GPUs in the system")

args, unknown_args = arg_parser.parse_known_args()
print(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")
print(f"  KNOWN CMD ARGUMENTS: {args.__dict__}")

training.tools.set_visible_gpu([args.gpu])
training.tools.set_memory_growth()
training.tools.set_precision(32)

with open(args.exp, 'r') as f:
    experiments = yaml.safe_load_all(f)
    defaults, *exp_queue = experiments

for idx, _exp in enumerate(exp_queue):
    exp = defaults.copy()
    exp.update(_exp)

    print(f"RUNNING {idx + 1}/{len(exp_queue)}")
    pprint(exp)

    training.run.run_experiment(exp)
    with open(exp["logs"], "a") as f:
        yaml.safe_dump(exp, stream=f, explicit_start=True, sort_keys=False)
    print(f"SAVED DEFINITION LOGS: {exp['logs']}")
