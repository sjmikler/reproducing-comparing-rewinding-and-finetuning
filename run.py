import argparse
import os
import time
from copy import deepcopy

import yaml

from tools import parser, utils

print = utils.get_cprint(color='red')

arg_parser = argparse.ArgumentParser(prefix_chars='-+')
arg_parser.add_argument("--dry",
                        action="store_true",
                        help="skip execution but parse experiments")
arg_parser.add_argument("--exp",
                        default='experiment.yaml',
                        type=str,
                        help="path to .yaml file with experiments")
arg_parser.add_argument("--pick",
                        type=int,
                        nargs='*',
                        help="run only selected experiments, e.g. 0 1 3 or just 1")
args, unknown_args = arg_parser.parse_known_args()
print(f"UNKNOWN CMD ARGUMENTS: {unknown_args}")
print(f"  KNOWN CMD ARGUMENTS: {args.__dict__}")

default_config, experiment_queue = parser.load_from_yaml(yaml_path=args.exp,
                                                         cmd_parameters=unknown_args,
                                                         private_keys=("Global",))
print(f"GLOBAL CONFIG:\n{default_config.Global}")

use_slack = 'slack' in default_config.Global and default_config.Global.slack
if use_slack:
    slacklogger = utils.SlackLogger(config=default_config.Global.slack_config,
                                    host=default_config.HOST,
                                    desc=default_config.get("Desc"))

for exp_idx, exp in enumerate(experiment_queue):
    assert isinstance(exp, utils.Experiment)

    if args.pick and exp_idx not in args.pick:
        print(f"SKIPPING EXPERIMENT {exp_idx} (not picked)")
        continue
    if not exp.Name or exp.Name == "skip":
        print(f"SKIPPING EXPERIMENT {exp_idx} (Name = {exp.Name})")
        continue

    print()
    print(f"NEW EXPERIMENT {exp_idx} / {len(experiment_queue)}:\n{exp}")

    solved_exp = parser.solve_python_objects(deepcopy(exp))
    if backup_diff := solved_exp.difference(exp):
        solved_diff = exp.difference(solved_exp)
        assert solved_diff.keys() == backup_diff.keys()
        print(f"SOLVED DIFF:\n{solved_diff}")

    exp_cp = deepcopy(exp)
    exp = solved_exp

    if args.dry:
        continue

    exp.reset_usage_counts(ignore_keys=['REP', 'RND_IDX', 'HOST',
                                        'Name', 'Desc', 'Repeat', 'Module',
                                        'YamlLog']).freeze()
    try:
        t0 = time.time()
        exp.Run(exp)  # RUN MODULE
        exp.TIME_ELAPSED = time.time() - t0

        exp.unfreeze()
        exp.deep_update(backup_diff)
        exp.freeze()

        exp_by_module = exp_cp.difference(exp)
        print(f"LOGGED BY MODULE:")
        print(exp_by_module)

        if dirpath := os.path.dirname(exp.YamlLog):
            os.makedirs(dirpath, exist_ok=True)
        with open(exp.YamlLog, "a") as f:
            yaml.safe_dump(exp.todict(), stream=f, explicit_start=True, sort_keys=False)
        print(f"SAVED LOGS: {exp['YamlLog']}")

        if use_slack:
            slacklogger.add_exp_report(exp)

    except KeyboardInterrupt:
        print("\n")
        print(f"SKIPPING EXPERIMENT {exp_idx}, WAITING 2 SECONDS BEFORE RESUMING...")
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            if use_slack:
                slacklogger.interrupt_short()
            print(f"EXITED GRACEFULLY!")
            raise KeyboardInterrupt

if isinstance(experiment_queue, parser.YamlExperimentQueue):
    print(f"REMOVING QUEUE {experiment_queue.path}")
    experiment_queue.close()

if use_slack:
    slacklogger.finalize()
    slacklogger.finalize_short()
