import importlib
import os
import random
import socket
import sys
from collections.abc import Iterable
from copy import copy, deepcopy

import yaml

from tools import utils

print = utils.get_cprint(color='yellow')


class YamlExperimentQueue:
    def __init__(self, experiments=None, path='.queue.yaml'):
        self.path = path
        self.num_popped = 0
        if experiments:  # if None, can just read existing experiments
            self.write_content(experiments)
        else:
            assert os.path.exists(path), "Neither experiments or queue were given!"
            raise NotImplementedError("UNTESTED!")

    def read_content(self):
        with open(self.path, 'r') as f:
            z = list(yaml.safe_load_all(f))
        return [utils.Experiment(exp) for exp in z]

    def write_content(self, exps):
        assert isinstance(exps, Iterable)

        with open(self.path, 'w') as f:
            yaml.safe_dump_all((exp.todict() for exp in exps),
                               stream=f,
                               explicit_start=True,
                               sort_keys=False)

    def append_content(self, exps):
        existing_content = self.read_content()
        exps = existing_content + exps
        self.write_content(exps)

    def __bool__(self):
        z = self.read_content()
        return bool(z)

    def pop(self):
        if self:  # else is empty
            exps = self.read_content()
        else:
            return None
        exp = exps.pop(0)
        self.write_content(exps)
        self.num_popped += 1
        return exp

    def __iter__(self):
        print(f"LOADING EXPERIMENT FROM {self.path}")
        while self:
            exp = self.pop()
            yield exp

    def __len__(self):
        return len(self.read_content()) + self.num_popped

    def close(self):
        os.remove(self.path)


def cool_parse_exp(exp, exp_history, parent_scope={}):
    assert 'E' not in parent_scope
    assert 'E' not in exp

    exp_history_dict = {}
    if exp_history:
        for idx, prev_exp in enumerate(exp_history):
            exp_history_dict[idx] = prev_exp
            exp_history_dict[prev_exp.Name] = prev_exp  # make aliases in history
        exp_history_dict[-1] = exp_history[-1]

    scope = deepcopy(parent_scope)
    scope.update(exp)
    for key, value in exp.items():
        if isinstance(value, utils.Experiment):
            value = cool_parse_exp(value, exp_history, scope)

        elif isinstance(value, str) and value.startswith('parse '):
            org_expr = value
            value = value[6:].strip()
            escope = deepcopy(scope)
            escope['E'] = exp_history_dict  # make experiment history available to user
            value = eval(value, escope, escope)
            print(f"{key}: {org_expr} --> {value}")

        elif isinstance(value, str):  # e.g. for parsing float in scientific notation
            try:
                value = eval(value, {}, {})
            except (NameError, SyntaxError):
                pass
        scope[key] = value
        exp[key] = value
    return exp


def load_from_yaml(yaml_path, cmd_parameters=(), private_keys=()):
    experiments = yaml.safe_load_all(open(yaml_path, "r"))
    experiments = [utils.Experiment(exp) for exp in experiments]
    default = experiments.pop(0)

    assert 'Global' in default, "Global missing from default confi!g"

    parameters = [p for p in cmd_parameters if p.startswith('+')]
    print(f"RECOGNIZED CMD PARAMETERS: {parameters}")

    for cmd_param in parameters:
        try:
            sys.argv.remove(cmd_param)
            param = cmd_param.strip('+ ')
            param = 'default.' + param
            exec(param)
        except Exception as e:
            print(f"ERROR WHEN PARSING {cmd_param}!")
            raise e

    default.HOST = socket.gethostname()
    all_unpacked_experiments = []

    print("FANCY PARSING BEGINS! KEY: VALUE --> PARSED VALUE")
    for global_rep in range(default.Global.repeat):
        unpacked_experiments = []
        for exp in experiments:
            nexp = deepcopy(exp)
            default_cpy = deepcopy(default)

            for key in nexp:
                if key in default_cpy:
                    default_cpy.pop(key)  # necessary to preserve order in dict
            nexp.update(default_cpy)

            for key in private_keys:
                if key in nexp:
                    nexp.pop(key)

            if "RND_IDX" in nexp:  # allow custom RND_IDX
                rnd_idx = nexp.RND_IDX
            else:
                rnd_idx = random.randint(100000, 999999)

            for rep in range(nexp.Repeat):
                nexp_rep = deepcopy(nexp)
                nexp_rep.RND_IDX = rnd_idx
                nexp_rep.REP = rep
                nexp_rep = cool_parse_exp(nexp_rep, unpacked_experiments)
                unpacked_experiments.append(nexp_rep)
        all_unpacked_experiments.extend(unpacked_experiments)

    if path := default.Global.queue:
        queue = YamlExperimentQueue(all_unpacked_experiments, path=path)
    else:
        queue = all_unpacked_experiments
    print(f"QUEUE TYPE: {type(queue)} | QUEUE LENGTH: {len(queue)}")
    return default, queue


def load_python_object(path, scope={}):
    assert isinstance(path, str)
    try:
        func = eval(path, scope, scope)
        return func
    except (NameError, AttributeError) as e:
        try:
            module = importlib.import_module(path)
            return module
        except ModuleNotFoundError as e:
            if '.' in path:
                sep = path.rfind('.')
                scope['__m__'] = importlib.import_module(path[:sep])
                func = eval(f"__m__.{path[sep + 1:]}", scope, scope)
                return func
            else:
                raise e


def solve_python_objects(exp, parent_scope={}):
    new_scope = copy(parent_scope)
    for key, value in exp.items():
        if isinstance(value, utils.Experiment):
            value = solve_python_objects(value, parent_scope=new_scope)

        elif isinstance(value, str) and value.startswith('solve '):
            value = value[6:].strip()
            value = load_python_object(value, scope=new_scope)

        new_scope[key] = value
        exp[key] = value
    return exp

