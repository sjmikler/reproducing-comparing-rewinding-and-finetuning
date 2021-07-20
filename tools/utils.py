import datetime
import pprint
import random
import time

from tools import constants as C


def get_cprint(color):
    def cprint(*args, **kwds):
        if not args:
            print(**kwds)
            return

        args = list(args)
        args[0] = C.color2code[color] + '# ' + str(args[0])
        args[-1] = str(args[-1]) + C.color2code['reset']
        print(*args, **kwds)

    return cprint


class LazyExperiment:
    def __init__(self, from_dict=None):
        if from_dict is None:
            from_dict = {}

        self.dict = {k: LazyExperiment(v) if isinstance(v, dict) else v for k, v in
                     from_dict.items()}
        self._recipes = {}

    def _get_recipe(self, key):
        return self.dict[key]

    def _get_object(self, key):
        recipe = self.dict[key]
        if recipe in self._recipes:
            return self._recipes[recipe]
        else:
            self._make_object_from_recipe(recipe)
            assert recipe in self._recipes
            return self._recipes[recipe]

    def _make_object_from_recipe(self, recipe):
        assert isinstance(recipe, str)
        self._recipes[recipe] = None


class Experiment:
    """Dict like structure that allows for dot indexing."""
    _internal_names = ['dict', '_usage_counts', '_ignored_counts', '_frozen']

    def __init__(self, from_dict={}):
        self._usage_counts = {key: 0 for key in from_dict}
        self._ignored_counts = set()
        self._frozen = False

        self.dict = {k: Experiment(v) if isinstance(v, dict) else v for k, v in
                     from_dict.items()}

    def __setattr__(self, key, value):
        if key in self._internal_names:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __getattr__(self, key):
        # complicated for `deepcopy` to work
        # fallbacks to normal dictionary
        if key in super().__getattribute__('dict'):
            return self.__getitem__(key)
        else:
            return self.__getattribute__(key)

    def __getitem__(self, item):
        if item in self.dict:
            self._usage_counts[item] += 1
            return self.dict[item]
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Experiment(value)
        if self._frozen and key in self.dict and value != self.dict[key]:
            raise RuntimeError("Exising values cannot be modified!")
        self.dict[key] = value
        self._usage_counts[key] = 0

    def __iter__(self):
        return self.dict.__iter__()

    def __str__(self):
        return pprint.pformat(self.todict(), sort_dicts=False)

    def __contains__(self, item):
        return self.dict.__contains__(item)

    def __bool__(self):
        if self.dict:
            return True
        else:
            return False

    def freeze(self):
        self._frozen = True
        return self

    def unfreeze(self):
        self._frozen = False
        return self

    def reset_usage_counts(self, ignore_keys):
        self._usage_counts = {key: 0 for key in self.dict}
        self._ignored_counts = set(ignore_keys)
        return self

    def get_unused_parameters(self):
        unused_keys = []
        for key in self.dict.keys():
            if key in self._ignored_counts:
                continue

            if self._usage_counts[key] == 0:
                unused_keys.append(key)
        return unused_keys

    def todict(self):
        new_dict = {}
        for key, value in self.dict.items():
            if isinstance(value, Experiment):
                value = value.todict()
            if hasattr(value, 'tolist'):  # for numpy objects
                value = value.tolist()
            new_dict[key] = value
        return new_dict

    def update(self, other):
        for key in other.keys():
            self._usage_counts[key] = 0
        self.dict.update(other)

    def pop(self, key):
        return self.dict.pop(key)

    def get(self, key):
        return self.dict.get(key)

    def keys(self):
        return self.dict.keys()

    def values(self):
        return self.dict.values()

    def items(self):
        return self.dict.items()

    def difference(self, other):
        if isinstance(other, dict):
            other = Experiment(other)
        diff = Experiment()
        for key, value in other.items():
            if key in self:
                if isinstance(value, Experiment) and isinstance(self[key], Experiment):
                    if d := self[key].difference(value):
                        diff[key] = d
                elif str(self[key]) != str(value):
                    diff[key] = value
            else:
                diff[key] = value
        return diff

    def deep_update(self, other):
        if isinstance(other, dict):
            other = Experiment(other)
        for key, value in other.items():
            if key in self:
                if isinstance(value, Experiment) and isinstance(self[key], Experiment):
                    self[key].deep_update(value)
                    self._usage_counts[key] = 0
                elif str(self[key]) != str(value):
                    self[key] = value
                    self._usage_counts[key] = 0
            else:
                self[key] = value


def filter_argv(argv: list, include: list, exclude: list):
    filtered = []
    adding = False
    for arg in argv:
        if arg[0] in include:
            adding = True
        if arg[0] in exclude:
            adding = False

        if adding:
            filtered.append(arg)
    return filtered


def parse_time(strtime):
    for time_format in C.time_formats:
        try:
            return datetime.datetime.strptime(strtime, time_format)
        except ValueError:
            continue
    raise Exception("UNKNOWN TIME FORMAT!")


class SlackLogger:
    def __init__(self, config, host, desc):
        import slack
        self.host = host
        self.desc = desc
        self.config = config
        self.client = slack.WebClient(config.token)
        self.channel2thread_short = {}
        self.queue_final = []
        self.queue_short = []

    def get_description(self):
        return f'_{self.desc}_'

    def add_exp_report(self, exp):
        message = eval(self.config.say, {'exp': exp}, {'exp': exp})
        message = '`' + message + '`'

        if self.config.get('channel_final'):
            self.queue_final.append(message)

        if self.config.get('channel_short'):
            self.queue_short.append(message)
            self.send_short()

    def add_finish_report(self):
        message = f"Experiment on {self.host} is completed!"
        if self.desc:
            message += f"\n{self.get_description()}"
        self.queue_final.insert(0, message)

    def send_message(self, msg, channel, thread_ts=None, retries=2):
        try:
            response = self.client.chat_postMessage(channel=channel,
                                                    text=msg,
                                                    thread_ts=thread_ts)
            print("SLACK LOGGING SUCCESS!")
            return response
        except Exception as e:
            if retries > 0:
                time.sleep(random.random() * 4 + 1)
                self.send_message(msg, channel, thread_ts, retries=retries - 1)
            else:
                print("SLACK LOGGING FAILED!")
                print(e)

    def get_thread_short(self, channel):
        if channel in self.channel2thread_short:
            return self.channel2thread_short[channel]
        else:
            message = f"Experiment on {self.host} is running!"
            if self.desc:
                message += f"\n{self.get_description()}"

            r = self.send_message(message, channel)
            if r and r['ok']:
                self.channel2thread_short[channel] = r['ts']
                return self.channel2thread_short[channel]

    def send_short(self):
        thread = self.get_thread_short(self.config.channel_short)
        if thread:
            while self.queue_short:
                r = self.send_message(msg=self.queue_short[0],
                                      channel=self.config.channel_short,
                                      thread_ts=thread)
                if r and r['ok']:
                    self.queue_short.pop(0)
                else:
                    break

    def finalize(self):
        if self.queue_final:
            self.add_finish_report()
            final_message = '\n'.join(self.queue_final)
            final_message = final_message.replace('`\n`', '\n')
            final_message = final_message.replace('`', '```')
            self.send_message(final_message, channel=self.config.channel_final)

    def finalize_short(self):
        for channel, thread in self.channel2thread_short.items():
            message = f"Experiment is completed!"
            self.send_message(message, channel, thread)

    def interrupt_short(self):
        for channel, thread in self.channel2thread_short.items():
            message = f"Experiment has been interrupted!"
            self.send_message(message, channel, thread)
