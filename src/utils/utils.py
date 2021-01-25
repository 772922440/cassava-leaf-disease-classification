import yaml
import os.path as path
import collections
import sys
import torch
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
import re
import os
import math
import time

# support 1e-4
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

# access dict by obj.xxx
class DotDict(dict):
    """
    a.b.c
    >>>data = {
    ...    'api': '/api',
    ...    'data': {
    ...        'api': '/data/api'
    ...    }
    ...}
    >>>obj = DotDict(data)
    >>>obj.api
    '/api'
    >>>obj.data
    {'api': '/data/api'}
    >>>obj.data.api
    '/data/api'
    """
    def __init__(self, data_map=None):
        super(DotDict, self).__init__(data_map)
        if isinstance(data_map, dict):
            for k, v in data_map.items():
                if not isinstance(v, dict):
                    self[k] = v
                else:
                    self.__setattr__(k, DotDict(v))

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


# read yaml config
def read_config(config_name):
    with open(path.join("./config", "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=loader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


# for combining config dict
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# parse command to dict
def parse_command(params):
    result = {}
    for _i, _v in enumerate(params):
        if "=" not in _v:
            continue

        key = _v.split("=")[0].strip()
        val = _v[_v.index('=')+1:].strip()
        result[key] = val
    return result
    
# read default.yam, your_config.yaml and command line key+vals
def read_all_config(params = None):
    if params is None:
        params = sys.argv[1:]

    default_config = read_config('default')
    command_config = parse_command(params)
    if 'name' not in command_config:
        assert False, "please specify your config name. (name=xxx)"

    yaml_config = read_config(command_config['name'])

    # combine dict
    default_config = recursive_dict_update(default_config, yaml_config)
    default_config = recursive_dict_update(default_config, command_config)
    
    config = DotDict(default_config)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device 
    return config


# init logger
def get_logger(config):
    logger = getLogger(config.config_name)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)

    if 'logfile' in config:
        handler2 = FileHandler(filename=config.logfile)
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))