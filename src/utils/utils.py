import yaml
import os.path as path
import collections

# for combining config dict
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# read yaml config
def read_config(config_name):
    with open(path.join("./config", "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


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
    

def read_all_config(params):
    default_config = read_config('default')
    command_config = parse_command(params)
    yaml_config = read_config(command_config['config'])

    # combine dict
    default_config = recursive_dict_update(default_config, yaml_config)
    default_config = recursive_dict_update(default_config, command_config)
    return default_config