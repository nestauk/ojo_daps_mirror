###########################################################
### Automatically added by daps-utils metaflowtask-init ###
###########################################################

def path_to_init(_file=__file__, cast_to_str=False):
    """Return the path to this file"""
    from pathlib import Path
    path = Path(_file).resolve().parent
    return str(path) if cast_to_str else path


def path_to_this(this):
    """Return the path to what "this" is."""
    return path_to_init() / this


def load(path):
    """Load a config file from the given path."""    
    import yaml, json, configparser
    with open(path) as f:
        if path.suffix in ('.yml', '.yaml'):
            return yaml.safe_load(f)
        if path.suffix == '.json':
            return json.load(f)
        if path.suffix in ('.cnf', '.cfg', '.conf', '.config'):
            config = configparser.ConfigParser()
            config.read(path)
            return config
        if path.suffix in ('.sh',):
            return path
        raise ValueError(f'Unknown config file type "{path.suffix}"')


def recursive_load(path_to_config):
    """Recursively load files in a directory. If the file is not
    a config file, then return the path to that file."""
    config = {}
    for child in path_to_config.iterdir():
        if child.is_dir():
            config[child.name] = recursive_load(child)
        elif child.suffix == '':
            config[child.name] = str(child)
        else:
            config[child.stem] = load(child)
    return config


def load_config():
    """Load all of the config files"""
    path_to_config = path_to_this("config")
    return recursive_load(path_to_config)


def load_current_version():
    """Load the current version of this package."""
    path_to_version = path_to_init(__file__) / "VERSION"
    with open(path_to_version) as f:
        v = f.read()
    return v


__basedir__ = path_to_init(cast_to_str=True)
