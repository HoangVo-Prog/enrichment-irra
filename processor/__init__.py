def do_train(*args, **kwargs):
    from .processor import do_train as _do_train

    return _do_train(*args, **kwargs)


def do_inference(*args, **kwargs):
    from .processor import do_inference as _do_inference

    return _do_inference(*args, **kwargs)
