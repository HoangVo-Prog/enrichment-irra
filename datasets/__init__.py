def build_dataloader(*args, **kwargs):
    from .build import build_dataloader as _build_dataloader

    return _build_dataloader(*args, **kwargs)
