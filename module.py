import numpy as np
from tensor import Tensor

class Module:
    def __init__(self):
        # store child modules and parameters
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        # auto-register Tensors that require grad
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        # auto-register submodules
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        # yield this module’s params…
        for p in self._parameters.values():
            yield p
        # …and all child modules’ params
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # for x = model(x)
        return self.forward(*args, **kwargs)
