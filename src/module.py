from parameter import Parameter

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
            
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def parameters(self):
        for _, param in self._parameters.items():
            yield param
        for _, module in self._modules.items():
            yield from module.parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        module_str = ', '.join(f"{name}" for name, module in self._modules.items())
        return f"{self.__class__.__name__}({module_str})"
    
    # def to(self, device):
    #     # Move each parameter to the device (e.g., GPU or CPU)
    #     for param in self.parameters:
    #         param.data = param.data.to(device)
        
    #     return self 