# class for allowing iteration over all instances of the Kernel class
class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)

class Filter(metaclass=IterRegistry):
    # registry holding all instances of the Kernel class
    _registry = []

    def __init__(self, name, kernel, operation_type="correlation"):
        # add self to registry
        self._registry.append(self)

        self.name = name
        self.kernel = kernel
        self.operation_type = operation_type

    def __str__(self):
        return self.name

    def getName(self):
        return self.name

    def getKernel(self):
        return self.kernel