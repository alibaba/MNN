from _mnncengine import expr, c_train
class Module(object):
    def __init__(self):
        self.training = True
        self._parameters = {}
        self._children = {}
    def __call__(self, *input, **kwargs):
        return self.forward(input[0])
    def setName(self, name):
        self.name = name
    def train(self, isTraining = True):
        self.training = isTraining
        for m in self._children:
            self._children[m].train(isTraining)
    def _collectParametes(self, result):
        for p in self._parameters:
            result.append(self._parameters[p])
        for m in self._children:
            subPara = self._children[m].parameters()
            for i in subPara:
                result.append(i)
    def parameters(self):
        result = []
        self._collectParametes(result)
        return result
    def loadParameters(self, var):
        result = []
        self._collectParametes(result)
        if len(result) != len(var):
            print("Error for load para")
            return
        for i in range(0, len(result)):
            if (result[i].length != var[i].length):
                print("Error for load para")
                return
        for i in range(0, len(result)):
            result[i].replace(var[i])
            result[i].fix(expr.Trainable)
    def forward(self, inputs):
        raise NotImplementedError
    def clearCache():
        for m in self._children:
            self._children[m].clearCache()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        def remove_from(dicts):
            if name in dicts:
                del d[name]
        if isinstance(value, expr.Var):
            remove_from(self._parameters)
            value.setName(name)
            self._parameters[name] = value
            return
        if isinstance(value, Module) or isinstance(value, c_train.CppModule):
            remove_from(self._children)
            value.setName(name)
            self._children[name] = value
            return
