import time
import MNN.numpy as np
import MNN
nn = MNN.nn
F = MNN.expr

# open lazy evaluation for train
F.lazy_eval(True)

# month_pay=pow(rate/12+1, times)*(rate/12)*total/(pow(rate/12+1,times)-1)
# Know month_pa, total, times, solve rate
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        one = np.array([0.001])
        one.fix_as_trainable()
        self.rate = one

    def forward(self, times, total):
        r12 = self.rate / 12.0
        r12_1 = r12 + np.array([1.0])
        total_rate = np.power(r12_1, times)
        p0 = (total_rate * r12 * total) / (total_rate-np.array([1.0]))
        return p0

model = Net()
opt = MNN.optim.SGD(model, 0.0000000001, 0.9, 0.0005)

times = np.array([60.0])
month_diff = np.array([1.0])
month_diff.fix_as_placeholder()
month_diff.name = "month_diff"
total = np.array([630000.0])
month_comp = model.forward(times, total)
rates, rates_grad  = opt.grad([month_comp], [month_diff], [model.rate])
lr_rate = np.array([0.0000001])
lr_rate.fix_as_placeholder()
lr_rate.name = "lr_rate"

rates, rates_update = opt.get_update_graph(rates, rates_grad, [lr_rate])
opt.save_graph("update.mnn", [], rates, rates_update)


