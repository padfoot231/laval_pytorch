#register hook is called everytime gradient is computed
import torch
from torch.autograd import Variable
x = Variable(torch.ones(2,2), requires_grad = True)
y = x+2
z = y*y*3
out = z.mean()
def hook(grad):
    grad_f = grad.data.numpy()
    print("ass ",grad_f)
    print('ass',grad[0,:])
x.register_hook(hook)
print(out.backward())
