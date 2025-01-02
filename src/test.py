from tensor import Tensor

a = Tensor(2.0, requires_grad=True)
b = Tensor(1.0, requires_grad=True)
e = (a + b) * (b + 1)

print(f"a: {a}")
print(f"b: {b}")
print(f"e: {e}")

e.backward()
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"e.grad: {e.grad}")

val_a, val_b = 0., 15.
learning_rate = 0.1
#learning_rate = 1 # with lr=1, we end-up diverging
for i in range(10):
    print('################## iteration {} ##################'.format(i))
    print('a ', val_a, ' b ', val_b)
    print('e = ', (val_a+val_b)*(val_b+1))
    a = Tensor(val_a, requires_grad=True)
    b = Tensor(val_b, requires_grad=True)
    e = (a+b)*(b+1)
    e.backward()
    grad_a = a.detach().grad
    grad_b = b.detach().grad
    print('grad_a ', grad_a, ' grad_b ', grad_b)
    # update parameters
    val_a = val_a-learning_rate*grad_a
    val_b = val_b-learning_rate*grad_b