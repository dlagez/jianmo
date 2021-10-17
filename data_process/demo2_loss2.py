import scipy.io as scio
import matplotlib.pyplot as plt

loss_bp = scio.loadmat("loss_bp.mat")['loss'].reshape(-1).astype(float)
loss_kernel = scio.loadmat("loss_kernel.mat")['loss_k'].reshape(-1).astype(float)
loss_random = scio.loadmat("loss_random.mat")['loss_r'].reshape(-1).astype(float)
loss_bp = list(loss_bp)
loss_kernel = list(loss_kernel)
loss_random = list(loss_random)

loss_bp = loss_bp[::10]
loss_kernel = loss_kernel[1::10]
loss_random = loss_random[2::10]


index = list(range(48))



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(index, loss_bp)
ax.plot(index, loss_kernel)
ax.plot(index, loss_random)
plt.show()