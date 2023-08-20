
import numpy as np

W1 = np.array(([[-0.2, 0.3, -0.3], [0.1, -0.3, -0.4]]))
W2 = np.array([0.2, 0.3])

def f(x):
	return 2 / (1 + np.exp(-x)) - 1


def df(x):
	return 0.5 * (1 + x)* (1 - x)


def go_forward(inp):
	sum = np.dot(W1, inp)
	out = np.array([f(x) for x in sum])

	sum = np.dot(W2, out)
	y = f(sum)
	return (y, out)


def train(epoch):
	global W1, W2
	lmd = 0.01
	N = 10000
	count = len(epoch)

	for k in range(N):
		x = epoch[np.random.randint(0, count)]
		y, out = go_forward(x[0:3])
		e = y - x[-1]
		delta = e * df(y)
		W2[0] = W2[0] - lmd * delta * out[0]
		W2[1] = W2[1] - lmd * delta * out[1]

		delta2 = W2 * delta * df(out)

		W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
		W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd


def __main__():
	epoch = [(-1, -1, -1, -1),
			(-1, -1, 1, 1), 
			(-1, 1, -1, -1), 
			(-1, 1, 1, 1), 
			(1, -1, -1, -1),
			(1, -1, 1, 1), 
			(1, 1, -1, -1),
			(1, 1, 1, -1)]

	train(epoch)

	for x in epoch:
		y, out = go_forward(x[0:3])
		print(f"Out NN value:  {y} => {x[-1]}")

	print("W1: ", W1)
	print("W2: ", W2)


if __name__ == '__main__':
	__main__()

