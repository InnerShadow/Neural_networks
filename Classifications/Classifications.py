import numpy as np
import matplotlib.pyplot as plt

N = 5
bias = 3

def bais():
	x1 = np.random.random(N)
	x2 = x1 + [np.random.randint(10) / 10 for i in range(N)] + bias
	C1 = [x1, x2]

	x1 = x1 = np.random.random(N)
	x2 = x1 - [np.random.randint(10) / 10 for i in range(N)] - 0.1 + bias
	C2 = [x1, x2]

	f = [0 + bias, 1 + bias]

	w2 = 0.5
	w3 = -bias + w2
	w = np.array([-w2, w2, w3])
	for i in range(N):
		x = np.array([C1[0][i], C1[1][i], 1])
		y = np.dot(w, x)
		if y >= 0:
			print("C1")
		else:
			print("C2")

	plt.scatter(C1[0][:], C1[1][:], s = 10, c = 'red')
	plt.scatter(C2[0][:], C2[1][:], s = 10, c = 'blue')
	plt.plot(f)
	plt.grid(True)
	plt.show()

def activation(x):
	return 0 if x <= 0 else 1


def go(C):
	x = np.array([C[0], C[1], 1])
	w1 = [1, 1, -1.5]
	w2 = [1, 1, -0.5]
	w_hidden = np.array([w1, w2])
	w_out = np.array([-1, 1, -0.5])

	sum = np.dot(w_hidden, x)
	out = [activation(x) for x in sum]
	out.append(1)
	out = np.array(out)

	sum = np.dot(w_out, out)
	y = activation(sum)
	return y


def xor():
	C1 = [(1, 0), (0, 1)]
	C2 = [(0, 0), (1 , 1)]

	print(go(C1[0]), go(C1[1]))
	print(go(C2[0]), go(C2[1]))


if __name__ == '__main__':
	#bais()
	xor()
