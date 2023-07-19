import numpy as np
import matplotlib.pyplot as plt

N = 5
bias = 3

def __main__():
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



if __name__ == '__main__':
	__main__()
