import numpy as np
import matplotlib.pyplot as plt

N = 5

def __main__():
	x1 = np.random.random(N)
	x2 = x1 + [np.random.randint(10) / 10 for i in range(N)]
	C1 = [x1, x2]

	x1 = x1 = np.random.random(N)
	x2 = x1 - [np.random.randint(10) / 10 for i in range(N)] - 0.1
	C2 = [x1, x2]

	f = [0, 1]

	w = np.array([-0.3, 0.3])
	for i in range(N):
		x = np.array([C1[0][i], C1[1][i]])
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
