
import numpy as np

def activation(x):
	return 0 if x < 0.5 else 1

def go(house, rock, atteaction):
	x = np.array([house, rock, atteaction])
	w11 = [0.3, 0.3, 0]
	w12 = [0.4, -0.5, 1]
	weight1 = np.array([w11, w12])
	weight2 = np.array([-1, 1])

	sum_hidden = np.dot(weight1, x)
	print("Hidden layer sum: ", str(sum_hidden))

	out_hidden = np.array([activation(x) for x in sum_hidden])
	print("Outer hidden layer sum ", str(out_hidden))

	sum_end = np.dot(weight2, out_hidden)
	y = activation(sum_end)

	return y


def __main__():
	house = 1
	rock = 1
	atteaction = 1

	res = go(house, rock, atteaction)

	if res == 1:
		print("True")
	else:
		print("False")


if __name__ == '__main__':
	__main__()
