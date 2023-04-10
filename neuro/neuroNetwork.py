import numpy as np


class NeuroNetwork:

    def __init__(self, inputs=2, outputs=1, hidden=[]):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden

        if (np.array(hidden)).ndim > 1:
            raise Exception("Hidden matrix must have only one dimension")

        if hidden and np.count_nonzero(hidden) != len(hidden):
            raise Exception("Hidden matrix must not have zero")

        if not hidden:

            self.theta = np.random.randint(-1000, 1000, (inputs + 1, outputs)) / 100
        else:

            self.theta = [None] * (len(hidden) + 1)
            self.theta[0] = np.matrix(np.random.randint(-1000, 1000, (inputs + 1, hidden[0])) / 100)

            for i in range(0, len(hidden)):

                self.theta[i+1] = np.matrix(np.random.randint(-1000, 1000, (hidden[i]+1, outputs if i == (len(hidden)-1)
                                                                            else hidden[i + 1])) / 100)

    def get_theta(self, thetas):
        self.theta = thetas

        if type(thetas) == list:

            if not thetas:
                self.theta = []
                return

            i, _ = thetas[0].shape
            _, o = thetas[len(thetas) - 1].shape

            for ind in range(len(thetas) - 1):
                _, h = thetas[ind].shape
                self.hidden.append(h)

        else:
            i, o = thetas.shape
            self.hidden = []

        self.inputs = i - 1
        self.outputs = o

    @staticmethod
    def sigmoid(x: float):
        return 1/(1 + np.exp(-x))

    def predict(self, input_data, border=-1, first_element_is_one=True):

        if not first_element_is_one:
            input_data = np.append([[1]], input_data, 0)

        if self.hidden:

            output = input_data

            for i, theta in enumerate(self.theta):

                if i > 0:
                    _, n = output.shape
                    output = 1 / (1 + np.exp(-output))
                    output = np.append(np.matrix(np.ones((1, n))), output, 0)

                output = theta.T * output

        else:
            output = self.theta.T * input_data

        if 1e-2 < border < (1 - 1e-2):
            border = -np.log(1/border - 1)
            bl = np.vectorize(lambda o: 1 if o >= border else 0)
            output = bl(output)
        else:
            bl = np.vectorize(lambda o: 30 if o > 30 else (-30 if o < -30 else o))
            output = bl(output)
            output = 1/(1 + np.exp(-output))

        return output
