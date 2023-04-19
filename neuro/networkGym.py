from neuro.neuroNetwork import NeuroNetwork
import numpy as np


class NetworkGym:

    def __init__(self, network=NeuroNetwork(), train_examples=None, test_examples=None):
        self.network = network
        self.train_examples = train_examples
        self.test_examples = test_examples

    def cost(self, lamda=0):

        if not self.train_examples:
            raise Exception("There are no training examples for cost measuring")

        m = len(self.train_examples)

        y = np.zeros((self.network.outputs, m))
        x_m = [None] * m

        for i in range(0, len(self.train_examples)):
            val = self.train_examples[i]['result']
            if self.network.outputs > 1:
                if val > 0:
                    y[val - 1, i] = 1
            else:
                y[0, i] = val

            x_m[i] = np.insert(np.array(self.train_examples[i]['arguments']), 0, 1)

        x = np.matrix(x_m).T
        p = self.network.predict(x)

        cost = -np.multiply(y, np.log(p)) - np.multiply((1 - y), np.log(1 - p))
        s = 0
        m_c, n_c = cost.shape
        for i in range(m_c):
            for j in range(n_c):
                if np.isinf(cost[i, j]):
                    s += 50
                elif not np.isnan(cost[i, j]):
                    s += cost[i, j]

        s = s/m

        if self.network.hidden:
            for theta in self.network.theta:
                theta[0, :] = 0
                s += np.sum(np.multiply(theta, theta)) * lamda / 2 / m
        else:
            theta = self.network.theta
            theta[0, :] = 0
            s += np.sum(np.multiply(theta, theta)) * lamda / 2 / m

        return s

    def train(self, alpha=1, lamda=0, number_of_iterations=10, write_log=True, speed_up=False):

        if not self.train_examples:
            raise Exception("There are no training examples for training")

        m = len(self.train_examples)

        y = np.zeros((self.network.outputs, m))
        x_m = [None] * m

        for i in range(0, len(self.train_examples)):
            val = self.train_examples[i]['result']
            if self.network.outputs > 1:
                if val > 0:
                    y[val - 1, i] = 1
            else:
                y[0, i] = val

            x_m[i] = np.insert(np.array(self.train_examples[i]['arguments']), 0, 1)

        x = np.matrix(x_m).T

        old_cost = -1

        for j in range(0, number_of_iterations):

            p = self.network.predict(x)

            if self.network.hidden:
                n = len(self.network.theta)
                z = [None] * n
                sz = [None] * n

                for i in range(n):

                    if i == 0:
                        z[0] = x.T * self.network.theta[0]
                    else:
                        z[i] = sz[i - 1] * self.network.theta[i]

                    sz[i] = NeuroNetwork.sigmoid(z[i])
                    sz[i] = np.insert(sz[i], 0, 1, axis=1)

                error = [None] * n
                error[n - 1] = sz[n - 1][:, 1:] - y.T

                for i in range(n - 2, -1, -1):
                    error[i] = error[i + 1] * self.network.theta[i + 1].T
                    error[i] = error[i][:, 1:]

                for i in range(n):
                    error[i] = np.multiply(error[i], np.multiply(sz[i][:, 1:], 1.0 - sz[i][:, 1:]))

                grad = [None] * n

                grad[0] = (error[0].T * x.T) / m

                for i in range(1, n):
                    grad[i] = (error[i].T * sz[i - 1]) / m

                for i in range(n):
                    theta = self.network.theta[i]
                    theta[0, :] = 0
                    grad[i] += np.multiply(theta.T, lamda / m)
                    self.network.theta[i] -= np.multiply(grad[i].T, alpha)

            else:
                grad = x * (p - y).T / m

                theta = self.network.theta
                theta[0, :] = 0
                grad += np.multiply(theta, lamda / m)
                self.network.theta -= np.multiply(grad, alpha)

            if write_log or speed_up:
                cost = self.cost(lamda=lamda)

            if write_log:
                s = ''
                if speed_up:
                    s = " | Alpha: {}".format(alpha)

                print(f'Iteration: {j + 1} | Cost: {cost:.6}{s}')

            if speed_up:
                if old_cost > 0 and alpha * 1e-3 > old_cost - cost > 0:
                    alpha *= 1.1

                old_cost = cost

    def test(self, border=0.5):

        if not self.test_examples:
            raise Exception("There are no test examples for accuracy measuring")

        m = len(self.test_examples)

        stat = np.zeros(self.network.outputs + 1)
        num = np.zeros(self.network.outputs + 1)

        for t_e in self.test_examples:

            y = t_e['result']
            x = t_e['arguments']
            p = self.network.predict(np.matrix(x).T, border, first_element_is_one=False, only_output=True)

            stat[y] += 1 if p == y else 0
            num[y] += 1

        accuracy = np.sum(stat) / m

        any_false = True

        if num[0] == 0:
            any_false = False
            num[0] = 1

        stat = np.divide(stat, num)

        return accuracy, stat, any_false
