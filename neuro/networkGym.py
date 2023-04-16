from neuro.neuroNetwork import NeuroNetwork
import numpy as np


class NetworkGym:

    def __init__(self, network=NeuroNetwork(), train_examples=None, test_examples=None, lamda=0):
        self.network = network
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.lamda = lamda

    def cost(self):

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
        # cost = cost.sum()
        s = 0
        m_c, n_c = cost.shape
        for i in range(m_c):
            for j in range(n_c):
                if np.isinf(cost[i, j]):
                    # print(f'Inf: {i}, {j}, Y: {y[i, j]}, P: {p[i, j]}')
                    s += 50
                # elif np.isnan(cost[i, j]):
                #     print(f'NaN: {i}, {j}, Y: {y[i, j]}, P: {p[i, j]}')
                # elif p[i, j] > 0.5 and y[i, j] == 0:
                #     print(f'{cost[i, j]}: {i}, {j}, Y: {y[i, j]}, P: {p[i, j]}')
                elif not np.isnan(cost[i, j]):
                    s += cost[i, j]

        # if self.network.hidden:
        #     for theta in self.network.theta:
        #         cost += np.sum(np.multiply(theta[1:, :],  theta[1:, :])) * self.lamda / 2 / m
        # else:
        #     cost += np.sum(np.multiply(self.network.theta[1:, :], self.network.theta[1:, :])) * self.lamda / 2 / m

        return s / m
        # return cost

    def train(self, alpha=1, number_of_iterations=10, write_log=True, speed_up=False):

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
                    self.network.theta[i] -= np.multiply(grad[i].T, alpha)

            else:
                grad = x * (p - y).T / m
                grad += np.matrix(np.insert(self.network.theta[1:], 0, 0)).T * self.lamda / m

                self.network.theta = self.network.theta - grad * alpha

            cost = self.cost()

            if write_log:
                s = ''
                if speed_up:
                    s = " | Alpha: {}".format(alpha)

                print(f'Iteration: {j + 1} | Cost: {cost:.6}{s}')

            if speed_up:
                if old_cost > 0 and 1e-4 > (old_cost - cost) / alpha > 0:
                    alpha *= 1.2

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
            p = self.network.predict(np.matrix(np.insert(x, 0,  1)).T, border, only_output=True)

            stat[y] += 1 if p == y else 0
            num[y] += 1

        accuracy = np.sum(stat) / m

        if num[0] == 0:
            num[0] = 1

        stat = np.divide(stat, num)

        return accuracy, stat
