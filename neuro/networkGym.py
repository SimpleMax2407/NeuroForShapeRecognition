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

        if self.network.outputs == 1:

            m = len(self.train_examples)

            y = [None] * m
            x_m = [None] * m

            for i in range(0, len(self.train_examples)):
                y[i] = self.train_examples[i]['result']
                x_m[i] = np.insert(np.array(self.train_examples[i]['arguments']), 0, 1)

            y = np.matrix(y)
            x = np.matrix(x_m).T

            p = self.network.predict(x)

            cost = np.sum(-np.multiply(y, np.log(p)) - np.multiply((1 - y), np.log(1 - p))) / m

            # if self.network.hidden:
            #     for theta in self.network.theta:
            #         cost += np.sum([t ** 2 for t in theta[1:]]) * self.lamda / 2 / m
            # else:
            #     cost += np.sum([t ** 2 for t in self.network.theta[1:]]) * self.lamda / 2 / m
        else:
            # TODO: Cost calculation for One vs Many classification
            cost = 0
        return cost

    def train(self, alpha=1, number_of_iterations=10, diode=False, write_log=True, speed_up=False):

        if not self.train_examples:
            raise Exception("There are no training examples for training")

        m = len(self.train_examples)

        y = [None] * m
        x_m = [None] * m

        for i in range(0, len(self.train_examples)):
            y[i] = self.train_examples[i]['result']
            x_m[i] = np.insert(np.array(self.train_examples[i]['arguments']), 0, 1)

        y = np.matrix(y)
        x = np.matrix(x_m).T

        old_cost = -1
        old_theta = []

        for j in range(0, number_of_iterations):

            p = self.network.predict(x)

            if self.network.hidden:
                # TODO: Training of the neural network
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
                grad = x * (p - y).T / m + np.matrix(np.insert(self.network.theta[1:], 0, 0)).T * self.lamda / m

                self.network.theta = self.network.theta - grad * alpha

            cost = self.cost()
            theta = self.network.theta

            if write_log:
                s = ''
                if speed_up:
                    s = " | Alpha: {}".format(alpha)

                print(f'Iteration: {j + 1} | Cost: {cost:.6}{s}')

            if diode or speed_up:
                if old_cost < 0:
                    old_cost = cost
                    old_theta = theta.copy()
                elif old_cost <= cost:
                    if diode:
                        theta = old_theta.copy()
                else:
                    if speed_up:
                        alpha = 1/(old_cost - cost)
                    old_cost = cost
                    old_theta = theta.copy()

        return

    def test(self, border=0.5):

        # TODO: Test for One vs Many classification

        if not self.test_examples:
            raise Exception("There are no test examples for accuracy measuring")

        m = len(self.test_examples)
        trues = 0

        tt = 0
        ff = 0

        for t_e in self.test_examples:

            y = t_e['result']
            trues += y
            x = t_e['arguments']
            p = self.network.predict(np.matrix(np.insert(x, 0,  1)).T, border)

            if p == y:
                if y == 1:
                    tt += 1
                else:
                    ff += 1

        accuracy = (tt + ff) / m

        tt /= trues
        ff /= (m - trues)

        return accuracy, tt, ff
