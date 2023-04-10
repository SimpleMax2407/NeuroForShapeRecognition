from neuro.neuroNetwork import NeuroNetwork
from neuro.networkGym import NetworkGym

n = NeuroNetwork(3, 1, [5])

ng = NetworkGym(n, train_examples=[{'result': 0, 'arguments': [19, 54, 64]},
                                   {'result': 1, 'arguments': [13, -53, -14]},
                                   {'result': 0, 'arguments': [13, -23, 20]},
                                   {'result': 1, 'arguments': [-4, 90, -101]},
                                   {'result': 0, 'arguments': [1, 0, 0]},
                                   {'result': 1, 'arguments': [-1, 0, 0]},
                                   {'result': 0, 'arguments': [1, 2, 4]},
                                   {'result': 0, 'arguments': [3, 0, 9]},
                                   {'result': 1, 'arguments': [-12, 9, -1]},
                                   {'result': 0, 'arguments': [0, 1, 0]},
                                   {'result': 1, 'arguments': [-3, 2, -5]},
                                   {'result': 0, 'arguments': [-11, 9, 25]},
                                   {'result': 1, 'arguments': [19, 0, -100]},
                                   {'result': 0, 'arguments': [18, 13, 0]},
                                   {'result': 1, 'arguments': [-19, 18, -4]},
                                   {'result': 0, 'arguments': [36, 19, 9]},
                                   {'result': 1, 'arguments': [-90, 23, 1]},
                                   {'result': 0, 'arguments': [5, 12, -8]},
                                   {'result': 1, 'arguments': [-31, 13, 1]},
                                   {'result': 0, 'arguments': [0, -14, 21]},
                                   {'result': 1, 'arguments': [-1, 12, -21]},
                                   {'result': 0, 'arguments': [21, 5, -8]}],
                test_examples=[{'result': 0, 'arguments': [3, 2, 0]},
                               {'result': 1, 'arguments': [0, 2, -7]},
                               {'result': 0, 'arguments': [3, 1, 0]},
                               {'result': 1, 'arguments': [0, -1, -1]},
                               {'result': 0, 'arguments': [1, 3, 4]},
                               {'result': 1, 'arguments': [-11, 12, -8]},
                               {'result': 0, 'arguments': [12, 1, 0]},
                               {'result': 1, 'arguments': [-31, -6, -12]}],
                lamda=0.0)

ng.train(number_of_iterations=100, alpha=0.1, diode=False)

a, tt, ff = ng.test(0.7)

print(f'Accuracy: {a:.1%}\n'
      f'Correct:    True - {tt:.1%}  False - {ff:.1%}\n'
      f'Incorrect:  True determ. as false- {1-tt:.1%}  False determ. as true - {1-ff:.1%}')

print(n.theta)
