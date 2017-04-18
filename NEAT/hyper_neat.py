import gym
import numpy as np
from functools import partial
from itertools import product
from peas.methods.hyperneat import Substrate
from peas.methods.aggregated import AggregatedGenotype, AggregatedHyperNEATDeveloper
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.hyperneat import HyperNEATDeveloper

class OpenAITask:
    def __init__(self):
        self.gym_name = 'Breakout-v0'
        self.step_limit = 300
        self.env = gym.make(self.gym_name)
        self.width = 21
        self.height = 16
        self.downsample_factor = 10

    def evaluate(self, network):
        state = self.env.reset()
        fitness = 0
        for step in range(self.step_limit):
            network.flush()
            state = self.preprocess(state)
            # print state.reshape((21, 16))
            output = network.feed(state)
            action = self.get_action(output)
            state, reward, done, info = self.env.step(action)
            fitness += reward
            if info['ale.lives'] != 5:
                break
        if fitness == 0:
            fitness += 1
        return {'fitness': fitness}

    def preprocess(self, state):
        state = np.asarray(state)
        state = np.mean(state, axis=2)
        state /= np.max(state)
        new_state = np.zeros((self.width, self.height))
        for w, h in product(range(self.width), range(self.height)):
            new_state[w, h] = np.mean(state[w * self.downsample_factor: (w + 1) * self.downsample_factor,
                                      h * self.downsample_factor: (h + 1) * self.downsample_factor])
        return new_state.flatten()

    def get_action(self, output):
        output_units_begin = self.width * self.height
        output_units_end = output_units_begin + self.action_space()
        output = output[output_units_begin: output_units_end]
        # output = output[output_units_begin: output_units_end].reshape((2, self.action_space()))
        # output = np.mean(output, axis=0)
        return np.argmax(output)

    def state_shape(self):
        return (self.width, self.height)

    def action_space(self):
        return 6

task = OpenAITask()

def evaluate(individual, task, developer):
    stats = task.evaluate(developer.convert(individual))
    return stats

def solve(individual, task, developer):
    return individual.stats['fitness'] > 10

def run(generations=250, popsize=3):
    input_shape = task.state_shape()
    substrate = Substrate()
    substrate.add_nodes(input_shape, 'i')
    substrate.add_nodes(input_shape, 'h')
    # substrate.add_nodes((task.action_space(), ), 'o')
    substrate.add_nodes((2, 3), 'o')
    substrate.add_connections('i', 'h')
    substrate.add_connections('h', 'o')
    # substrate.add_connections('i', 'o')

    cppn_geno_kwds = dict(feedforward=True,
                     inputs=4,
                     weight_range=(-3.0, 3.0),
                     prob_add_conn=0.1,
                     prob_add_node=0.03,
                     bias_as_node=False,
                     types=['linear', 'sin', 'bound', 'gauss', 'tanh'])

    # output_geno_kwds = dict(feedforward=True,
    #                         inputs=input_shape[0] * input_shape[1],
    #                         outputs=task.action_space(),
    #                         )

    # cppn_geno_kwds2 = dict(feedforward=True,
    #                  inputs=4,
    #                  weight_range=(-3.0, 3.0),
    #                  prob_add_conn=0.1,
    #                  prob_add_node=0.03,
    #                  bias_as_node=False,
    #                  types=['linear', 'sin', 'bound', 'gauss', 'tanh'])

    geno = lambda: NEATGenotype(**cppn_geno_kwds)
    # geno = lambda: AggregatedGenotype(
    #     lambda: [NEATGenotype(**cppn_geno_kwds1),
    #              NEATGenotype(**cppn_geno_kwds2)])
    pop = NEATPopulation(geno, popsize=popsize, n_components=1, max_cores=1)
    # developer = AggregatedHyperNEATDeveloper(substrates=substrates,
    #                                          sandwich=True,
    #                                          add_deltas=False,
    #                                          node_type='tanh')
    developer = HyperNEATDeveloper(substrate=substrate, sandwich=False)

    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve, task=task, developer=developer))

    return results

run()


