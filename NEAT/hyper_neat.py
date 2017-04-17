import gym
import numpy as np
from functools import partial
from itertools import product
from peas.methods.hyperneat import Substrate
from peas.methods.aggregated import AggregatedGenotype, AggregatedHyperNEATDeveloper
from peas.methods.neat import NEATPopulation, NEATGenotype

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
        output_units_begin = self.width * self.height + 1
        output_units_end = output_units_begin + self.action_space()
        output = output[output_units_begin: output_units_end]
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

def run(generations=250, popsize=200):
    input_shape = task.state_shape()
    substrate = Substrate()
    substrate.add_nodes(input_shape, 'l')
    substrate.add_connections('l', 'l')

    cppn_geno_kwds = dict(feedforward=True,
                     inputs=6,
                     weight_range=(-3.0, 3.0),
                     prob_add_conn=0.1,
                     prob_add_node=0.03,
                     bias_as_node=False,
                     types=['sin', 'bound', 'linear', 'gauss', 'sigmoid'])

    output_geno_kwds = dict(feedforward=True,
                            inputs=input_shape[0] * input_shape[1],
                            outputs=task.action_space(),
                            )

    geno = lambda: AggregatedGenotype(
        lambda: [NEATGenotype(**cppn_geno_kwds),
                 NEATGenotype(**output_geno_kwds)])
    pop = NEATPopulation(geno, popsize=popsize, n_components=2, max_cores=6)
    developer = AggregatedHyperNEATDeveloper(substrate=substrate,
                                             sandwich=True,
                                             add_deltas=True,
                                             node_type='tanh')

    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve, task=task, developer=developer))

    return results

run()


