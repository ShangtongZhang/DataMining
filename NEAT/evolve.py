import gym
import math
import os
import pickle
import neat
import numpy as np
import nes
from itertools import product
from neat.reporting import *
import logging

class OpenAITask:
    n_cpus = 6
    def __init__(self):
        self.env = self.get_env()

    def play(self, net, render=False, test=True):
        state = self.scale_state(self.env.reset())
        fitness = 0.0
        steps = 0
        fitness_stagation = 0
        for step in range(self.step_limit):
            if render:
                self.env.render()
            action = self.get_action(net.activate(state))
            state, reward, done, info = self.env.step(action)
            steps += 1
            state = self.scale_state(state)
            fitness_inc = self.get_fitness(reward)
            if fitness_inc != 0:
                fitness_stagation = 0
            else:
                fitness_stagation += 1
            if fitness_stagation > 300:
                break
            fitness += fitness_inc
            if done:
                break
        if test:
            fitnesses = np.zeros(self.test_repeat)
            for i in range(self.test_repeat):
                fitnesses[i], _ = self.play(net, False, False)
            mean_fitness = np.mean(fitnesses)
            if mean_fitness >= self.success_threshold:
                return mean_fitness, steps
            else:
                if fitness >= self.success_threshold:
                    return self.success_threshold - 1, steps
                else:
                    return fitness, steps
        else:
            return fitness, steps

    def load_config(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-%s.txt' % self.tag)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        config.fitness_threshold = self.success_threshold
        return config

    def show(self, winner=None):
        config = self.load_config()
        if winner is None:
            with open('winner-%s.bin' % self.tag, 'rb') as f:
                winner = pickle.load(f)
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        print OpenAITask.play(self, net, True, False)

class CartPole(OpenAITask):
    gym_name = 'CartPole-v0'
    tag = 'cartpole'
    step_limit = 200
    test_repeat = 50
    success_threshold = 195
    angle_limit_radians = 15 * math.pi / 180
    position_limit = 2.4

    def __init__(self):
        OpenAITask.__init__(self)

    def scale_state(self, state):
        return [0.5 * (state[0] + self.position_limit) / self.position_limit,
                (state[1] + 0.75) / 1.5,
                0.5 * (state[2] + self.angle_limit_radians) / self.angle_limit_radians,
                (state[3] + 1.0) / 2.0]

    def get_action(self, value):
        return 1 if value[0] > 0.5 else 0

    def get_fitness(self, reward):
        return reward

    def get_env(self):
        return gym.make(self.gym_name)

    def set_step_limit(self, step_limit):
        return

    def run(self):
        config = self.load_config()
        pop_sizes = [300, 250, 200, 150, 100]
        step_limit = 200
        runs = 30
        results = dict()
        for pop_size in pop_sizes:
            success_generation = np.zeros(runs)
            fitness_info = []
            for r in range(runs):
                logger.debug('pop size: %d, step limit: %d, run: %d' % (pop_size, step_limit, r))
                winner, success_generation[r], all_fitnesses = evolve(config, pop_size, step_limit)
                fitness_info.append(all_fitnesses)
            results[(pop_size, step_limit)] = (success_generation, fitness_info, winner)
            with open('statistics-%s.bin' % self.tag, 'wb') as f:
                pickle.dump(results, f)

    def draw(self):
        with open('statistics-%s.bin' % self.tag, 'rb') as f:
            results = pickle.load(f)
        pop_sizes = [300, 250, 200, 150, 100]
        step_limits = [200]
        runs = 30
        for pop_size in pop_sizes:
            for step_limit in step_limits:
                success_generation, all_fitnesses, _ = results[(pop_size, step_limit)]
                success_generation += 1
                success_fitness = np.zeros(runs)
                for run, fitnesses in enumerate(all_fitnesses):
                    for generation_fitness in fitnesses:
                        success_fitness[run] += np.sum(np.array(generation_fitness))
                print 'pop size: %d, avg generation: %f(%f), avg steps %f(%f)' % (
                    pop_size,
                    np.mean(success_generation),
                    np.std(success_generation) / np.sqrt(runs),
                    np.mean(success_fitness),
                    np.std(success_fitness) / np.sqrt(runs)
                )

class MountainCar(OpenAITask):
    gym_name = 'MountainCar-v0'
    tag = 'mountain-car'
    step_limit = 250
    test_repeat = 10
    success_threshold = -110

    def __init__(self):
        OpenAITask.__init__(self)

    def scale_state(self, state):
        return [(state[0] + 1.2) / 1.8,
                (state[1] + 0.07) / 0.14]

    def get_action(self, value):
        return np.argmax(value)

    def get_fitness(self, reward):
        return reward

    def get_env(self):
        env = gym.make(self.gym_name)
        env._max_episode_steps = self.step_limit
        return env

    def set_step_limit(self, step_limit):
        self.step_limit = step_limit
        self.env._max_episode_steps = step_limit

    def run(self):
        config = self.load_config()
        pop_sizes = [300, 250, 200, 150]
        step_limits = [300, 250, 200, 150]
        runs = 30
        results = dict()
        for pop_size in pop_sizes:
            for step_limit in step_limits:
                success_generation = np.zeros(runs)
                fitness_info = []
                for r in range(runs):
                    logger.debug('pop size: %d, step limit: %d, run: %d' % (pop_size, step_limit, r))
                    winner, success_generation[r], all_fitnesses = evolve(config, pop_size, step_limit)
                    fitness_info.append(all_fitnesses)
                results[(pop_size, step_limit)] = (success_generation, fitness_info, winner)
                with open('statistics-%s.bin' % self.tag, 'wb') as f:
                    pickle.dump(results, f)

    def draw(self):
        with open('statistics-%s.bin' % self.tag, 'rb') as f:
            results = pickle.load(f)
        pop_sizes = [300, 250, 200, 150]
        step_limits = [300, 250, 200, 150]
        runs = 30
        for pop_size in pop_sizes:
            for step_limit in step_limits:
                success_generation, all_fitnesses, _ = results[(pop_size, step_limit)]
                success_generation += 1
                success_fitness = np.zeros(runs)
                for run, fitnesses in enumerate(all_fitnesses):
                    for generation_fitness in fitnesses:
                        success_fitness[run] += np.sum(np.array(generation_fitness))
                print 'pop size: %d, step limit: %d, avg generation: %f(%f), avg steps %f(%f)' % (
                    pop_size, step_limit,
                    np.mean(success_generation),
                    np.std(success_generation) / np.sqrt(runs),
                    -np.mean(success_fitness),
                    np.std(success_fitness) / np.sqrt(runs)
                )

class MountainCarCTS(OpenAITask):
    gym_name = 'MountainCarContinuous-v0'
    tag = 'mountain-car-cts'
    step_limit = 250
    test_repeat = 10
    success_threshold = 90

    def __init__(self):
        OpenAITask.__init__(self)

    def scale_state(self, state):
        return [(state[0] + 1.2) / 1.8,
                (state[1] + 0.07) / 0.14]

    def get_action(self, value):
        return np.tanh(value)

    def get_fitness(self, reward):
        return reward

    def get_env(self):
        env = gym.make(self.gym_name)
        env._max_episode_steps = self.step_limit
        return env

    def set_step_limit(self, step_limit):
        self.step_limit = step_limit
        self.env._max_episode_steps = step_limit

    def run(self):
        config = self.load_config()
        pop_sizes = [300, 250, 200, 150]
        step_limits = [300, 250, 200, 150]
        runs = 30
        results = dict()
        for pop_size in pop_sizes:
            for step_limit in step_limits:
                success_generation = np.zeros(runs)
                fitness_info = []
                for r in range(runs):
                    logger.debug('pop size: %d, step limit: %d, run: %d' % (pop_size, step_limit, r))
                    winner, success_generation[r], all_fitnesses = evolve(config, pop_size, step_limit)
                    fitness_info.append(all_fitnesses)
                results[(pop_size, step_limit)] = (success_generation, fitness_info, winner)
                with open('statistics-%s.bin' % self.tag, 'wb') as f:
                    pickle.dump(results, f)

    def draw(self):
        with open('statistics-%s.bin' % self.tag, 'rb') as f:
            results = pickle.load(f)
        pop_sizes = [300, 250, 200, 150]
        step_limits = [300, 250, 200, 150]
        runs = 30
        for pop_size in pop_sizes:
            for step_limit in step_limits:
                success_generation, all_fitnesses, _ = results[(pop_size, step_limit)]
                success_generation += 1
                success_fitness = np.zeros(runs)
                for run, fitnesses in enumerate(all_fitnesses):
                    for generation_fitness in fitnesses:
                        success_fitness[run] += np.sum(np.array(generation_fitness))
                print 'pop size: %d, step limit: %d, avg generation: %f(%f), avg steps %f(%f)' % (
                    pop_size, step_limit,
                    np.mean(success_generation),
                    np.std(success_generation) / np.sqrt(runs),
                    np.mean(success_fitness),
                    np.std(success_fitness) / np.sqrt(runs)
                )

class Pendulum(OpenAITask):
    gym_name = 'Pendulum-v0'
    tag = 'pendulum'
    step_limit = 250
    test_repeat = 5
    success_threshold = -400

    def __init__(self):
        OpenAITask.__init__(self)

    def scale_state(self, state):
        return [state[0], state[1], state[2] / 8]

    def get_action(self, value):
        return 2 * np.tanh(value)

    def get_fitness(self, reward):
        return reward

    def get_env(self):
        env = gym.make(self.gym_name)
        env._max_episode_steps = self.step_limit
        return env

    def set_step_limit(self, step_limit):
        return

    def run(self):
        config = self.load_config()
        winner, _, all_fitnesses = evolve(config, 400, 200)
        while True:
            self.show(winner)

class SuperMario(OpenAITask):
    n_cpus = 1
    step_limit = 100000
    tag = 'super-mario'
    success_threshold = 3000

    def __init__(self):
        self.client = nes.Client()
        self.actions = [self.client.msg_press_right,
                        self.client.msg_press_left,
                        self.client.msg_press_up,
                        self.client.msg_press_down,
                        self.client.msg_press_A]

    def reset(self):
        self.client.reset()
        return self.client.info()

    def step(self, action):
        self.client.send(self.actions[action])
        info = self.client.info()
        info['dead'] = info['state'] == 11
        info['x'] = info['mario']['x']
        return info

    def get_action(self, values):
        return np.argmax(values)

    def scale_state(self, state):
        return state + [1.0]

    def play(self, net, render=True, test=False):
        max_x = -1
        step_counter = 0
        info = self.reset()
        state = self.scale_state(info['tiles'])
        for step in range(self.step_limit):
            step_counter += 1
            values = net.activate(state)
            action = self.get_action(values)
            info = self.step(action)
            if info['x'] > max_x:
                max_x = info['x']
                step_counter = 0
            if step_counter > 20:
                break
            if max_x >= self.success_threshold:
                break
            if info['dead']:
                break
            state = self.scale_state(info['tiles'])
        return max_x

    def set_step_limit(self, step_limit):
        return

    def run(self):
        config = self.load_config()
        winner, _, _ = evolve(config, 250, -1)
        with open('winner-%s.bin' % self.tag, 'wb') as f:
            pickle.dump(winner, f)

class Breakout(OpenAITask):
    gym_name = 'Breakout-v0'
    tag = 'breakout'
    step_limit = int(1e5)
    test_repeat = 0
    success_threshold = 10
    width = 21
    height = 16
    downsample_factor = 10

    def __init__(self):
        OpenAITask.__init__(self)
        self.play = lambda net: OpenAITask.play(self, net, False, False)

    def scale_state(self, state):
        state = np.asarray(state)
        state = np.mean(state, axis=2)
        state /= np.max(state)
        new_state = np.zeros((self.width, self.height))
        for w, h in product(range(self.width), range(self.height)):
            new_state[w, h] = np.mean(state[w * self.downsample_factor: (w + 1) * self.downsample_factor,
                                      h * self.downsample_factor: (h + 1) * self.downsample_factor])
        return new_state.flatten()

    def get_action(self, value):
        return np.argmax(value)

    def get_fitness(self, reward):
        return reward

    def get_env(self):
        return gym.make(self.gym_name)

    def set_step_limit(self, step_limit):
        self.step_limit = step_limit
        return

    def run(self):
        config = self.load_config()
        pop_sizes = [300]
        step_limit = self.step_limit
        runs = 1
        results = dict()
        for pop_size in pop_sizes:
            success_generation = np.zeros(runs)
            fitness_info = []
            for r in range(runs):
                logger.debug('pop size: %d, step limit: %d, run: %d' % (pop_size, step_limit, r))
                winner, success_generation[r], all_fitnesses = evolve(config, pop_size, step_limit)
                fitness_info.append(all_fitnesses)
            results[(pop_size, step_limit)] = (success_generation, fitness_info, winner)
            with open('statistics-%s.bin' % self.tag, 'wb') as f:
                pickle.dump(results, f)

    def draw(self):
        with open('statistics-%s.bin' % self.tag, 'rb') as f:
            results = pickle.load(f)
        pop_sizes = [300]
        step_limits = [200]
        runs = 30
        for pop_size in pop_sizes:
            for step_limit in step_limits:
                success_generation, all_fitnesses, winner = results[(pop_size, step_limit)]
                success_generation += 1
                success_fitness = np.zeros(runs)
                for run, fitnesses in enumerate(all_fitnesses):
                    for generation_fitness in fitnesses:
                        success_fitness[run] += np.sum(np.array(generation_fitness))
                print 'pop size: %d, step limit: %d, avg generation: %f(%f), avg steps %f(%f)' % (
                    pop_size, step_limit,
                    np.mean(success_generation),
                    np.std(success_generation) / np.sqrt(runs),
                    np.mean(success_fitness),
                    np.std(success_fitness) / np.sqrt(runs)
                )
        with open('winner-%s.bin' % self.tag, 'wb') as f:
            pickle.dump(winner, f)

# task = CartPole()
# task = MountainCar()
# task = MountainCarCTS()
# task = Pendulum()
# task = SuperMario()
task = Breakout()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/%s.txt' % task.tag)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness, steps = task.play(net)
    return (fitness, steps)

class CustomReporter(BaseReporter):
    def __init__(self):
        self.all_steps = []

    def found_solution(self, config, generation, best):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        steps = [c.steps for c in itervalues(population)]
        self.all_steps.append(steps)

class SingleEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        '''
        eval_function should take one argument (a genome object) and return
        a single float (the genome's fitness).
        '''
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout

    def evaluate(self, genomes, config):
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.eval_function(genome, config))

def evolve(config, pop_size, step_limit):
    config.pop_size = pop_size
    task.set_step_limit(step_limit)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    reporter = CustomReporter()
    pop.add_reporter(reporter)

    pe = neat.ParallelEvaluator(task.n_cpus, eval_genome)
    winner = pop.run(pe.evaluate)

    return winner, reporter.generation, reporter.all_steps

if __name__ == '__main__':
    task.run()
    # task.draw()
    # task.show()
