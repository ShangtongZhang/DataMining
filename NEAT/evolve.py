import gym
import math
import os
import pickle
import neat
import numpy as np
import nes

class OpenAITask:
    runs_per_net = 5
    n_cups = 4

    def __init__(self):
        self.env = self.get_env()

    def play(self, net, render=False):
        state = self.scale_state(self.env.reset())
        fitness = 0.0
        for step in range(self.step_limit):
            if render:
                self.env.render()
            action = self.get_action(net.activate(state))
            state, reward, done, info = self.env.step(action)
            state = self.scale_state(state)
            fitness += self.get_fitness(reward)
            if done:
                break
        return fitness


class CartPole(OpenAITask):
    gym_name = 'CartPole-v0'
    tag = 'cartpole'
    step_limit = 600
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

class MountainCar(OpenAITask):
    gym_name = 'MountainCar-v0'
    tag = 'mountain-car'
    step_limit = 200
    
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
        return env

class SuperMario:
    runs_per_net = 1
    n_cpus = 1
    step_limit = 100000
    tag = 'super-mario'
    goal = 3266

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
        return state

    def play(self, net, render=True):
        max_x = -1
        step_counter = 0
        info = self.reset()
        state = self.scale_state(info['tiles'])
        for step in range(self.step_limit):
            step_counter += 1
            action = self.get_action(net.activate(state))
            info = self.step(action)
            if info['x'] > max_x:
                max_x = info['x']
                step_counter = 0
            if step_counter > 20:
                break
            if max_x >= self.goal:
                break
            if info['dead']:
                break
            state = self.scale_state(state)
        return max_x

# task = CartPole()
# task = MountainCar()
task = SuperMario()

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    for runs in range(task.runs_per_net):
        fitness = task.play(net)
        fitnesses.append(fitness)
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-%s.txt' % task.tag)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(task.n_cpus, eval_genome)
    winner = pop.run(pe.evaluate)

    with open('winner-%s.bin' % task.tag, 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

def show():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-%s.txt' % task.tag)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('winner-%s.bin' % task.tag, 'rb') as f:
        winner = pickle.load(f)
    print(winner)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    task.play(net, True)

if __name__ == '__main__':
    # show()
    run()