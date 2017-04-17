from peas.methods.hyperneat import HyperNEATDeveloper
from peas.networks.rnn import NeuralNetwork

class AggregatedGenotype(object):
    def __init__(self, genos):
        if callable(genos):
            self.genos = genos()
        elif isinstance(genos, list):
            self.genos = genos
        else:
            raise Exception('Failed to construct aggregated geno.')

    def get_genos(self):
        return self.genos

    def distance(self, other):
        dists = 0.0
        for geno1, geno2 in zip(self.genos, other.get_genos()):
            dists += geno1.distance(geno2)
        return dists / len(self.genos)

    def mate(self, other):
        children = []
        for geno1, geno2 in zip(self.genos, other.get_genos()):
            children.append(geno1.mate(geno2))
        return AggregatedGenotype(children)

    def mutate(self, innovations, global_innov):
        for i, geno in enumerate(self.genos):
            geno.mutate(innovations[i], global_innov[i])

class AggregatedNetwork(object):
    def __init__(self, networks):
        self.networks = networks

        self.sandwich = True

    def flush(self):
        for net in self.networks:
            net.flush()

    def feed(self, input_activation, add_bias=True, propagate=1):
        output = input_activation
        for net in self.networks:
            output = net.feed(output, add_bias, propagate)
        return output

class AggregatedHyperNEATDeveloper(HyperNEATDeveloper):
    def __init__(self, substrate,
                 sandwich=False,
                 feedforward=False,
                 add_deltas=False,
                 weight_range=3.0,
                 min_weight=0.3,
                 activation_steps=10,
                 node_type='tanh'):
        HyperNEATDeveloper.__init__(self, substrate, sandwich, feedforward, add_deltas,
                                    weight_range, min_weight, activation_steps, node_type)

    def convert(self, aggregated_geno):
        nets = []
        genos = aggregated_geno.get_genos()
        for geno in genos[: -1]:
            nets.append(HyperNEATDeveloper.convert(self, geno))
        nets.append(NeuralNetwork(genos[-1]))

        return AggregatedNetwork(nets)