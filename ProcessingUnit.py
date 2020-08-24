import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, float64, int64


''' neural systems '''


# neuron blueprint

class Neuron:
    def __init__(self, tau=30, rand_init=0):

        # params
        self.tau = tau
        self.threshold = -55
        self.rest = -70

        # variables
        self.v = (1-rand_init)*self.rest + rand_init*np.random.randint(-70, -50)  # cupy

        # record
        self.spike = 0

    def run(self, Iext):

        self.v += (self.rest - self.v) / self.tau + Iext
        self.spiking()

    def spiking(self):
        if self.v > self.threshold:
            self.spike = 1
            self.v = self.rest
        else:
            self.spike = 0


# sense
class SensoryNetwork:
    def __init__(self, N, radius, stdv=10, ker_type='Gaussian'):

        # params
        self.stdv = stdv
        self.ker_type = 0 if ker_type == 'Gaussian' else 0
        self.radius = radius//10 + 1  # intensity proportional to the distance
        self.N = N

        # variables
        self.neurons = []
        self.build_net()

        self.current = np.ones(N)*(-360)

        # record
        self.spikes = np.zeros((N, 1))
        self.h = 20
        self.curve = np.zeros(N)

    def build_net(self):
        for _ in range(self.N):
            self.neurons.append(Neuron(tau=50))

    @staticmethod
    def gaussian_kernel(angle, neuron_id, stdv):
        return np.exp(-0.5 * ((neuron_id - angle) / stdv) ** 2) / (stdv * np.sqrt(2 * np.pi))

    def exponential_kernel(self, angle, neuron_id):
        return np.exp(-abs(neuron_id - angle)) / self.stdv

    def dispersion(self, angle):    # only gaussian now
        for i in range(self.N):
            self.current[i] = self.gaussian_kernel(angle=angle, neuron_id=i, stdv=self.stdv)
        self.current = np.clip(self.current/max(self.current), 0.04, 2)

    def run(self, angle, distance):
        self.dispersion(angle)
        avg = []
        for i in range(self.N):
            self.neurons[i].run(np.random.binomial(1, self.current[i])*(self.radius-distance/10))
            self.spikes[i, 0] = self.neurons[i].spike
            avg.append(self.spikes[i, 0])
            if avg.__len__() > self.h: del avg[0]
            self.curve[i] = np.mean(avg)


# hunger
class HungerNet:
    def __init__(self, n=5, maxhunger=11):
        self.n = n
        self.maxhunger = maxhunger
        self.net = []
        self.build()

        # variables
        self.current = 0

        # record
        self.volts = np.zeros(n)
        self.spikes = np.zeros((n, 1))

    def build(self):
        for i in range(self.n):
            self.net.append(Neuron())

    def run(self, hunger):
        self.current = np.random.binomial(1, hunger/self.maxhunger, size=self.n)
        for j in range(self.n):
            self.net[j].run(Iext=self.current[j]*20)
            self.volts[j] = self.net[j].v
            self.spikes[j, 0] = self.net[j].spike


# motion
class ActionUnit:
    def __init__(self, npre, w=1):
        self.npre = npre
        self.preMotor = []
        self.unitMot = Neuron(tau=30)
        self.build()

        self.w = w

        # variables
        self.inps = 0

        # record
        self.spikes = np.zeros(self.npre+1)

    def build(self):
        for i in range(self.npre):
            self.preMotor.append(Neuron(tau=50))

    def run(self, Iexts):
        self.inps = 0
        for j in range(self.npre):
            self.preMotor[j].run(Iexts[j])
            self.inps += self.preMotor[j].spike * self.w
            self.spikes[j] = self.preMotor[j].spike

        self.unitMot.run(Iext=self.inps)
        self.spikes[-1] = self.unitMot.spike


class MotorArea:
    def __init__(self, nunit=4, npre=4, wexc=3, winh=-20):
        self.nunit = nunit
        self.npre = npre
        self.units = []

        self.wexc = wexc
        self.winh = winh

        self.build()

        # record
        self.spikes = np.zeros((nunit, npre+1))
        self.pre_spikes = np.zeros(nunit)
        self.out_spikes = np.zeros((1, nunit))
        self.out_volts = np.zeros(nunit)

    def build(self):
        for i in range(self.nunit):
            self.units.append(ActionUnit(npre=self.npre, w=self.wexc))

    def run(self, Iexts):
        for i in range(self.nunit):  # each units is sequentially run
            self.units[i].run(Iexts[i*self.npre:i*self.npre+self.npre])

            for j in range(self.nunit):  # all the preMotor neurons of the other units are depressed
                if j != i:
                    self.units[j].unitMot.v += self.units[i].spikes[-1] * self.winh

            self.spikes[i, :] = self.units[i].spikes
            self.pre_spikes[i] = sum(self.units[i].spikes[:-1])
            self.out_spikes[0, i] = self.units[i].spikes[-1]
            self.out_volts[i] = self.units[i].unitMot.v


# pool
class ShallowPool:
    def __init__(self, n, sparseness=0.2, maxw=2):
        self.N = n
        self.sparseness = sparseness

        self.neurons = []
        self.weights = np.random.binomial(1, sparseness, size=(n, n)) * np.random.uniform(-maxw, maxw, size=(n, n))

        self.build()

        # record
        self.spikes = np.zeros(n)

    def build(self):
        for i in range(self.N):
            self.neurons.append(Neuron(rand_init=1))

    def run(self, Iext):
        for j in range(self.N):
            self.neurons[j].run(Iext=np.dot(self.spikes, self.weights)[j]+Iext[j])
            self.spikes[j] = self.neurons[j].spike


# default inputs
class DefaultNet:
    def __init__(self, n, sparseness=0.2, maxw=5, intensity=0.4):
        self.N = n
        self.sparseness = sparseness

        self.neurons = []
        self.weights = np.random.binomial(1, sparseness, size=(n, n)) * np.random.uniform(-maxw, maxw, size=(n, n))

        self.build()

        self.intensity = intensity

        # record
        self.spikes = np.zeros((n, 1))

    def build(self):
        for i in range(self.N):
            self.neurons.append(Neuron(rand_init=1))

    def run(self, go):
        for j in range(self.N):
            self.neurons[j].run(Iext=np.dot(self.weights, self.spikes)[j] + go * np.random.binomial(1,
                                                                                                  self.intensity, 1)*5)
            self.spikes[j, 0] = self.neurons[j].spike


# Associative Cortex
class ProcessNet:
    def __init__(self, n_pool, sparse_pool=0.2, w_pool=2,
                       n_sight=360, sparse_sight=0.05, w_sight=2, radius=200,
                       n_hung=5, sparse_hung=0.4, w_hung=4,
                       n_deaf=16, sparse_deaf=0.4, w_deaf=2, w_dfpool=2, sparse_intdeaf=0.3):

        #### pool ####
        self.N = n_pool
        self.sparseness = sparse_pool

        self.neurons = []
        self.weights_pool = np.random.binomial(1, sparse_pool, size=(n_pool, n_pool)) * np.random.uniform(-w_pool,
                                                                                        w_pool, size=(n_pool, n_pool))

        self.build()

        # record
        self.pool_spikes = np.zeros(n_pool)


        #### sight ####
        self.radius = radius
        self.sight = SensoryNetwork(N=n_sight, radius=radius)
        self.weights_sight = np.random.binomial(1, sparse_sight, size=(n_pool, n_sight)) * np.random.uniform(-w_sight,
                                                                                    w_sight, size=(n_pool, n_sight))

        #### default stimuli ####
        self.default = DefaultNet(n=n_deaf, sparseness=sparse_intdeaf, maxw=w_deaf, intensity=0.07)
        self.weights_deaf = np.random.binomial(1, sparse_deaf, size=(n_pool, n_deaf)) * np.random.uniform(-w_dfpool,
                                                                                    w_dfpool, size=(n_pool, n_deaf))
        self.weights_inside_deaf = self.default.weights

        #### hunger ####
        self.hypot = HungerNet(n=n_hung)
        self.weights_hung = np.random.binomial(1, sparse_hung, size=(n_pool, n_hung)) * np.random.uniform(-1,
                                                                                    1, size=(n_pool, n_hung))


    def build(self):
        for i in range(self.N):
            self.neurons.append(Neuron(rand_init=1))

    def run(self, angle, distance, hunger, Iext):

        self.sight.run(angle=angle, distance=distance)
        self.hypot.run(hunger=hunger)
        self.default.run(go=distance > self.radius)

        for j in range(self.N):
            self.neurons[j].run(Iext=np.dot(self.pool_spikes, self.weights_pool)[j] +
                                     np.dot(self.weights_sight, self.sight.spikes)[j] +
                                     np.dot(self.weights_hung, self.hypot.spikes)[j] +
                                     np.dot(self.weights_deaf, self.default.spikes)[j] +
                                     Iext[j])
            self.pool_spikes[j] = self.neurons[j].spike


''' Sensory Network simulation '''

test_sight = 0
if test_sight:
    k = 20
    angles = []
    distances = []
    for u in range(0, 360, 5):
        angles.append([u]*k)
        distances.append([np.random.randint(1, 200)]*k)


    Sight = SensoryNetwork(N=360, radius=200)

    for angle, distance in zip(np.reshape(np.array(angles), angles.__len__()*k),
                               np.reshape(np.array(distances), distances.__len__()*k)):
        Sight.run(angle=angle, distance=distance)

        plt.clf()
        plt.plot(range(360), Sight.curve, '-k')
        plt.title('angle: {} - distance: {}'.format(angle, distance))
        plt.ylim((0, 1.3))
        plt.pause(0.0001)

''' Hunger Network simulation '''

test_hypot = 0
if test_hypot:

    hunglv = 1

    Hunger = HungerNet(n=50)

    while hunglv < 10:
        Hunger.run(hunger=hunglv)

        plt.clf()
        plt.subplot(211)
        plt.plot(1, hunglv, '^r')
        plt.title('hunger: {}'.format(hunglv))
        plt.ylim((0, 11))

        plt.subplot(212)
        plt.imshow([Hunger.spikes], cmap='Greys')
        plt.pause(0.0001)

        hunglv += np.random.binomial(1, 0.3, 1)*0.1

''' Motor Network simulation '''

test_mot = 0
if test_mot:

    Tmax = 2000
    stm = np.random.binomial(1, 0.2, size=(16, Tmax))*5

    MotSy = MotorArea(nunit=4, npre=4, winh=-20)

    for t in range(Tmax):
        MotSy.run(Iexts=stm[:, t])

        plt.clf()
        plt.axhline(y=-55, linestyle='--')
        plt.axhline(y=-70, linestyle='--')

        plt.plot([1, 2, 3, 4], MotSy.out_volts, '^r')
        plt.ylim((-75, -50))

        plt.pause(0.01)


''' Shallow Pool Network simulation '''

test_spool = 0
if test_spool:

    Tmax = 2000
    stm = np.random.binomial(1, 0.1, size=(64, Tmax))*5

    ShaPool = ShallowPool(n=64, sparseness=0.1, maxw=2)

    for t in range(Tmax):
        ShaPool.run(Iext=stm[:, t])

        plt.clf()

        plt.imshow(ShaPool.spikes.reshape((8, 8)), cmap='Greys')

        plt.pause(0.0001)


''' Processing Network simulation '''

test_proc = 0
if test_proc:
    n = 64
    Tmax = 2000
    Procc = ProcessNet(n_pool=n, radius=200)


    # target
    k = 20
    angles = []
    distances = []
    for u in range(0, 360, 5):
        angles.append([u] * k)
        distances.append([np.random.randint(1, 200)] * k)

    # pool stm
    stm = np.random.binomial(1, 0.1, size=(64, angles.__len__()*k))
    stm = np.zeros(n)

    for angle, distance in zip(np.reshape(np.array(angles), angles.__len__() * k),
                               np.reshape(np.array(distances), distances.__len__() * k)):
        Procc.run(angle=angle, distance=distance, hunger=4, Iext=stm)

        plt.clf()

        plt.subplot(211)
        plt.plot(range(360), Procc.sight.curve, '-k')
        plt.title('angle: {} - distance: {}'.format(angle, distance))
        plt.ylim((0, 1.3))

        plt.subplot(223)
        plt.imshow(Procc.pool_spikes.reshape((8, 8)), cmap='Greys')

        plt.subplot(224)
        plt.imshow(Procc.hypot.spikes, cmap='Greys')
        plt.pause(0.0001)
