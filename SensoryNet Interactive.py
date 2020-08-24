import numpy as np
import matplotlib.pyplot as plt
import pygame



''' neurons '''


class Neuron:

    def __init__(self, wspan, tau=20):

        # params
        self.tau = tau
        self.rest = -70
        self.threshold = -55

        # variables
        self.volt = self.rest

        # record
        self.wspan = wspan
        self.voltages = []
        self.spikes = []


    def run(self, Iext):

        # timestep
        self.volt += (self.rest - self.volt) / self.tau + Iext

        # record
        self.voltages.append(self.volt)

        # check spikes
        self.spiking()

        self.slide(self.voltages)
        self.slide(self.spikes)


    def spiking(self):
        if self.volt > self.threshold:
            self.volt = self.rest
            self.spikes.append(1)
        else:
            self.spikes.append(0)

    def slide(self, vect):
        if vect.__len__() > self.wspan:
            del vect[0]


class Network:
    def __init__(self, wspan, radius, stdv=20, ker_type='Gaussian'):

        # params
        self.stdv = stdv
        self.ker_type = 0 if ker_type == 'Gaussian' else 0
        self.wspan = wspan
        self.radius = radius//10 + 1  # intensity proportional to the distance

        # variables
        self.neurons = []
        self.build_net()

        self.current = np.ones(N)*(-360)

        # record
        self.spikes = np.zeros((1, N))
        self.h = 20
        self.curve = np.zeros(N)
        # self.currs = np.zeros((N, wspan))

    def build_net(self):
        for _ in range(N):
            self.neurons.append(Neuron(wspan=self.wspan, tau=50))

    def gaussian_kernel(self, angle, neuron_id):
        return np.exp(-0.5 * ((neuron_id - angle) / self.stdv) ** 2) / (self.stdv * np.sqrt(2 * np.pi))

    def exponential_kernel(self, angle, neuron_id):
        return np.exp(-abs(neuron_id - angle)) / self.stdv

    def dispersion(self, angle):
        for i in range(N):
            self.current[i] = self.ker_type*self.exponential_kernel(angle=angle, neuron_id=i) + \
                              (1-self.ker_type)*self.gaussian_kernel(angle=angle, neuron_id=i)
        self.current = np.clip( self.current/max(self.current), 0.04, 2)

    def run(self, angle, distance):
        self.dispersion(angle)
        avg = []
        for i in range(N):
            self.neurons[i].run(np.random.binomial(1, self.current[i])*(self.radius-distance/10))
            self.spikes[0, i] = self.neurons[i].spikes[-1]
            avg.append(self.spikes[0, i])
            if avg.__len__() > self.h: del avg[0]
            self.curve[i] = np.mean(avg)



simulate = 0

if simulate == 1:

    ''' simulation '''

    # params
    Tmax = 1000
    N = 360

    cells = Network(wspan=5, ker_type='Gaussian')

    # environment
    stimuli = np.ones(Tmax)-300
    step = 200

    stimuli[30: 30+step] = 50
    stimuli[300: 300+step] = 180
    stimuli[550: 550+step] = 270

    fig = plt.figure(figsize=(90, 3))
    # run
    for t in range(Tmax):
        cells.run(stimuli[t], 10)

        plt.clf()
        plt.imshow(cells.spikes, cmap='Greys')
        plt.pause(0.001)



''' Space '''


class Space:
    def __init__(self, win_x, win_y, radius):
        # for display
        self.win_x, self.win_y = win_x, win_y
        self.env = pygame.display.set_mode((win_x, win_y))
        self.run = True
        self.font = 0

        # for interaction
        self.unlocked = True
        self.position = 0

        # for dynamics
        self.center = (win_x//2, win_y//2)
        self.angle = 0
        self.ang_cos = 0
        self.ang_sin = 0
        self.distance = 0

        # Neurons
        self.network = Network(wspan=5, radius=radius, ker_type='Gaussian')
        self.radius = radius

        self.fig = plt.figure(figsize=(50, 3))

    def run_it(self):

        pygame.init()
        pygame.display.set_caption('SPACE')

        self.font = pygame.font.Font('freesansbold.ttf', 15)

        print('\nSpAcE about to begin')
        while self.run:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False

            self.env.fill((0, 0, 0))
            #print('one ')
            # radius
            pygame.draw.circle(self.env, (10, 10, 200), (self.win_x//2, self.win_y//2), self.radius)
            pygame.draw.circle(self.env, (200, 10, 10), (self.win_x // 2, self.win_y // 2), 7)

            # exit
            pygame.draw.rect(self.env, (200, 10, 10), (self.win_x-10, self.win_y-10, 10, 10))

            self.get_events()

            self.print_text('distance: {}px'.format(self.distance), px=70, py=50)
            self.print_text('angle: {}°'.format(self.angle), px=70, py=70)

            pygame.display.update()

            self.neural_activity()


        pygame.quit()
        print('\nSpAcE dead')

    def get_events(self):

        # get position
        if self.unlocked:
            if pygame.mouse.get_pressed():
                self.position = pygame.mouse.get_pos()

                self.measurements()

        # exit if in the bottom right corner
        if self.position[0] > self.win_x-10 and self.position[1] > self.win_y-10:
            self.run = False

    def measurements(self):
        self.distance = round(np.sqrt((self.center[0]-self.position[0])**2 + (self.center[1]-self.position[1])**2))
        self.ang_sin = round(np.arcsin((self.center[1]-self.position[1]) / self.distance) * (180/np.pi))

        self.ang_cos = round(np.arccos((self.position[0] - self.center[0]) / self.distance) * (180/np.pi))


        if self.ang_sin > 0 and self.ang_cos < 90: self.angle = 90 - self.ang_sin
        elif self.ang_sin > 0 and self.ang_cos > 90: self.angle = 270 + self.ang_sin
        elif self.ang_sin < 0 and self.ang_cos < 90: self.angle = -self.ang_sin + 90
        elif self.ang_sin < 0 and self.ang_cos > 90: self.angle = self.ang_sin + 270

    def print_text(self, message, px, py):

        # message
        text1 = self.font.render(message, True, (200, 200, 200))
        textRect1 = text1.get_rect()
        textRect1.center = (px, py)  # display at the center fo the screen
        self.env.blit(text1, textRect1)

        pygame.display.flip()

    def neural_activity(self):
        if self.distance < self.radius:
            self.network.run(angle=int(self.angle), distance=self.distance)
        else:
            self.network.run(angle=-500, distance=0)

        plt.clf()
        plt.subplot(211)
        plt.plot(range(360), self.network.curve, '-k')
        plt.ylim((0, 1.5))
        plt.title('average activity')

        plt.subplot(212)
        plt.imshow(self.network.spikes, cmap='Greys')
        plt.title('spikes')
        plt.xlabel('360° sensory neurons')
        plt.pause(0.0001)



N = 360
rogue = Space(win_x=400, win_y=400, radius=200)
rogue.run_it()
