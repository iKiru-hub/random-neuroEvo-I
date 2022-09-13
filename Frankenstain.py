import numpy as np
import matplotlib.pyplot as plt
import ProcessingUnit
import keyboard
import time


''' 
- Frankenstein -

I thought the name was appropiate, in the sense that different functional component are glued together for building a -living- entity
'''



class Creature:
    ''' 
    The creature has sensory visual neurons projecting to a main internal pool, which receive projections also from a default network
    and hunger neurons; the pool then project to the motor neurons.
    
    Note: the default and hunger network are silenced by default, because it is faster to see good individuals, feel free to change their weight values
    to explore a larger parameter space. [w_hung, w_deaf, w_dfpool]
    '''
    def __init__(self, n_pool, n_premot, n_sight, n_hung, radius, n_deaf=36, hung_rate=0.01,
                       sparse_pool=0.05, sparse_sight=0.1, sparse_hung=0.05, sparse_deaf=0.1, sparse_intdeaf=0.3,
                       sparse_mot=0.05, w_pool=4, w_sight=25, w_hung=0, w_mot=15,
                       wexc=5, winh=-15, w_deaf=0, w_dfpool=0, 
                       wave_win=200, ext_W=(), pos=(0, 0),
                       gen=0, birthmark='ra', col=(0, 0), food_gain=5):

        # ID
        self.name = ''
        self.race = 'N'
        self.species = 'orphan'
        self.gen = gen
        self.color = (30, 80, 30)
        self.get_name(gen, birthmark, col)


        # params
        self.n_pool = n_pool
        self.n_premot = n_premot
        self.n_sight = n_sight
        self.n_hung = n_hung

        self.n_default = n_deaf
        self.food_gain = food_gain

        self.sqrt_pool = int(np.sqrt(n_pool))
        self.sqrt_hung = int(np.sqrt(n_hung))
        self.sqrt_deaf = int(np.sqrt(n_deaf))

        self.w_pool = w_pool
        self.w_sight = w_sight
        self.w_hung = w_hung
        self.w_mot = w_mot
        self.w_dfpool = w_dfpool
        self.w_deaf = w_deaf
        self.wexc = wexc
        self.winh = winh


        # systems
        self.AssociativeCortex = ProcessingUnit.ProcessNet(n_pool=n_pool, sparse_pool=sparse_pool, w_pool=w_pool,
                                                           n_sight=n_sight, sparse_sight=sparse_sight, w_sight=w_sight,
                                                           radius=radius, n_hung=n_hung, sparse_hung=sparse_hung,
                                                           w_hung=w_hung, n_deaf=n_deaf, sparse_deaf=sparse_deaf,
                                                           w_deaf=w_deaf, w_dfpool=w_dfpool,
                                                           sparse_intdeaf=sparse_intdeaf)
        self.MotorCortex = ProcessingUnit.MotorArea(nunit=4, npre=n_premot, wexc=wexc, winh=winh)

        # Processing Unit to Motor Unit
        self.weights_mot = np.random.binomial(1, sparse_mot, size=(n_premot*4, n_pool)) * np.random.uniform(0, w_mot,
                                                                                            size=(n_premot*4, n_pool))


        # variables
        self.life = True

        self.hunger = 0
        self.hung_rate = hung_rate
        self.variation = 0
        self.death = 10

        self.action = -1

        self.blank_I = np.zeros(self.n_pool)  # the pools doesn't receive any input current

        # food and body
        self.food = 0
        self.eaten = 0
        self.position = [pos[0], pos[1]]

        # DNA
        self.DNA = []
        self.weights_structure = []

        if ext_W:
            self.weights_built(ext_W)
        self.DNA_definition()

        self.genetic_score = 0


        # record
        self.whole_spikes = np.zeros((94, 94))
        self.sight_view = np.zeros((n_sight//4+4, n_sight//4+4))
        # self.wave_win = wave_win
        # self.wave =[]


    def inherit_name(self, name, gen, col):
        self.name = str(gen) + 'C' + name[-5:]
        self.color = col[1]
        self.race = 'Copy'

    def get_name(self, gen, birthmark, col=(0, 0)):
        self.name = str(gen) + birthmark
        for _ in range(4):
            self.name += np.random.choice(('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'w', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'))
        if col[0]:
            self.color = tuple(np.clip([col[1][0]+35, col[1][1]+10, col[1][2]+35], 0, 255))

        if birthmark == 'R':
            self.race = 'Mutated'
        else:
            self.race = 'New'

    def nutrition(self):
        self.hunger = np.around(np.clip(self.hunger + self.hung_rate - self.food*self.food_gain, 0, self.death+1), 3)
        self.life = self.hunger < self.death
        self.hung_rate *= np.max((1, int(self.eaten > 20 and self.food) * 1.1))


    def weights_built(self, wmatrix):
        self.AssociativeCortex.weights_pool = wmatrix[0]
        self.AssociativeCortex.weights_sight = wmatrix[1]
        self.AssociativeCortex.weights_deaf = wmatrix[2]
        self.AssociativeCortex.weights_hung = wmatrix[3]
        self.weights_mot = wmatrix[4]
        self.AssociativeCortex.weights_inside_deaf = wmatrix[5]


    def DNA_definition(self):
        # print(self.AssociativeCortex.weights_pool.flatten().__len__(),
        #       self.AssociativeCortex.weights_sight.flatten().__len__(),
        #       self.AssociativeCortex.weights_deaf.flatten().__len__(),
        #       self.AssociativeCortex.weights_hung.flatten().__len__(),
        #       self.weights_mot.flatten().__len__())
        self.DNA = [self.gene_definition(self.AssociativeCortex.weights_pool),
                    self.gene_definition(self.AssociativeCortex.weights_sight),
                    self.gene_definition(self.AssociativeCortex.weights_deaf),
                    self.gene_definition(self.AssociativeCortex.weights_hung),
                    self.gene_definition(self.weights_mot),
                    self.gene_definition(self.AssociativeCortex.weights_inside_deaf)]

        self.weights_structure = [self.AssociativeCortex.weights_pool,
                                  self.AssociativeCortex.weights_sight,
                                  self.AssociativeCortex.weights_deaf,
                                  self.AssociativeCortex.weights_hung,
                                  self.weights_mot,
                                  self.AssociativeCortex.weights_inside_deaf]

    def gene_definition(self, w):
        w = w.flatten()
        where = []
        what = []
        for i, x in enumerate(w):
            if x != 0:
                where.append(i)
                what.append(x)

        return [where, what]

    def run(self, angle, distance):
        self.AssociativeCortex.run(angle=angle, distance=distance, hunger=self.hunger, Iext=self.blank_I)

        self.MotorCortex.run(Iexts=np.dot(self.weights_mot, self.AssociativeCortex.pool_spikes))

        # action definition
        self.action = (1-self.MotorCortex.out_spikes.__contains__(1)) * -1 + \
                      self.MotorCortex.out_spikes.__contains__(1) * np.argmax(self.MotorCortex.out_spikes)

        # hunger
        self.nutrition()
        self.eaten += int(self.food)

    def rearrange(self):
        # sight
        self.sight_view[0, 2:-2] = self.AssociativeCortex.sight.spikes[:self.n_sight//4].T
        self.sight_view[-1, 2:-2] = np.flip(self.AssociativeCortex.sight.spikes[self.n_sight//2:
                                                                                  self.n_sight//4*3].T)
        self.sight_view[2:-2, 0] = np.flip(self.AssociativeCortex.sight.spikes[self.n_sight//4*3:].T)
        self.sight_view[2:-2, -1] = self.AssociativeCortex.sight.spikes[self.n_sight//4: self.n_sight//2].T

        # pool
        self.whole_spikes[30:30+self.sqrt_pool, 10:10+self.sqrt_pool] = self.AssociativeCortex.pool_spikes.reshape(
            self.sqrt_pool, self.sqrt_pool)

        # hunger
        self.whole_spikes[10:10+self.sqrt_hung, -self.sqrt_hung-10:-10] = self.AssociativeCortex.hypot.spikes.reshape(
            self.sqrt_hung, self.sqrt_hung)

        # motor
        self.whole_spikes[-11:-7, 30:31+self.n_premot] = self.MotorCortex.spikes.reshape(4, self.n_premot+1)
        self.whole_spikes[-11:-7, 40+self.n_premot] = self.MotorCortex.out_spikes

        # default
        self.whole_spikes[40:40+self.sqrt_deaf, -self.sqrt_deaf-15:-15] = self.AssociativeCortex.default.spikes.reshape(
            self.sqrt_deaf, self.sqrt_deaf)

    def weights_activity(self):
        plt.clf()
        for i, w, in enumerate(self.weights_structure):
            plt.subplot(2, 3, i+1)
            plt.imshow(w, cmap='Greys')
            plt.title('Weights DNA of {}'.format(self.name))
            plt.yticks(())
            plt.xticks(())
        plt.pause(0.0001)

    def neural_activity(self):
        self.rearrange()

        # self.wave.append(np.mean(self.AssociativeCortex.pool_spikes))
        # if self.wave.__len__() > self.wave_win: del self.wave[0]

        plt.clf()

        plt.subplot(121)
        plt.imshow(self.sight_view, cmap='Greys')
        plt.title('Sight view {}'.format(self.name))
        plt.xticks(())

        plt.subplot(122)
        plt.imshow(self.whole_spikes, cmap='Greys')
        plt.title('Deep brain of {}'.format(self.name))
        plt.xticks(())

        plt.pause(0.0001)


    def network_connectivity(self):

        plt.figure()

        maxwidth = 0.07

        #### sight
        lowbound = 40
        side = range(self.n_sight//4)
        xsight, ysight = [q+5+1 for q in side] + [side.__len__()+5] * side.__len__() + [side.__len__()-1+5-q for q in side] \
                 + [5] * side.__len__(), [side.__len__()+lowbound]*side.__len__() \
                 + [side.__len__()-1+lowbound-q for q in side] + [lowbound] * side.__len__() + [q+lowbound for q in side]


        #### pool
        lw2, hw2 = 20, 35
        xpool, ypool = np.random.uniform(lw2, hw2, self.n_pool), np.random.uniform(lw2, hw2, self. n_pool)

        # sight -> pool
        self.display_connections(n1=self.n_pool, n2=self.n_sight, x1=xpool, x2=xsight, y1=ypool, y2=ysight,
                                 w=self.AssociativeCortex.weights_sight, r=0.006, colorful=False)

        # internal pool
        self.display_connections(n1=self.n_pool, n2=self.n_pool, x1=xpool, x2=xpool, y1=ypool, y2=ypool,
                                 w=self.AssociativeCortex.weights_pool, r=maxwidth, colorful=False)

        #### motor
        lw3, hw3 = 55, 60
        xpmot, ypmot = [np.random.uniform(lw3, hw3, self.n_premot) for _ in range(4)], \
                     [np.random.uniform(14+max(i*8, 1), 20+max(i*8, 1), self.n_premot) for i in range(4)]

        xmot, ymot = [hw3+7]*4, [17+max(i*8, 1) for i in range(4)]
        for un in range(4):
            for i in range(self.n_premot):
                plt.plot((xpmot[un][i], xmot[un]), (ypmot[un][i], ymot[un]), '-k', linewidth=maxwidth)

        for i in range(4):
            for j in range(4):
                plt.plot((xmot[i], xmot[j]), (xmot[i], xmot[j]), '-r', linewidth=maxwidth)

        # pool -> motor
        for un in range(4):
            for post_premot in range(self.n_premot):
                for pre_pool in range(self.n_pool):
                    if self.weights_mot[un*self.n_premot+post_premot, pre_pool] != 0:
                        plt.plot((xpmot[un][post_premot], xpool[pre_pool]), (ypmot[un][post_premot], ypool[pre_pool]),
                                 '-k', linewidth=maxwidth)


        # hunger
        lw4, hw4 = 20, 30
        xhung, yhung = np.random.uniform(lw4, hw4, size=self.n_hung), np.random.uniform(5, 10, size=self.n_hung)

        self.display_connections(n1=self.n_pool, n2=self.n_hung, x1=xpool, x2=xhung, y1=ypool, y2=yhung,
                                 w=self.AssociativeCortex.weights_hung, r=maxwidth, colorful=False)



        #### default
        lw5, hw5 = 5, 15

        xdef, ydef = np.random.uniform(lw5, hw5, size=self.n_default), np.random.uniform(lw5, hw5, size=self.n_default)

        # pool -> deafult
        self.display_connections(n1=self.n_pool, n2=self.n_default, x1=xpool, x2=xdef, y1=ypool, y2=ydef,
                                 w=self.AssociativeCortex.weights_deaf, r=maxwidth, colorful=False)

        # internal
        self.display_connections(n1=self.n_default, n2=self.n_default, x1=xdef, x2=xdef, y1=ydef, y2=ydef,
                                 w=self.AssociativeCortex.weights_inside_deaf, r=0.01, colorful=False)


        # plot nodes
        m = 2
        plt.plot(xsight, ysight, 'ok', markersize=m)
        plt.plot(xpool, ypool, 'ok', markersize=m)
        plt.plot(xpmot, ypmot, 'ok', markersize=m)
        plt.plot(xmot, ymot, 'ok', markersize=m)
        plt.plot(xhung, yhung, 'ok', markersize=m)
        plt.plot(xdef, ydef, 'ok', markersize=m)

        plt.xticks(())
        plt.yticks(())
        plt.ylim((0, 60))
        plt.title('Internal Connectivity')

        plt.show()


    def display_connections(self, n1, n2, x1, x2, y1, y2, w, r, colorful=True):
        if colorful:
            for i in range(n1):
                for j in range(n2):
                    if w[i, j] > 0:
                        plt.plot((x1[i], x2[j]), (y1[i], y2[j]), '-g', linewidth=abs(w[i, j])*r)
                    elif w[i, j] < 0:
                        plt.plot((x1[i], x2[j]), (y1[i], y2[j]), '-r', linewidth=abs(w[i, j]) * r)
        else:
            for i in range(n1):
                for j in range(n2):
                    if w[i, j] != 0:
                        plt.plot((x1[i], x2[j]), (y1[i], y2[j]), '-k', linewidth=abs(w[i, j])*r)



''' Frankenstein simulation '''

if __name__ == '__main__':

    Tmax = 2000
    Frankie = Creature(n_pool=8**2, n_premot=10, n_sight=100, n_hung=16, n_deaf=16, radius=200, hung_rate=0.0, w_mot=3)


    # target: moving angle and random distances
    k = 5
    angles = []
    distances = []
    for u in range(0, 360, 5):
        angles.append([u] * k)
        distances.append([np.random.randint(1, 200)] * k)

    print('\nA stimulus will change angular position over time with random distance from the network.\n'
            'The "brain" spiking is visible on the right, with the square ini the middle-upper part being the main pool.\n'
            '\nPress < h > for display the internal connectivity (close all the windows to go on)\nPress < e > to exit\n\n'
            )

    for angle, distance in zip(np.reshape(np.array(angles), angles.__len__() * k),
                               np.reshape(np.array(distances), distances.__len__() * k)):
        Frankie.run(angle=angle, distance=distance)

        Frankie.neural_activity()
        if keyboard.is_pressed('h'):

            Frankie.network_connectivity()
  

        if Frankie.life > Frankie.death or keyboard.is_pressed('e'):
            print('\n++++ RIP ++++')
            break

