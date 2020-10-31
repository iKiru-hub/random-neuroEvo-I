import numpy as np
import matplotlib.pyplot as plt
import pygame
import Frankenstain
#import pickle
import time
import sys



''' 
the following keywords are possible action to be done at runtime:

- K_DOWN: kill all the creatures

- K_RIGHT: show the neural activity of the fittest

- K_3: show the weights values as an image

- K_7: the next generation will be a hero-only run (hero: particulary talented creatures appeared in the past)

- K_9: show the species

- K_s: show the scores

- K_c: show the neural connectivity 

- K_n: show the names of the creatures

- K_e: exit

- K_LEFT: randomly change the position of the food

'''


''' Evolution Instance '''


class BioSphere:
    def __init__(self, n_pop=9, n_copies=1, n_rand=5, muta_rate=0.4, muta_rate2=0.05, hung_rate=0.1, uni_w=5,
                 pos=(0, 0), radius=200, ancient_DNA=(False, 0), ancient_hero=(False, 0)):

        # params
        self.n_pop = n_pop
        self.hung_rate = hung_rate
        self.muta_rate = muta_rate  # probability of mutation
        self.muta_rate2 = muta_rate2  # sparseness of the overlapping random weight matrix
        self.uni_w = uni_w
        self.n_copies = n_copies
        self.n_rand = n_rand

        self.radius = radius

        self.pos = pos

        # creatures
        self.n_pool = 49
        self.n_premot = 10
        self.n_sight = 60
        self.n_hung = 9
        self.n_deaf = 9


        self.matrix_vals = ((self.n_pool, self.n_pool),
                            (self.n_pool, self.n_sight),
                            (self.n_pool, self.n_deaf),
                            (self.n_pool, self.n_hung),
                            (self.n_premot*4, self.n_pool),
                            (self.n_deaf, self.n_deaf))

        # variables
        self.fittest = 0
        self.old_fit = 0
        self.heroes_pantheon = {}
        self.year_of_the_heroes = False
        self.generation = 0

        # population
        self.population = []
        self.ancient_gen = 0
        if not ancient_DNA[0]:
            for i in range(self.n_pop):
                self.population.append(Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot,
                                                             n_sight=self.n_sight, 
                                                             n_hung=self.n_hung, n_deaf=self.n_deaf,
                                                             hung_rate=hung_rate, col=(False, 0),
                                                             radius=radius, pos=pos, birthmark='ra'))
        else:
            self.generation = ancient_DNA[1]['gen']
            self.ancient_gen = ancient_DNA[1]['gen']
            self.from_ancient(fossils=ancient_DNA[1])
            print('\nthe former progenitor gen: ', ancient_DNA[1]['gen'], '\nhis name: ', ancient_DNA[1]['name'], '\n')


        # smart sim
        self.p = 0

        # species
        self.current_species = 0
        self.species = {'current': {'name': self.species_name(gen=0),
                                    'color': (30, 30, 30),
                                    'duration': 0,
                                    'record': [],
                                    'mutations': []},
                        'ancient': {'name': '',
                                    'color': (30, 30, 30),
                                    'duration': 0,
                                    'record': 0,
                                    'mutations': 0}}
        self.score_evolution = []

        # hero
        print('\nA mystical ancient Hero, the strongest of all, lies in his tomb')
        self.hero = 0
        self.next_hero = 0
        if ancient_hero[1] != 0:
            self.hero_birth(DNA=ancient_hero[1]['DNA'], gen_score=ancient_hero[1]['record'])
        else:
            self.hero_birth()


    @staticmethod
    def species_name(gen):
        name = str(gen)
        name += np.random.choice(('cherry', 'gum', 'gun', 'mouse', 'blue', 'red', 'hub', 'yolo',
                                                  'tuck', 'tommy', 'pop', 'efron', 'dope', 'cocaine', 'maria',
                                                  'lsd', 'mdma', 'heroine', 'lazy', 'codeine', 'xanax', 'red',
                                                  'morphine', 'malone', 'eminem', 'felicia', 'cocky', 'boobs'))
        for _ in range(2):
            name += np.random.choice(('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'))

        return name

    def get_species_continuity(self):
        new = self.population[self.fittest]
        print('\n-------> fittest race: {} - score: {}'.format(new.race, new.eaten))

        if new.race == 'New':
            print('\na new Hera is about to begin!')
            # ancient become current
            if self.species['current']['duration'] > self.species['ancient']['duration']:
                self.species['ancient']['name'] = self.species['current']['name']
                self.species['ancient']['color'] = self.species['current']['color']
                self.species['ancient']['duration'] = self.species['current']['duration']
                self.species['ancient']['record'] = self.species['current']['record']
                self.species['ancient']['mutations'] = self.species['current']['mutations']


            # new species definition
            self.current_species = self.species_name(gen=new.gen)

            self.species['current']['name'] = self.current_species
            self.species['current']['color'] = (np.random.randint(30, 60),
                                                np.random.randint(30, 60),
                                                np.random.randint(30, 60))
            self.species['current']['duration'] = 1
            self.species['current']['record'] = [new.eaten]
            self.species['current']['mutations'] = []

        # swap if the residual individual of the former species is again perfomant -> the former species re-takes over
        elif new.species == self.species['ancient']['name']:
            self.species['current']['name'], self.species['ancient']['name'] = self.species['ancient']['name'], \
                                                                               self.species['current']['name']
            self.species['current']['color'], self.species['ancient']['color'] = self.species['ancient']['color'], \
                                                                                 self.species['current']['color']
            self.species['current']['duration'], self.species['ancient']['duration'] = self.species['ancient']['duration'],\
                                                                                       self.species['current']['duration']
            self.species['current']['record'], self.species['ancient']['record'] = self.species['ancient']['record'], \
                                                                                   self.species['current']['record']
            self.species['current']['mutations'], self.species['ancient']['mutations'] = self.species['ancient']['mutations'], \
                                                                                         self.species['current']['mutations']

        elif new.race == 'Mutated':
            self.species['current']['duration'] += 1
            self.species['current']['record'].append(new.eaten)
            self.species['current']['mutations'].append([self.species['current']['duration']-1, new.eaten])
            col = self.species['current']['color']
            self.species['current']['color'] = (col[0] + np.random.randint(0, 20),
                                                col[1] + np.random.randint(0, 20),
                                                col[2] + np.random.randint(0, 20))

        else:
            self.species['current']['duration'] += 1
            self.species['current']['record'].append(new.eaten)


        print('\n|| ++ SPECIES ++ ||\ncurrent species: {} - duration: {} - record: {}'
              '\nancient species: {} - duration: {} '
                  '- record: {}'.format(self.species['current']['name'], self.species['current']['duration'],
                                        self.species['current']['record'], self.species['ancient']['name'],
                                        self.species['ancient']['duration'], self.species['ancient']['record']))


    def fitness(self, rand=0):
        self.old_fit = self.fittest
        self.fittest = 0
        heroes = []
        print('\n---------------------------------------------------\nfittest selection\n---------------------------')
        for i in range(self.n_pop):
            one = self.population[i]
            if one.eaten > self.population[self.fittest].eaten:
                self.fittest = i
                print('fittest: ', one.name, ' - score: ', one.eaten)
            if one.eaten > 30:
                heroes.append(one)
                print('\n(nominal hunger rate: {}\nthis guy s hunger rate: {}'.format(self.hung_rate, one.hung_rate))
            if one.eaten > self.hero.genetic_score:
                self.next_hero = one

        print('chosen: ', self.population[self.fittest].name, '\n---------------------------------------------------')

        print('\n|| Pantheon of Heroes ||\nsize: {}'.format(self.heroes_pantheon.__len__()+heroes.__len__()))
        for hero in heroes:
            self.heroes_pantheon[str(self.heroes_pantheon.__len__())] = {'name': hero.name,
                                                                         'score': hero.eaten,
                                                                         'obj': hero}
            print('\nname: {}\nscore: {}'.format(hero.name, hero.eaten))

        # species
        self.get_species_continuity()
        self.score_evolution.append(self.population[self.fittest].eaten)


    def display_score(self):
        plt.plot(range(self.score_evolution.__len__()), self.score_evolution)
        plt.xlabel('generations')
        plt.ylabel('score')
        plt.title('Evolution of the score over time')
        plt.show()


    def display_genome(self):
        for i, c in enumerate(self.population):
            plt.subplot(2, 9, i+1+self.p)
            plt.imshow(c.AssociativeCortex.weights_pool, cmap='Greys')
        self.p = self.n_pop

    def new_generation(self):
        fittest = self.population[self.fittest]
        fittest2 = self.population[self.old_fit]
        self.generate(DNA=fittest.DNA, gen=fittest.gen, name=fittest.name, col=fittest.color, spec1=fittest.species,
                      DNA2=fittest2.DNA, gen2=fittest2.gen, name2=fittest2.name, col2=fittest2.color,
                      spec2=fittest2.species)
        # print('DNA: \n', fittest.DNA[0], '\n', fittest.DNA[1], '\n', fittest.DNA[2],
        #            '\n', fittest.DNA[3], '\n', fittest.DNA[4], '\n', fittest.DNA[5])

    def golden_generation(self):

        new_population = []

        for i in range(self.n_pop):
            hero = self.heroes_pantheon[str(np.random.randint(0, self.heroes_pantheon.__len__()))]['obj']
            DNA, gen, name, gen, col = hero.DNA, hero.gen, hero.name, hero.gen, hero.color
            babyFrank = Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                              n_sight=self.n_sight, 
                                              n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(False, 0),
                                radius=self.radius, pos=self.pos, ext_W=self.DNA_pass(DNA), gen=gen, birthmark='')
            babyFrank.inherit_name(name=name, gen=gen, col=(True, col))
            new_population.append(babyFrank)

        self.population = new_population

    def from_ancient(self, fossils):
        self.generate(DNA=fossils['DNA'], gen=fossils['gen'], name=fossils['name'], col=fossils['color'],
                      spec1=fossils['species'],
                      DNA2=fossils['DNA'], gen2=fossils['gen'], name2=fossils['name'], col2=fossils['color'],
                      spec2=fossils['species'])


    def hero_birth(self, DNA=0, gen_score=0):
        print('hero genetic score: ', gen_score)
        if DNA:
            self.hero = Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                                  n_sight=self.n_sight,
                                                  n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(False, 0),
                                    radius=self.radius, pos=self.pos, ext_W=self.DNA_pass(DNA), gen=0, birthmark='_')
            self.hero.inherit_name(name='_Hero00', gen=0, col=(True, (100, 100, 250)))
            self.hero.species = '00AncientSpace'
            self.hero.genetic_score = gen_score

        else:
            self.hero = Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                              n_sight=self.n_sight, 
                                              n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(False, 0),
                                              radius=self.radius, pos=self.pos, gen=0,
                                              birthmark='_')
            self.hero.inherit_name(name='_Hero00', gen=0, col=(True, (100, 100, 250)))
            self.hero.species = '00AncientSpace'


    def generate(self, DNA, gen, name, col, spec1, DNA2, gen2, name2, col2, spec2):
        new_population = []

        if col[0] > 250 and col[0] > 250 and col[0] > 250:
            col = (np.random.randint(5, 140), np.random.randint(5, 140), np.random.randint(5, 140))

        # copies
        for i in range(self.n_copies):
            babyFrank = Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                              n_sight=self.n_sight, 
                                              n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(False, 0),
                                radius=self.radius, pos=self.pos, ext_W=self.DNA_pass(DNA), gen=gen, birthmark='C')
            babyFrank.inherit_name(name=name, gen=gen, col=(True, col))
            babyFrank.species = spec1
            new_population.append(babyFrank)

        # copies 2
        for i in range(2):
            babyFrank = Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                              n_sight=self.n_sight, 
                                              n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(False, 0),
                                              radius=self.radius, pos=self.pos, ext_W=self.DNA_pass(DNA2), gen=gen2,
                                              birthmark='C2')
            babyFrank.inherit_name(name=name2, gen=gen2, col=(True, col2))
            babyFrank.species = spec2
            new_population.append(babyFrank)

        # brand new
        for j in range(self.n_pop - self.n_copies - self.n_rand - 2):
            new_population.append(Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                                        n_sight=self.n_sight, 
                                                        n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(False, 0),
                                            radius=self.radius, pos=self.pos, gen=self.generation+1, birthmark='N'))
        # mutated
        for k in range(self.n_rand):
            babyFrank = Frankenstain.Creature(n_pool=self.n_pool, n_premot=self.n_premot, n_hung=self.n_hung,
                                                        n_sight=self.n_sight,
                                                        n_deaf=self.n_deaf, hung_rate=self.hung_rate, col=(True, col),
                radius=self.radius, pos=self.pos, ext_W=self.DNA_mutation(DNA), gen=self.generation+1, birthmark='R')
            babyFrank.species = spec1
            new_population.append(babyFrank)

        self.population = new_population
        self.generation += 1

    def DNA_mutation(self, DNA):

        # mutation of a whole DNA
        new_DNA = []
        for i, gene in enumerate(DNA):
            if np.random.random() < self.muta_rate:
                new_DNA.append(self.gene_mutation(gene=gene, shape=self.matrix_vals[i]))
            else:
                new_DNA.append(self.ribosome(gene=gene, n=self.matrix_vals[i][0], m=self.matrix_vals[i][1]))

        return new_DNA

    def gene_mutation(self, gene, shape):
        # mutation of a gene
        old_w = self.ribosome(gene, n=shape[0], m=shape[1])

        return old_w + np.random.binomial(1, self.muta_rate2, size=(shape[0], shape[1])) * np.random.uniform(-self.uni_w,
                                                                                self.uni_w, size=(shape[0], shape[1]))

    def DNA_pass(self, DNA):
        # from DNA to weights
        new_DNA = []
        for i, gene in enumerate(DNA):
            new_DNA.append(self.ribosome(gene=gene, n=self.matrix_vals[i][0], m=self.matrix_vals[i][1]))
        return new_DNA

    def ribosome(self, gene, n, m):
        # from a gene build a weight matrix
        weights = np.zeros(n*m)

        for i, pos in enumerate(gene[0]):
            weights[pos] = gene[1][i]

        return weights.reshape(n, m)


    def species_track(self):

        print('\n// species track //\n-current\nduration: {}\nrecord: {}\nmutations: {}'.format(
            self.species['current']['duration'], self.species['current']['record'], self.species['current']['mutations']
        ))

        print('\n// species track //\n-ancient\nduration: {}\nrecord: {}\nmutations: {}'.format(
            self.species['ancient']['duration'], self.species['ancient']['record'], self.species['ancient']['mutations']
        ))

        # current track
        plt.plot(range(self.species['current']['duration']), self.species['current']['record'], '-r', label='current')
        plt.ylabel('score')
        plt.xlabel('generations')
        plt.title('current species: {} - duration: {} - record: {}\nancient species: {} - duration: {} '
                  '- record: {}'.format(self.species['current']['name'], self.species['current']['duration'],
                                        max(self.species['current']['record']), self.species['ancient']['name'],
                                        self.species['ancient']['duration'], max(self.species['ancient']['record'])))

        try:
            plt.plot(np.array(self.species['current']['mutations'])[:, 0],
                    np.array(self.species['current']['mutations'])[:, 1], '*r', label='mutations')
        except:
            pass

        plt.legend()

        # ancient track
        plt.plot(range(self.species['ancient']['duration']), self.species['ancient']['record'], '-', color='grey',
                 label='ancient')

        try:
            plt.plot(np.array(self.species['ancient']['mutations'])[:, 0],
                    np.array(self.species['ancient']['mutations'])[:, 1], '*', color='grey', label='mutations')
        except:
            pass

        plt.legend()
        plt.show()

    def sim(self):

        self.display_genome()
        self.fitness(rand=0)

        self.new_generation()
        self.display_genome()
        plt.show()



''' food '''


class Food:
    def __init__(self, pos, boundaries, radius=50):
        self.pos = pos
        self.boundaries = boundaries
        self.eaten = False
        self.radius = radius

    def move(self):  # cupy
        self.pos = (np.random.randint(self.radius, self.boundaries[0]-self.radius),
                    np.random.randint(self.radius, self.boundaries[1]-self.radius))



''' Natural Space '''


class Nature:
    def __init__(self, win_x, win_y, speed, n_pop=9, hung_rate=0.1, radius=200, uni_w=1,
                 store=False, load=False, overwrite_hero=False, filename='nobody', heroname='evoCool309'):

        # params
        self.n_pop = n_pop
        self.speed = speed

        # file
        self.filename = filename
        self.heroname = heroname
        self.store = store
        self.load = load
        self.overwrite_hero = overwrite_hero

        if load:
            in_record = open(self.filename, 'rb')
            ancient_guy = pickle.load(in_record)
            print('\nDNA of the ancient guy: \n', ancient_guy)
            in_record.close()
            print('\n|** from an ancient DNA a new generation is about to rise **|\nLOAD!')

            # hero loading
            in_record = open(heroname, 'rb')
            ancient_hero_DNA = pickle.load(in_record)
            in_record.close()
        else:
            ancient_guy = (False, 0)

            ancient_hero_DNA = (False, 0)

        # for display
        self.win_x, self.win_y = win_x, win_y
        self.env = pygame.display.set_mode((win_x, win_y))
        self.run = True
        self.genocide = False
        self.show_names = False
        self.show_activity = False
        self.show_weights = False
        self.show_species = False
        self.show_scores = False
        self.show_connectivity = False


        # for dynamics
        self.angle = np.zeros(self.n_pop)
        self.ang_cos = 0
        self.ang_sin = 0
        self.distance = 0

        # food
        self.cake = Food(pos=(win_x//2-50, win_y//2-50), boundaries=(win_x, win_y), radius=200)

        # creatures
        self.Bios = BioSphere(n_pop=n_pop, n_copies=int(n_pop*0.1), n_rand=int(n_pop*0.5), muta_rate=0.3,
                              muta_rate2=0.05, hung_rate=hung_rate, uni_w=uni_w,
                              pos=(win_x//2, win_y//2), radius=radius, ancient_DNA=ancient_guy,
                                                                       ancient_hero=ancient_hero_DNA)
        self.survivors = n_pop
        self.best = self.Bios.population[0]
        self.scores = np.zeros(n_pop)   # cupy



    def run_silent(self):
        pygame.quit()

        print('\na silent run is about to start.')
        duration = int(input('how many generation? '))
        t = self.Bios.generation

        ttest = time.time()

        # run
        while self.Bios.generation - t < duration:

            # population living
            self.seq_life(silence=True)

            # rigeneration
            self.rigeneration()

            if self.Bios.generation == 50:
                print('\ntime needed for 10 gen: ', np.around(time.time() - ttest, 3))
                break

        print('\nFINISHED\n')

        if self.store:
            out_record = open(self.filename, 'wb')
            print('\nthe fittest is about to be buried.\ngen: ', self.best.gen, '\nname: ', self.best.name)
            fossil = (True, {'DNA': self.best.DNA,
                      'gen': self.best.gen,
                      'name': self.best.name})
            pickle.dump(fossil, out_record)
            out_record.close()


    def run_it(self):

        pygame.init()
        pygame.display.set_caption('URANUS')

        self.font = pygame.font.Font('freesansbold.ttf', 15)
        self.font2 = pygame.font.Font('freesansbold.ttf', 13)
        self.font3 = pygame.font.Font('freesansbold.ttf', 6)


        ttest = time.time()
        saved = 0

        print('\nURANUS is set')
        while self.run:

            self.get_events()

            self.env.fill((0, 0, 0))

            # population living
            self.seq_life(silence=False)

            # food
            pygame.draw.circle(self.env, (np.random.randint(5, 255), 150, 150), self.cake.pos, 3)

            # exit
            pygame.draw.rect(self.env, (200, 10, 10), (self.win_x - 10, self.win_y - 10, 10, 10))

            # text
            self.print_text('population: {}'.format(self.survivors), px=70, py=50, font=self.font)
            self.print_text('best species: {}'.format(self.Bios.current_species), px=90, py=70, font=self.font)
            self.print_text('best creature: {}'.format(self.best.name), px=90, py=90, font=self.font)
            self.print_text('score: {}'.format(self.best.eaten), px=70, py=110, font=self.font)
            self.print_text('generation: {}'.format(self.Bios.generation), px=self.win_x//2, py=50, font=self.font)


            # activity
            if self.show_activity:
                self.best.neural_activity()
            self.print_text(self.best.name, px=self.best.position[0], py=self.best.position[1],
                                font=self.font3, col=(10, 10, 10))

            if self.show_weights:
                self.best.weights_activity()

            if self.show_species:
                self.Bios.species_track()
                self.show_species = False

            if self.show_scores:
                self.Bios.display_score()
                self.show_scores = False

            if self.show_connectivity:
                self.best.network_connectivity()
                self.show_connectivity = False


            pygame.display.update()

        pygame.quit()
        print('\nSpAcE dead')

        if self.store:
            with open(self.filename, 'wb') as out_record:
                print('\nthe fittest is about to be buried\ngen: ', self.best.gen, '\nname: ', self.best.name)
                fossil = (True, {'DNA': self.best.DNA,
                                 'gen': self.best.gen,
                                 'name': self.best.name,
                                 'color': self.best.color,
                                 'species': self.best.species,
                                 'record': self.best.eaten})
                pickle.dump(fossil, out_record)

        if self.overwrite_hero and self.Bios.next_hero.eaten > self.Bios.hero.genetic_score:
            with open(self.heroname, 'wb') as out_record:
                print('\nthe fittest is about to be buried\ngen: ', self.Bios.next_hero.gen,
                      '\nname: ', self.Bios.next_hero.name)
                fossil = (True, {'DNA': self.Bios.next_hero.DNA,
                                 'gen': self.Bios.next_hero.gen,
                                 'name': self.Bios.next_hero.name,
                                 'color': self.Bios.next_hero.color,
                                 'species': self.Bios.next_hero.species,
                                 'record': self.Bios.next_hero.eaten})
                pickle.dump(fossil, out_record)

    def seq_life(self, silence=False):
        self.survivors = 0

        for i, creature in enumerate(self.Bios.population):

            if creature.life:

                # measures
                a, d = self.measurements(pos=creature.position, cake=self.cake)

                # run
                creature.run(angle=a, distance=d)

                # move position
                creature.position[0] += int(self.speed * (creature.action == 0) - self.speed * (creature.action == 1))
                creature.position[1] += int(self.speed * (creature.action == 2) - self.speed * (creature.action == 3)) 


                # draw
                if not silence:
                    pygame.draw.circle(self.env, creature.color, creature.position, 10)

                # check boundaries
                creature.life *= self.check_boundaries(creature.position)

                # food
                creature.food = self.check_food(creature.position)

                self.survivors += 1

        for j, creature in enumerate(self.Bios.population):
            if creature.life and not silence and self.show_names:
                self.print_text('{} - {}'.format(creature.name, creature.hunger), px=self.win_x-100,
                            py=100+j*20, font=self.font2)
            creature.life *= creature.hung_rate < creature.death
            self.scores[j] = creature.eaten
        self.best = self.Bios.population[np.argmax(self.scores).item()]  # cupy

        self.rigeneration()


    def check_boundaries(self, X):
        if X[0] > self.win_x or X[0] < 0 or X[1] < 0 or X[1] > self.win_y:
            return False
        else:
            return True


    def check_food(self, X):
        if X[0]-10 < self.cake.pos[0] < X[0]+10 and X[1]-10 < self.cake.pos[1] < X[1]+10:
            self.cake.move()
            return True
        else:
            return False


    def get_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    self.run = False

                elif event.key == pygame.K_DOWN:
                    self.genocide = True

                elif event.key == pygame.K_RIGHT:
                    self.show_activity = bool(int(self.show_activity - 1) * -1)

                elif event.key == pygame.K_3:
                    self.show_weights = bool(int(self.show_weights - 1) * -1)

                elif event.key == pygame.K_7:
                    if self.Bios.heroes_pantheon.__len__() > 0:
                        self.Bios.year_of_the_heroes = True
                        print('\nnext generation will rise from '
                              'the DNA of %s ancient Heros' % self.Bios.heroes_pantheon.__len__())
                    else:
                        print('no Heroes in the Pantheon yet')

                elif event.key == pygame.K_9:
                    self.show_species = True

                elif event.key == pygame.K_s:
                    self.show_scores = True

                elif event.key == pygame.K_c:
                    self.show_connectivity = True

                elif event.key == pygame.K_n:
                    self.show_names = bool(int(self.show_names - 1) * -1)

                elif event.key == pygame.K_LEFT:
                    self.cake.move()

                elif event.key == pygame.K_e:
                    sys.exit()


    def rigeneration(self):
        if self.survivors == 0 or self.genocide:
            print('\n---------------------------------------------------------------------------------')
            print('\n---------------------------------------------------------------------------------')

            self.Bios.fitness()
            if not self.Bios.year_of_the_heroes:

                self.Bios.new_generation()

                print('\nNEW GENERATION ', self.Bios.generation)

            else:
                self.Bios.golden_generation()
                print('\n--> ||  YEAR OF THE HEROES || <--')
                self.Bios.year_of_the_heroes = 0

            self.survivors = self.n_pop
            self.genocide = False
            self.show_activity = False
            self.show_weights = False

            plt.close()

    @staticmethod
    def measurements(pos, cake):

        # beware, it's all in cupy!

        angle = 0
        distance = round(
            np.sqrt((pos[0] - cake.pos[0]) ** 2 + (pos[1] - cake.pos[1]) ** 2))
        try:
            ang_sin = np.around(np.arcsin((pos[1] - cake.pos[1]) / distance) * (180 / np.pi))
        except ZeroDivisionError:
            ang_sin = 0

        try:
            ang_cos = np.around(np.arccos((cake.pos[0] - pos[0]) / distance) * (180 / np.pi))
        except ZeroDivisionError:
            ang_cos = 0

        if ang_sin > 0 and ang_cos < 90:
            angle = 90 - ang_sin
        elif ang_sin > 0 and ang_cos > 90:
            angle = 270 + ang_sin
        elif ang_sin < 0 and ang_cos < 90:
            angle = -ang_sin + 90
        elif ang_sin < 0 and ang_cos > 90:
            angle = ang_sin + 270

        return angle, distance


    def print_text(self, message, px, py, font, col=(200, 200, 200)):

        # message
        text1 = font.render(message, True, col)
        textRect1 = text1.get_rect()
        textRect1.center = (px, py)  # display at the center fo the screen
        self.env.blit(text1, textRect1)

        pygame.display.flip()


    def hero_life(self, creature):
        if creature.life:

            # measures
            a, d = self.measurements(pos=creature.position, cake=self.cake)

            # run
            creature.run(angle=a, distance=d)

            # move position
            creature.position[0] += int((self.speed * (creature.action == 0) - self.speed * (creature.action == 1)) \
                                    * creature.SpeedCortex.speed_magnification)
            creature.position[1] += int(self.speed * (creature.action == 2) - self.speed * (creature.action == 3) \
                                    * creature.SpeedCortex.speed_magnification)


            # draw
            pygame.draw.circle(self.env, creature.color, creature.position, 10)

            # check boundaries
            creature.life *= self.check_boundaries(creature.position)

            # food
            creature.food = self.check_food(creature.position)

        else:
            print('\nParadise has fallen')

            creature.life = True
            creature.hunger = 0
            creature.eaten = 0
            creature.position = [self.win_x//2, self.win_y//2]


    def paradise(self):

        pygame.init()
        pygame.display.set_caption('PARADISE')

        self.font = pygame.font.Font('freesansbold.ttf', 15)
        self.font2 = pygame.font.Font('freesansbold.ttf', 13)
        self.font3 = pygame.font.Font('freesansbold.ttf', 6)


        print('\nPARADISE is set')
        while self.run:

            self.get_events()

            self.env.fill((0, 0, 0))

            # population living
            self.hero_life(self.Bios.hero)

            # food
            pygame.draw.circle(self.env, (np.random.randint(5, 255), 150, 150), self.cake.pos, 3)

            # exit
            pygame.draw.rect(self.env, (200, 10, 10), (self.win_x - 10, self.win_y - 10, 10, 10))

            # text
            self.print_text('hero name: {}'.format(self.Bios.hero.name), px=80, py=50, font=self.font)
            self.print_text('best species: {}'.format(self.Bios.hero.species), px=90, py=70, font=self.font)
            self.print_text('score: {}'.format(self.Bios.hero.eaten), px=70, py=90, font=self.font)
            self.print_text('hunger: {}'.format(self.Bios.hero.hunger), px=70, py=110, font=self.font)


            # activity
            if self.show_activity:
                self.Bios.hero.neural_activity()

            if self.show_weights:
                self.Bios.hero.weights_activity()
                self.show_weights = False

            if self.show_connectivity:
                self.Bios.hero.network_connectivity()
                self.show_connectivity = False

            pygame.display.update()

        pygame.quit()
        print('\nSpAcE dead')



''' main simulation '''


if __name__ == "__main__":
    Uranus = Nature(store=False, load=False, overwrite_hero=False, win_x=500, win_y=500, speed=8, n_pop=50,
                hung_rate=0.01, radius=300, uni_w=5,
                filename='evoCool303')
    Uranus.run_it()

