# NeuroEvolution
Can a simple food-reaching behavior emerge from an unstructured LIF cells reservoir endowed with sensory and motor neurons? 

## Connectivity

        - Sight neurons -->  main pool (<- recurrent) --> premotor neurons -> motor neuron (4 actions)
        - Speed neurons -->  main pool
        - Hunger neurons --> main pool

![evolution_connectivity](https://user-images.githubusercontent.com/70176926/189867206-20418f13-985d-4570-bc37-492dc94efb0d.png)
       
## Evolution

Evolution through natural selection:

creation of an initial population with random connectivity weights
- selection of the fittest individual (has eaten more)
- definition of its DNA as the connectivity matrix of its brain networks (and weights values)
- creation of a new population with individual endowed with:

        - same DNA as the fittest
        
        - mutated DNA
        
        - brand new DNA
        
- repeat


Disclaimer: sorry for if the code looks awfully messy and informal, I will take care of cleaning it soon. This project started and will continue to be an hobby. It comes straight out of my laptop repository 

## Run-time user actions

the following keywords are possible action to be done at runtime:

- K_DOWN: kill all the creatures

- K_RIGHT: show the neural activity of the fittest

- K_LEFT: randomly change the position of the food

- K_3: show the weights as an image

- K_7: the next generation will be a hero-only run (hero: particulary talented creatures appeared in the past)

- K_9: show the species

- K_s: show the scores

- K_c: show the neural connectivity 

- K_n: show the names of the creatures

- K_e: exit

- K_p: pruning, kill the lower half of the population based on score
