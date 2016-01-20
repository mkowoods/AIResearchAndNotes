
import random


#Genetic Algorithm Hello World!

TARGET_STR = 'Hello, world!'
HELLO_WORLD_LEN = len(TARGET_STR)
POPULATION_SIZE = 100
SURVIVAL_RATE = 0.6 #percent of best candidates that are moved to the next iteration
MUTATION_RATE = 0.1 #probability of breeding

def create_random_string(str_len, min_char = 32, max_char = 127):
    return ''.join([chr(random.randint(min_char, max_char)) for _ in range(str_len)])

def hamming_distance(string, target = TARGET_STR):
    #num positions where the instances differ
    return sum([float(c1 != c2) for c1, c2 in zip(string, target)])

def get_new_candidate():
    rand_str = create_random_string(HELLO_WORLD_LEN)
    score = hamming_distance(rand_str)
    return score, rand_str

#create initial population
POPULATION = [get_new_candidate() for _ in range(POPULATION_SIZE)]

def run_generation(pop = []):
    
    pop.sort()
    best = pop[0]  # best 10 candidates
    print best
    
    num_survivors = int(SURVIVAL_RATE*len(pop))
    survivor_pop = pop[:num_survivors]
    resampled_survivors = survivor_pop[:]
    for survivor in survivor_pop:
        for _ in range(10):
            #resamples survivors from the survivor pop based on their score with lower score, showing up more frequently 
            if random.random() > survivor[0]/HELLO_WORLD_LEN:
                resampled_survivors.append(survivor)
        
    
    new_pop = []
    
    
    for _ in range(POPULATION_SIZE):
        
        #cross_breed
        parent1 = random.choice(resampled_survivors)
        parent2 = random.choice(resampled_survivors)
        
        prob_parent1 = 1.0 - parent1[0]/(parent1[0] + parent2[0]) #inveresely proportional to the score
        
        child = ''
        
        for c1, c2 in zip(parent1[1], parent2[1]):
            if random.random() < MUTATION_RATE:
                child += create_random_string(HELLO_WORLD_LEN)[0]
            else:    
                if random.random() < prob_parent1:
                    child += c1
                else:
                    child += c2
        
        new_pop.append((hamming_distance(child), child))
        
    return new_pop
            
        
for i in range(200):
    print i
    POPULATION.sort()
    if POPULATION[0][0] < 0.0001:
        print POPULATION[0]
        break
    POPULATION = run_generation(POPULATION)
    








