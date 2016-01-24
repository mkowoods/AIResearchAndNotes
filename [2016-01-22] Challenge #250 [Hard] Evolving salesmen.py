import random
from math import floor
#random.seed(42)

CITIES_SAMPLE = """0   0
689 291
801 724
388 143
143 832
485 484
627 231
610 311
549 990
220  28
66 496
693 988
597 372
753 222
885 639
897 594
482 635
379 490
923 781
352 867
834 713
133 344
835 949
667 695
956 850
535 170
583 406"""

CITIES_BONUS = """0   0
194 956
908 906
585 148
666 196
 76  59
633 672
963 801
789 752
117 620
409  65
257 747
229 377
334 608
837 374
382 841
921 910
 54 903
959 743
532 477
934 794
720 973
117 555
519 496
933 152
408  52
750   3
465 174
790 890
983 861
605 790
314 430
272 149
902 674
340 780
827 507
915 187
483 931
466 503
451 435
698 569
"""
cities = CITIES_SAMPLE 
cities_pts = []
for city in cities.split("\n"):
    if city:
        x,y = city.split()
        x,y = float(x), float(y)
        cities_pts.append((x, y))
print cities_pts


def generate_random_route(num_cities = len(cities_pts)):
    tmp = range(len(cities_pts))
    random.shuffle(tmp)
    tmp.append(tmp[0])
    return tmp

def pythag_dist(pt1, pt2):
    return floor(( (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )**0.5)

def pythag_dist_from_idx(idx1, idx2):
    pt1, pt2 = cities_pts[idx1], cities_pts[idx2]
    return pythag_dist(pt1, pt2)


POPULATION = 100
MUTATION_PROBABILITY = 0.75
SURVIVAL_RATE = 0.50
BEST_CANDIDATES = 0.1 #Top Candidates added to new population
LEN_ROUTE = len(cities_pts) + 1
DIST_MATRIX = {(from_idx, to_idx) : pythag_dist_from_idx(from_idx, to_idx) 
               for from_idx in range(len(cities_pts)) 
               for to_idx in range(len(cities_pts))}

def score(route):
    return sum([ DIST_MATRIX[(route[i], route[i + 1])] for i in range(len(route) - 1) ])

get_route_score = lambda rt : (score(rt), rt)

pop = [ get_route_score( generate_random_route() ) for _ in range(POPULATION) ]

def evolve(population):
    population.sort()
    #automatically get passed to the next generation
    top10_perc = population[:int(BEST_CANDIDATES * len(population))]
    survivors = population[:int(SURVIVAL_RATE * len(population))]
    
    
    min_score = survivors[0][0]
    max_score = survivors[-1][0]
    #print min_score, max_score
    score_range = max_score - min_score
    
    new_population = []
    
    for _ in range(POPULATION):
        parent1 = random.choice( survivors )
        parent2 = random.choice( survivors )
        parent1_succ_dict = {parent1[1][i] : parent1[1][i+1] for i in range(LEN_ROUTE -1)}
        parent2_succ_dict = {parent2[1][i] : parent2[1][i+1] for i in range(LEN_ROUTE -1)}
        
        prob_parent1 = 1.0 - parent1[0]/(parent1[0] + parent2[0])        
        
        possible_keys = set(parent1[1])
    
        child = []
        
        for gene1, gene2 in zip(parent1[1][:-1], parent2[1][:-1]):
            
            if child == []: #means there is no gene
                if random.random() < prob_parent1:
                    child.append(gene1)
                    possible_keys.remove(gene1)
                else:
                    child.append(gene2)
                    possible_keys.remove(gene2)
            else:
                prior_gene = child[-1]
                succesor_city1 = parent1_succ_dict[prior_gene]
                succesor_city2 = parent2_succ_dict[prior_gene]
                
                #original algorithm based on random choice preferencing higher scoring parent
                #if random.random() < prob_parent1 and succesor_city1 in possible_keys:
                    #child.append(succesor_city1)
                    #possible_keys.remove(succesor_city1)
                #elif succesor_city2 in possible_keys:
                    #child.append(succesor_city2)
                    #possible_keys.remove(succesor_city2)
                #else:
                    #rand_city = random.choice( list(possible_keys) )
                    #child.append(rand_city)
                    #possible_keys.remove(rand_city)
                
                #greedy(locally optimal) solution that chooses shortest next hop
                
                dist_city1 = DIST_MATRIX[(prior_gene, succesor_city1)]
                dist_city2 = DIST_MATRIX[(prior_gene, succesor_city2)]
                
                if dist_city1 < dist_city2  and succesor_city1 in possible_keys:
                    child.append(succesor_city1)
                    possible_keys.remove(succesor_city1)   
                elif succesor_city2 in possible_keys:
                    child.append(succesor_city2)
                    possible_keys.remove(succesor_city2)
                else:
                    score, rand_city = min([(DIST_MATRIX[(prior_gene, k)], k) for k in possible_keys])
                    child.append(rand_city)
                    possible_keys.remove(rand_city)                
                    
                
                    
        if random.random() < MUTATION_PROBABILITY:
            allele1, allele2 = random.randint(0, 26), random.randint(0, 26)
            child[allele1], child[allele2] = child[allele2], child[allele1]
            
        child.append(child[0])
        new_population.append(get_route_score(child))
    new_population = new_population + top10_perc
    return new_population
                    
                
                
if __name__ == "__main__":
    #lowest observed score 4022.0
    import time
    start = time.time()
    print 'Init Min Pop',  min(pop)
    
    for _ in range(125):
        pop =  evolve(pop)
        print min(pop)[0], max(pop)[0]
    print min(pop)    
    print time.time() - start
