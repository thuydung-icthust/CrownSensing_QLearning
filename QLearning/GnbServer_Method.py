import random
from Package import Package

def get_random_prob(gnb):
    prob = [0.0 for i in range(0, gnb.total_node)]
    for i in range(0, gnb.total_node):
        new_prob = round(random.random(),1)
        prob[i] = new_prob
    return prob

def get_uniform_prob(gnb):
    new_prob = round(random.random(),1)
    prob = [new_prob for i in range(0, gnb.total_node)]
    # for i in range(0, gnb.total_node):
    #     new_prob = round(random.random(),1)
    #     prob[i] = new_prob
    return prob

def update_node_location(gnb):
    return True