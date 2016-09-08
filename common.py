import numpy as np
from operator import add
import pickle
import scipy.stats as st

def pairs():
    v = 5
    vals = range(v)
    rankings = [[a, b, c, d, e] for a in vals 
                                for b in vals
                                for c in vals 
                                for d in vals 
                                for e in vals]
    return [(k,l) for k in rankings for l in rankings]



def dcg(ranking):  # technically dcg@ranking_length ...which here equals 5
    score = 0
    for idx in range(len(ranking)):
        score += (2**ranking[idx] - 1) / np.log2(1 + (idx+1))
    return score

def rbp(ranking, theta=0.8):
    score = 0
    for idx in range(len(ranking)):
        score += ranking[idx] * theta**idx
    score *= (1 - theta)
    return score

def rel_R(ranking, gmax=4):
    return map(lambda x: (2.0**x - 1)/(2**gmax), ranking)
    
def err(ranking, rel_func=rel_R):
    score = 0.0
    p = 1.0
    ranking = rel_func(ranking)
    for idx in range(len(ranking)):        
        score += p * ranking[idx]/(idx+1)
        p *= 1 - ranking[idx]
    return score
def p5(ranking):
    score = 0.
    for idx in range(len(ranking)):
        if ranking[idx] > 3:
            score += 1
    score /= len(ranking)
    return score

def diff(measure, pair):
    return measure(pair[0]) - measure(pair[1])

def group_idx(score, thresh):
    idx = 0
    for t in thresh:
        if score < t:
            break
        idx += 1
    return idx

def make_groups(measure, pairs, ngroups=10):
    scores = [diff(measure, pair) for pair in pairs]
    pos_scores = [k for k in scores if k > 0]
    pos_pairs = [pairs[idx] for idx in range(len(scores)) if scores[idx] > 0]  # take only the pairs with positive delta
    thresh = np.linspace(min(pos_scores),max(pos_scores),num=ngroups, endpoint=False)[1:]  # compute 9 thresholds
    print thresh
    groups = [[] for k in range(ngroups)]
    for idx in range(len(pos_scores)):
        groups[group_idx(pos_scores[idx], thresh)].append(pos_pairs[idx])  # assign each pair its group
    return groups
