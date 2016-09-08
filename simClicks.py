from pyclick.search_session.SearchResult import SearchResult
from pyclick.search_session.SearchSession import SearchSession
import numpy as np
from operator import add
import pickle
import scipy.stats as st
from scipy.stats import norm


def interleave_td(ranka, rankb):
    ranktd = []
    posa = []
    posb = []
    rand = np.random.choice([True, False])
    
    for idx in range(len(ranka) + len(rankb)):
        a = len(posa)
        b = len(posb)
        if a < b or (a == b and rand):
            ranktd.append(ranka[a])
            posa.append(a+b)
        else:
            ranktd.append(rankb[b])
            posb.append(a+b)
    return ranktd, posa, posb

def credit(clicks, posa, posb):  # take clicks to be n-hot encoding
    creda = 0
    credb = 0
    for idx in range(len(clicks)):
        if clicks[idx] == 1 and idx in posa:
            creda += 1
        if clicks[idx] == 1 and idx in posb:
            credb += 1
    return creda, credb


def gen_clicks(probs):
    rands = np.random.uniform(size=len(probs))
    return [np.ceil(probs[k] - rands[k]) for k in range(len(probs))]  # 1 if prob > rand, 0 otherwise



def read_click_data(filename):
    clicks_list = []
    fin = open(filename, 'r')
    count = 0
    n_hot = [0]*10
    url_ids = [0]*10
    for l in fin:
        line = l.strip().split('\t')
        
        if line[2] == 'Q': # handle query
            clicks_list.append(n_hot)  # appends one empty at the start that is later deleted...
            n_hot = [0]*10
            url_ids = line[5:]
        elif line[2] == 'C': # handle click
            url_id = line[3]
            if url_id in url_ids:  # ignore the 710 clicks referring to prior queries
                n_hot[url_ids.index(url_id)] = 1  # ignore multiple clicks on same doc in a query
        else:
            print "unknown action %s" % line[2]
    return clicks_list[1:] # ...here


def probabilities(ranking, click_model):
    session = SearchSession('some_query')
    ix = 0
    for i in ranking:
        res = SearchResult('doc%d' % ix, 0)  
        if i > 2:
            res = SearchResult('doc%d' % ix, 1)  
        else:
            res = SearchResult('doc%d' % ix, 0)  
        session.web_results.append(res)
        ix += 1
    return click_model.get_conditional_click_probs(session)

def experiment(click_model, probabilities, groups, sims, group_samples):

    if group_samples > 0:  # subsample groups
        for idx in range(len(groups)):
            n_picks = min(len(groups[idx]), group_samples)  # make sure not to pick more than exist
            picks = np.random.choice(range(len(groups[idx])), size=n_picks, replace=False)  # sample random indices
            groups[idx] = [groups[idx][p] for p in picks]  # and pick corresponding pairs
    
    eval_groups = [[] for k in groups]
    for idx in range(len(groups)):
        print '%i ' % idx,
        for pair in groups[idx]:                               # pick pair
            rank, pos_e, pos_p = interleave_td(pair[0],pair[1])  # interleave
            wins_e = 0.0
            wins_p = 0.0
            for n in range(sims):  # run n simulations
                probs = probabilities(rank, click_model)
                clicks = gen_clicks(probs)
                crede, credp = credit(clicks, pos_e, pos_p)
                if crede > credp:
                    wins_e += 1
                if crede < credp:
                    wins_p += 1
            score = wins_e / (wins_e + wins_p)  # compute score
            eval_groups[idx].append(score)
    print '- done'
    return eval_groups

def min_samplesize(p1, p0=0.5, alpha=0.05, beta=0.1, continuity_correction=True):
    if p1 <= p0:
        return float('inf')
    num = st.norm.ppf(1 - alpha) * np.sqrt(p0*(1 - p0)) + \
          st.norm.ppf(1 - beta ) * np.sqrt(p1*(1 - p1))
    delta = abs(p0 - p1)
    samp = np.ceil((num/delta)**2)
    if continuity_correction:
        samp = np.ceil(samp + 1/delta)
    return samp

def compute_sample_size(p1, p0=0.5):
    z = norm.ppf(0.95) * np.sqrt(p0 * (1 - p0)) + norm.ppf(0.90) * np.sqrt(p1 * (1.0 - p1))
    if z == 0.0 or (p1 - p0) == 0.0:
        # handling infinity
        return 0
    else:
        return np.ceil((z / (p1 - p0)) ** 2)


