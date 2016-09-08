import common as cm
import simClicks as sim
from pyclick.click_models.UBM import UBM
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser
from pyclick.utils.Utils import Utils
import numpy as np
from operator import add
import pickle
import scipy.stats as st
import matplotlib.pyplot as mp


click_model = UBM()
search_sessions_path = "YandexRelPredChallenge.txt"
search_sessions_num = 10000

search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, search_sessions_num)

train_test_split = int(len(search_sessions) * 0.75)
train_sessions = search_sessions[:train_test_split]
train_queries = Utils.get_unique_queries(train_sessions)

test_sessions = Utils.filter_sessions(search_sessions[train_test_split:], train_queries)
test_queries = Utils.get_unique_queries(test_sessions)
click_model.train(train_sessions)




measures = {'dcg':cm.dcg, 'rbp':cm.rbp, 'err':cm.err}
ep_pairs = cm.pairs()
for m in measures:
    print "making groups"
    groups = cm.make_groups(measures[m], ep_pairs)
    
    print 'ubm with %s' % m
    ubm_data = sim.experiment(click_model, sim.probabilities, groups, sims=50, group_samples=1000)
    
    ubm_out = open('ubm_%s_data.pkl' % m, 'w')
    pickle.dump(ubm_data, ubm_out)
    ubm_out.close()


fin = open('ubm_dcg_data.pkl', 'r')
ubm_dcg_data = pickle.load(fin)
fin.close()

fin = open('ubm_err_data.pkl', 'r')
ubm_err_data = pickle.load(fin)
fin.close()

fin = open('ubm_rbp_data.pkl', 'r')
ubm_rbp_data = pickle.load(fin)
fin.close()

res_dict_ubm = {'ubm_dcg':ubm_dcg_data, 'ubm_err':ubm_err_data, 'ubm_rbp':ubm_rbp_data}

for res in res_dict_ubm:
    print res
    print '  min  | 5.qant | median |95.quant|   max  | median N  '
    for group in res_dict_ubm[res]:
                
        print '%6.3f | %6.3f | %6.3f | %6.3f | %6.3f' % \
            (min(group), np.percentile(group, 5),np.median(group), np.percentile(group, 95), max(group)),
        print '| %7g' % (sim.compute_sample_size(np.median(group)))
    print ''
    

final_res_ubm = {}
err_ubm = {}
for res in res_dict_ubm:
    final_res_ubm[res] = []
    err_ubm[res] = []
    for group in res_dict_ubm[res]:
        temp = [sim.compute_sample_size(x) for x in group]
        final_res_ubm[res].append(np.mean(temp))
        err_ubm[res].append(np.std(temp)/5)

x = [i / 10. for i in range(0,10)]
#mp.plot(x,final_res["sdcm_dcg"])
mp.errorbar(x, final_res_ubm["ubm_dcg"], yerr=err_ubm["ubm_dcg"])



