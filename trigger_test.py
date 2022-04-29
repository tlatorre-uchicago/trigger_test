import numpy as np
from collections import namedtuple

def generate_data(N, mc=True):
    data = np.zeros(N,dtype=[('pt',float),('ips',float),('L2_pt',float),('L2_ips',float),('L1_pt',float),('trigger',int),('cat',int)])
    data['pt'] = np.random.randn(N)*10 + 10
    data['ips'] = np.random.randn(N)*10 + 10
    data['L2_pt'] = data['pt']
    if mc:
        data['L2_ips'] = data['ips']
    else:
        data['L2_ips'] = data['ips'] + 1
    data['L1_pt'] = data['pt'] + np.random.randn(N)
    return data

L2Trigger = namedtuple('L2Trigger',['pt','ips','index'])
HLT_Mu7_IP4 = L2Trigger(7,4,1 << 0)
HLT_Mu9_IP6 = L2Trigger(9,6,1 << 1)
HLT_Mu12_IP6 = L2Trigger(12,6,1 << 2)
L1Seed = namedtuple('L1Seed',['pt','L2'])
Mu7 = L1Seed(7,[HLT_Mu7_IP4,HLT_Mu9_IP6,HLT_Mu12_IP6])
Mu8 = L1Seed(8,[HLT_Mu9_IP6,HLT_Mu12_IP6])
Mu9 = L1Seed(9,[HLT_Mu9_IP6,HLT_Mu12_IP6])
Mu10 = L1Seed(10,[HLT_Mu12_IP6])
Mu12 = L1Seed(12,[HLT_Mu12_IP6])
L1_SEEDS = [Mu7,Mu8,Mu9,Mu10,Mu12]

Category = namedtuple('Category',['name','min_L1_pt','min_pt','max_pt','min_ips','trigger','index'])
category_low = Category("Low",7,7,9,5,HLT_Mu7_IP4.index,1 << 0)
category_mid = Category("Mid",9,9,12,7,HLT_Mu9_IP6.index,1 << 1)
category_high = Category("High",12,12,100,7,HLT_Mu12_IP6.index,1 << 2)
categories = [category_low,category_mid,category_high]

def trigger_data(data):
    """
    Simulates the CMS trigger from a given set of true pts and L1 pts. `Seeds`
    should be a dictionary of the available L1 seeds where the key is the name
    of the L1 seed and the value is the fraction of time the seed is active.
    """
    for i in range(len(data)):
        for l1 in L1_SEEDS:
            if data['L1_pt'][i] > l1.pt:
                for l2 in l1.L2:
                    if data['L2_pt'][i] > l2.pt:
                        data['trigger'][i] |= l2.index
    return data

def trigger_selection(data):
    """
    This is how we apply the trigger selection into low, mid, high categories
    in our analysis. `l1s` is a list of L1 pts, `pts` is a list of the true pt,
    `seeds` is a list of the active L1 seeds, and `include_l1` determines
    whether we take into account the L1 trigger requirement or not (since we
    apply this cut to the data and MC but do not apply it when computing the
    trigger scale factor.
    """
    for i in range(len(data)):
        for cat in categories:
            if cat.min_pt < data['L2_pt'][i] < cat.max_pt and data['trigger'][i] & cat.trigger:
                data['cat'][i] |= cat.index
                break
    return data

def compute_trgsf(data, mc, pt_bins, ips_bins):
    """
    Compute the trigger scale factor as a function of true pt.
    """
    sf = {}
    for cat in categories:
        sf[cat.name] = {}
        for pt_bin, (pt_low, pt_high) in enumerate(zip(pt_bins[:-1],pt_bins[1:])):
            for ips_bin, (ips_low, ips_high) in enumerate(zip(ips_bins[:-1],ips_bins[1:])):
                l1_data = data['L1_pt'] > cat.min_L1_pt
                l1_mc = mc['L1_pt'] > cat.min_L1_pt
                l2_data = (data['trigger'] & cat.trigger) != 0
                l2_mc = (mc['trigger'] & cat.trigger) != 0
                tot_data = (data['L2_pt'] > pt_low) & (data['L2_pt'] > pt_high)
                tot_data &= (data['L2_ips'] > ips_low) & (data['L2_ips'] > ips_high)
                tot_mc = (mc['L2_pt'] > pt_low) & (mc['L2_pt'] > pt_high)
                tot_mc &= (mc['L2_ips'] > ips_low) & (mc['L2_ips'] > ips_high)
                tot_mc &= l1_mc
                tot_data &= l1_data
                if np.count_nonzero(l1_mc & l2_mc & tot_mc) == 0 or np.count_nonzero(tot_mc) == 0 or np.count_nonzero(tot_data) == 0:
                    sf[cat.name][(pt_bin,ips_bin)] = 0
                else:
                    sf[cat.name][(pt_bin,ips_bin)] = (np.count_nonzero(l1_data & l2_data & tot_data)/np.count_nonzero(tot_data))/(np.count_nonzero(l1_mc & l2_mc & tot_mc)/np.count_nonzero(tot_mc))
    return sf

def get_trgsf_weights(mc,sf,pt_bins,ips_bins,cat):
    weights = np.ones_like(mc['L2_pt'])
    for i in range(len(mc)):
        pt_bin = np.digitize(data['L2_pt'][i],pt_bins)
        ips_bin = np.digitize(data['L2_ips'][i],ips_bins)
        if (pt_bin, ips_bin) in sf[cat]:
            weights[i] = sf[cat][(pt_bin,ips_bin)]
    return weights

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("trigger simulation")
    parser.add_argument("-n",default=100,type=int,help="number of events")
    args = parser.parse_args()

    data = generate_data(args.n,mc=False)
    mc = generate_data(args.n,mc=False)
    data = trigger_data(data)
    mc = trigger_data(mc)
    data = trigger_selection(data)
    mc = trigger_selection(mc)
    pt_bins = np.linspace(7,100,100)
    ips_bins = np.linspace(0,100,100)
    trgSF = compute_trgsf(data,mc,pt_bins,ips_bins)

    data_high = data[(data['cat'] & category_high.index) != 0]
    data_mid = data[(data['cat'] & category_mid.index) != 0]
    data_low = data[(data['cat'] & category_low.index) != 0]

    mc_high = mc[(mc['cat'] & category_high.index) != 0]
    mc_mid = mc[(mc['cat'] & category_mid.index) != 0]
    mc_low = mc[(mc['cat'] & category_low.index) != 0]

    weights_high = get_trgsf_weights(mc_high,trgSF,pt_bins,ips_bins,'High')
    weights_mid = get_trgsf_weights(mc_mid,trgSF,pt_bins,ips_bins,'Mid')
    weights_low = get_trgsf_weights(mc_low,trgSF,pt_bins,ips_bins,'Low')

    plt.figure()
    plt.subplot(3,1,1)
    plt.hist(data_high['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_high['L2_pt'],bins=pt_bins,histtype='step',label='mc')
    plt.subplot(3,1,2)
    plt.hist(data_mid['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_mid['L2_pt'],bins=pt_bins,histtype='step',label='mc')
    plt.subplot(3,1,3)
    plt.hist(data_low['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_low['L2_pt'],bins=pt_bins,histtype='step',label='mc')
    plt.figure()
    plt.subplot(3,1,1)
    plt.hist(data_high['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_high['L2_pt'],weights=weights_high,bins=pt_bins,histtype='step',label='mc')
    plt.subplot(3,1,2)
    plt.hist(data_mid['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_mid['L2_pt'],weights=weights_mid,bins=pt_bins,histtype='step',label='mc')
    plt.subplot(3,1,3)
    plt.hist(data_low['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_low['L2_pt'],weights=weights_low,bins=pt_bins,histtype='step',label='mc')
    plt.title("with weights")
    plt.show()
