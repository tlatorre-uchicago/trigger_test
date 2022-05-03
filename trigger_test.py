#!/usr/bin/env python
import numpy as np
from collections import namedtuple

def generate_data(N, mc=True):
    data = np.zeros(N,dtype=[('pt',float),('ips',float),('L2_pt',float),('L2_ips',float),('L1_pt',float),('trigger',int),('cat',int),('prescale',int)])
    np.random.seed(0)
    data['pt'] = np.random.rand(N)*10 + 7
    data['ips'] = np.random.rand(N)*10 + 2
    data['L2_pt'] = data['pt']
    if mc:
        data['L2_ips'] = data['ips']
    else:
        data['L2_ips'] = data['ips'] - 2
    data['L1_pt'] = data['pt'] + np.random.randn(N)*2
    return data

L2Trigger = namedtuple('L2Trigger',['pt','ips','index'])
HLT_Mu7_IP4 = L2Trigger(7,4,1 << 0)
HLT_Mu9_IP6 = L2Trigger(9,6,1 << 1)
HLT_Mu12_IP6 = L2Trigger(12,6,1 << 2)
L1Seed = namedtuple('L1Seed',['pt','L2','time_on'])
Mu7 = L1Seed(7,[HLT_Mu7_IP4,HLT_Mu9_IP6,HLT_Mu12_IP6],0.1)
Mu8 = L1Seed(8,[HLT_Mu7_IP4,HLT_Mu9_IP6,HLT_Mu12_IP6],0.2)
Mu9 = L1Seed(9,[HLT_Mu9_IP6,HLT_Mu12_IP6],0.3)
Mu10 = L1Seed(10,[HLT_Mu12_IP6],0.8)
Mu12 = L1Seed(12,[HLT_Mu12_IP6],1.0)
L1_SEEDS = [Mu7,Mu8,Mu9,Mu10,Mu12]

Category = namedtuple('Category',['name','min_L1_pt','min_pt','max_pt','min_ips','trigger','index'])
category_low = Category("Low",7,7,9,5,HLT_Mu7_IP4.index,1 << 0)
category_mid = Category("Mid",9,9,12,7,HLT_Mu9_IP6.index,1 << 1)
category_high = Category("High",12,12,100,7,HLT_Mu12_IP6.index,1 << 2)
categories = [category_low,category_mid,category_high]

USE_REAL_IPS = True

def trigger_data(data, mc=False, calibration=False):
    """
    Simulates the CMS triggers.

    Parameters
    ==========

    data: np.array
        Data generated from generate_data() and trigger_data()
    mc: bool
        Is this MC?
    calibration: bool
        Is this the J/psi calibration?
    """
    data = data.copy()
    for i in range(len(data)):
        for l1 in L1_SEEDS:
            if not mc and np.random.rand() > l1.time_on:
                continue
            for l2 in l1.L2:
                data['prescale'][i] |= l2.index
            if data['L1_pt'][i] > l1.pt:
                for l2 in l1.L2:
                    if data['L2_pt'][i] > l2.pt and data['L2_ips'][i] > l2.ips:
                        data['trigger'][i] |= l2.index
    if calibration:
        return data
    return data[data['trigger'] != 0]

def trigger_selection(data, use_real_ips=USE_REAL_IPS):
    """
    This is how we apply the trigger selection into low, mid, high categories
    in our analysis.

    Parameters
    ==========

    data: np.array
        Data generated from generate_data() and trigger_data()
    use_real_ips: bool
        Use the true IPS instead of the L2 IPS. When True, this will be
        equivalent to when we correct for the muon impact parameter error.
    """
    for i in range(len(data)):
        for cat in categories:
            if not use_real_ips:
                if cat.min_pt < data['L2_pt'][i] < cat.max_pt and data['trigger'][i] & cat.trigger and data['L2_ips'][i] > cat.min_ips:
                    data['cat'][i] |= cat.index
                    break
            else:
                if cat.min_pt < data['L2_pt'][i] < cat.max_pt and data['trigger'][i] & cat.trigger and data['ips'][i] > cat.min_ips:
                    data['cat'][i] |= cat.index
                    break
    return data

def compute_trgsf(data_, mc_, pt_bins, ips_bins, use_real_ips=USE_REAL_IPS, include_l1=True, use_prescale=True):
    """
    Compute the trigger scale factor as a function of L2 pt and ips.

    Parameters
    ==========

    data: np.array
        Data generated from generate_data() and trigger_data(calibration=True)
    mc: np.array
        MC generated from generate_data() and trigger_data(calibration=True)
    pt_bins: np.array
        Transverse momentum bins
    ips_bins: np.array
        Impact parameter significance bin edges
    use_real_ips: bool
        Use the true impact parameter significance when binning the events.
    include_l1: bool
        Apply an L1 pT cut of 7, 9, and 12 for the low, mid, and high categories
    """
    sf = {}
    for cat in categories:
        sf[cat.name] = {}
        if use_prescale:
            data = data_[(data_['prescale'] & cat.trigger) != 0]
            mc = mc_[(mc_['prescale'] & cat.trigger) != 0]
        for pt_bin, (pt_low, pt_high) in enumerate(zip(pt_bins[:-1],pt_bins[1:])):
            for ips_bin, (ips_low, ips_high) in enumerate(zip(ips_bins[:-1],ips_bins[1:])):
                l1_data = data['L1_pt'] > cat.min_L1_pt
                l1_mc = mc['L1_pt'] > cat.min_L1_pt
                l2_data = (data['trigger'] & cat.trigger) != 0
                l2_mc = (mc['trigger'] & cat.trigger) != 0
                tot_data = (data['L2_pt'] > pt_low) & (data['L2_pt'] < pt_high)
                tot_mc = (mc['L2_pt'] > pt_low) & (mc['L2_pt'] < pt_high)
                if not use_real_ips:
                    tot_data &= (data['L2_ips'] > ips_low) & (data['L2_ips'] < ips_high)
                    tot_mc &= (mc['L2_ips'] > ips_low) & (mc['L2_ips'] < ips_high)
                else:
                    tot_data &= (data['ips'] > ips_low) & (data['ips'] < ips_high)
                    tot_mc &= (mc['ips'] > ips_low) & (mc['ips'] < ips_high)
                if include_l1:
                    l2_mc &= l1_mc
                    l2_data &= l1_data
                if np.count_nonzero(l2_mc & tot_mc) == 0 or np.count_nonzero(tot_mc) == 0 or np.count_nonzero(tot_data) == 0:
                    sf[cat.name][(pt_bin,ips_bin)] = 1
                else:
                    sf[cat.name][(pt_bin,ips_bin)] = (np.count_nonzero(l2_data & tot_data)/np.count_nonzero(tot_data))/(np.count_nonzero(l2_mc & tot_mc)/np.count_nonzero(tot_mc))
    return sf

def get_trgsf_weights(mc,sf,pt_bins,ips_bins,cat, use_real_ips=USE_REAL_IPS):
    """
    Returns the trigger scale factor weights for MC events.
    """
    weights = np.ones_like(mc['L2_pt'])
    for i in range(len(mc)):
        pt_bin = np.digitize(mc['L2_pt'][i],pt_bins)-1
        if not use_real_ips:
            ips_bin = np.digitize(mc['L2_ips'][i],ips_bins)-1
        else:
            ips_bin = np.digitize(mc['ips'][i],ips_bins)-1
        if (pt_bin, ips_bin) in sf[cat]:
            weights[i] = sf[cat][(pt_bin,ips_bin)]
        else:
            print("skipping")
    return weights

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("trigger simulation")
    parser.add_argument("-n",default=100,type=int,help="number of events")
    parser.add_argument("--disable-l1",default=False,action='store_true',help="include l1 pt cut in category definition")
    parser.add_argument("--use-real-ips",default=False,action='store_true',help="use real ips when computing trigger scale factors")
    parser.add_argument("--disable-prescale",default=False,action='store_true',help="disable prescale")
    args = parser.parse_args()

    data = generate_data(args.n,mc=False)
    mc = generate_data(args.n,mc=True)
    data_calibration = trigger_data(data,calibration=True)
    data = trigger_data(data)
    mc_calibration = trigger_data(mc,mc=True,calibration=True)
    mc = trigger_data(mc,mc=True)
    luminosity = {cat.name: 1 for cat in categories}
    if not args.disable_prescale:
        for cat in categories:
            luminosity[cat.name] = np.count_nonzero((data_calibration['prescale'] & cat.trigger) != 0)/len(data_calibration)
    print("luminosity = ", luminosity)
    data = trigger_selection(data,use_real_ips=args.use_real_ips)
    mc = trigger_selection(mc,use_real_ips=args.use_real_ips)
    pt_bins = np.linspace(7,17,17-7+1)
    pt_bins_low = np.linspace(7,9,10)
    pt_bins_mid = np.linspace(9,12,10)
    ips_bins = np.linspace(2,12,11)
    trgSF = compute_trgsf(data_calibration,mc_calibration,pt_bins,ips_bins,use_real_ips=args.use_real_ips,include_l1=not args.disable_l1)

    data_high = data[(data['cat'] & category_high.index) != 0]
    data_mid = data[(data['cat'] & category_mid.index) != 0]
    data_low = data[(data['cat'] & category_low.index) != 0]

    mc_high = mc[(mc['cat'] & category_high.index) != 0]
    mc_mid = mc[(mc['cat'] & category_mid.index) != 0]
    mc_low = mc[(mc['cat'] & category_low.index) != 0]

    weights_high = get_trgsf_weights(mc_high,trgSF,pt_bins,ips_bins,'High',use_real_ips=args.use_real_ips)*luminosity['High']
    weights_mid = get_trgsf_weights(mc_mid,trgSF,pt_bins,ips_bins,'Mid',use_real_ips=args.use_real_ips)*luminosity['Mid']
    weights_low = get_trgsf_weights(mc_low,trgSF,pt_bins,ips_bins,'Low',use_real_ips=args.use_real_ips)*luminosity['Low']

    plt.figure()
    plt.subplot(3,2,1)
    plt.hist(data_high['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_high['L2_pt'],bins=pt_bins,histtype='step',label='mc')
    plt.title("Without Weights")
    plt.subplot(3,2,3)
    plt.hist(data_mid['L2_pt'],bins=pt_bins_mid,histtype='step',label='data')
    plt.hist(mc_mid['L2_pt'],bins=pt_bins_mid,histtype='step',label='mc')
    plt.subplot(3,2,5)
    plt.hist(data_low['L2_pt'],bins=pt_bins_low,histtype='step',label='data')
    plt.hist(mc_low['L2_pt'],bins=pt_bins_low,histtype='step',label='mc')
    plt.xlabel(r"p$^\mathrm{T}$ (GeV)")
    plt.subplot(3,2,2)
    plt.hist(data_high['L2_pt'],bins=pt_bins,histtype='step',label='data')
    plt.hist(mc_high['L2_pt'],weights=weights_high,bins=pt_bins,histtype='step',label='mc')
    plt.title("With Weights")
    plt.subplot(3,2,4)
    plt.hist(data_mid['L2_pt'],bins=pt_bins_mid,histtype='step',label='data')
    plt.hist(mc_mid['L2_pt'],weights=weights_mid,bins=pt_bins_mid,histtype='step',label='mc')
    plt.subplot(3,2,6)
    plt.hist(data_low['L2_pt'],bins=pt_bins_low,histtype='step',label='data')
    plt.hist(mc_low['L2_pt'],weights=weights_low,bins=pt_bins_low,histtype='step',label='mc')
    plt.xlabel(r"p$^\mathrm{T}$ (GeV)")
    plt.show()
