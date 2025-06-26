import pandas as pd
import numpy as np
from hmmlearn import hmm
import random
from sklearn.neighbors import KernelDensity
from itertools import groupby
import scipy.stats as stats
import sys
import getopt
import json
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_result_data(file):
    df = pd.read_table(file,
                       usecols = ['SNP Name','Chr','Position','Log R Ratio','B Allele Freq',
                                  'Adjusted LRR','Transformed BAF','LRR_proba(0x,1x,2x,3x,4x)',
                                  'BAF_proba(1x,2x,3x)','CN call','grouped CN call'],
                       dtype={'SNP Name':str,'Chr':str,'Position':int,'Log R Ratio':float,'B Allele Freq':float,
                              'Adjusted LRR':float,'Transformed BAF':float,'LRR_proba(0x,1x,2x,3x,4x)':str,'BAF_proba(1x,2x,3x)':str,
                              'CN call':str,'grouped CN call':str})
    df['LRR_proba(0x,1x,2x,3x,4x)'] = df['LRR_proba(0x,1x,2x,3x,4x)'].apply(lambda x: np.array([float(p) for p in x.split(',')]))
    df['BAF_proba(1x,2x,3x)'] = df['BAF_proba(1x,2x,3x)'].apply(lambda x: np.array([float(p) for p in x.split(',')]))
    return(df)

def read_header(file):
    f = open(file,'r')
    info_dict = {'Input_file':None,'Microarray_type':None,'Original_ISCN':None,'Sex_chromosomes':None,'Chromosome_summary':None}
    chrom_sum_dict = dict()
    chrom_sum_titles = ['## ' + str(i) + ': ' for i in range(1,23)]
    n = 0
    while True:
        l = f.readline()
        if not l.startswith('#'):
            break
        if '=' in l:
            l = l.strip('#').strip().split('=')
            if l[0] in ['Input_file','Microarray_type','Original_ISCN','Sex_chromosomes']:
                info_dict.update({l[0]:l[1]})
        if any([c in l for c in chrom_sum_titles]):
            chrom = l.split(':')[-2].strip('# ')
            cn = l.split(':')[-1].strip()
            chrom_sum_dict.update({chrom:cn})
        n = n + 1
    info_dict.update({'Chromosome_summary':chrom_sum_dict,'nheader' : n})
    return(info_dict)

def read_info_json(file):
    with open(file,'r') as f:
        info_dict = json.load(f)
    model_names = info_dict['model_names']
    model_params = dict()
    for m in set(model_names.values()):
        model_params.update({m:info_dict[m]})
    filter_thresh = info_dict['filter_thresholds']
    return([model_names,model_params,filter_thresh])

def make_model_from_dict(model_params_dict,model_name):
    models_dict,labels_dict = dict(),dict()
    model_params = model_params_dict[model_name]
    lrr_params, baf_params = reorder_states(model_params['lrr_model']),reorder_states(model_params['baf_model'])
    lrr_model,baf_model = make_model(lrr_params),make_model(baf_params)
    models_dict.update({'lrr_model':lrr_model,'baf_model':baf_model})
    lrr_labels = {int(k):v for k,v in lrr_params['Labels'].items()}
    baf_labels = {int(k):v for k,v in baf_params['Labels'].items()}
    labels_dict.update({'lrr_model':lrr_labels,'baf_model':baf_labels})
    return(models_dict,labels_dict)

def count_Y(xy_str):
    counts = []
    for xy in xy_str.split('/'):
        nY = xy.count('Y')
        counts.append(nY)
    return(list(set(counts)))

def reorder_states(model_params):
    n = model_params['Model components']
    model_labels = {int(k):v for k,v in model_params['Labels'].items()}
    new_means,new_covars, new_startprob = [0] * n, [0] * n, [0] * n
    new_transmat = np.zeros((n,n))
    sorted_labels = [*sorted(model_labels.values())]
    new_labels = {str(i):v for i,v in enumerate(sorted_labels)}
    conversion_dict = {k:sorted_labels.index(v) for k,v in model_labels.items()}
    for k,v in conversion_dict.items():
        new_means[v] = model_params['means'][k]
        new_covars[v] = model_params['covars'][k]
        new_startprob[v] = model_params['startprob'][k]
        for k1,v1 in conversion_dict.items():
            new_transmat[v][v1] = model_params['transmat'][k][k1]
    new_params = model_params.copy()
    new_params.update({'Labels':new_labels,
                       'means':new_means,
                       'covars':new_covars,
                       'startprob':new_startprob,
                       'transmat':new_transmat.tolist()})
    return(new_params)

def make_model(model_params_dict):
    cov_type = model_params_dict['Covariance type']
    model = hmm.GaussianHMM(n_components=model_params_dict['Model components'],covariance_type=cov_type)
    model.n_features = 1
    if cov_type == 'spherical':
        cov_shape = (model.n_components)
    elif cov_type == 'diag':
        cov_shape = (model.n_components,model.n_features)
    elif cov_type == 'full':
        cov_shape = (model.n_components,model.n_features,model.n_features)
    else:
        cov_shape = (model.n_features,model.n_features)
    try:
        model.covars_ = np.array(model_params_dict['covars']).reshape(cov_shape)
    except ValueError as e:
        print(f'Cannot reshape covariance matrix (shape {np.array(model_params_dict["covars"]).shape}) according to covariance type ({cov_type} : {cov_shape})')
        raise ValueError(e)
    model.means_ = np.array(model_params_dict['means'])
    model.startprob_ = np.array(model_params_dict['startprob'])
    model.transmat_ = np.array(model_params_dict['transmat'])
    return(model)

def sample_distributions(model,labels, bounded = False, boundary = (0.5,1), n = 100000, seed = 1):
    means = np.array(model.means_).reshape(-1)
    stds = np.sqrt(np.array(model.covars_).reshape(-1))
    sample_dict = dict()
    for i,state in labels.items():
        rng = np.random.default_rng(seed = seed)
        r = rng.normal(means[i],stds[i],n)
        if bounded:
            lower, upper = min(boundary),max(boundary)
            r = np.abs(r - lower) + lower
            r = upper - np.abs(r - upper)
        sample_dict.update({state:r})
    return(sample_dict)

def read_table(file, skiprows = 0, keep_file_name = False):
    df = pd.read_table(file, skiprows=skiprows,sep='\t')
    df['Chr'] = df['Chr'].apply(str)
    ps = df['p_values(0x,1x,2x,3x,4x)'].values
    ps = [p.split(',') for p in ps]
    df['p(0x)'] = [float(p[0]) for p in ps]
    df['p(1x)'] = [float(p[1]) for p in ps]
    df['p(2x)'] = [float(p[2]) for p in ps]
    df['p(3x)'] = [float(p[3]) for p in ps]
    df['p(4x)'] = [float(p[4]) for p in ps]
    df.drop('p_values(0x,1x,2x,3x,4x)', axis = 1, inplace = True)
    df['section_CN'] = df[['p(0x)','p(1x)','p(2x)','p(3x)','p(4x)']].apply(lambda x: max_p(x.values), axis = 1)
    if keep_file_name:
        df['file'] = file
    return(df)

def max_p(ps):
    i = list(ps).index(max(ps))
    return(str(i) + 'x')

def find_anchors(summary_df,header_dict,data_df,model_ref_dict,model_means):
    chrom_cn = header_dict['Chromosome_summary']
    out_df_list = []
    for c,cn in chrom_cn.items():
        print(c)
        c_df = summary_df[summary_df['Chr'] == c].copy()
        c_df['lor'] = c_df.apply(lambda x: log_or(x['p(0x)'],x['p(1x)'],x['p(2x)'],x['p(3x)'],x['p(4x)']), axis = 1)
        c_df['lor_cn'] = c_df[['lor','section_CN']].apply(lambda x: lor_call(x.iloc[0],x.iloc[1]), axis = 1)
        c_df['BAF_class'] = summary_classify_baf(c_df,data_df)
        c_df = collapse_summary_df(c_df,data_df,model_ref_dict)
        c_df['lor'] = c_df.apply(lambda x: log_or(x['p(0x)'],x['p(1x)'],x['p(2x)'],x['p(3x)'],x['p(4x)']), axis = 1)
        c_df['lor_cn'] = c_df[['lor','section_CN']].apply(lambda x: lor_call(x.iloc[0],x.iloc[1]), axis = 1)
        c_df['BAF_class'] = summary_classify_baf(c_df,data_df)
        anchors = c_df.apply(lambda x: (x['section_CN'] == x['BAF_class']),axis = 1)
        c_df['fixed'] = anchors
        out_df_list.append(c_df)
    print('X')
    XY_str = header_dict['Sex_chromosomes']
    X_cn = count_X(XY_str)
    X_cn = float(X_cn.strip('x'))
    if X_cn != 1:
        expected_X_cn = 2
    else:
        expected_X_cn = 1
    c_df = summary_df[summary_df['Chr'] == 'X'].copy()
    c_df['lor'] = c_df.apply(lambda x: log_or(x['p(0x)'],x['p(1x)'],x['p(2x)'],x['p(3x)'],x['p(4x)'],h_null=expected_X_cn), axis = 1)
    c_df['lor_cn'] = c_df[['lor','section_CN']].apply(lambda x: lor_call(x.iloc[0],x.iloc[1]), axis = 1)
    c_df['BAF_class'] = summary_classify_baf(c_df,data_df)
    c_df = collapse_summary_df(c_df,data_df,model_ref_dict)
    c_df['lor'] = c_df.apply(lambda x: log_or(x['p(0x)'],x['p(1x)'],x['p(2x)'],x['p(3x)'],x['p(4x)'],h_null=expected_X_cn), axis = 1)
    c_df['lor_cn'] = c_df[['lor','section_CN']].apply(lambda x: lor_call(x.iloc[0],x.iloc[1]), axis = 1)
    c_df['BAF_class'] = summary_classify_baf(c_df,data_df)
    anchors = c_df.apply(lambda x: (x['section_CN'] == x['BAF_class']),axis = 1)
    c_df['fixed'] = anchors
    out_df_list.append(c_df)
    out_df = pd.concat(out_df_list,ignore_index=True)
    out_df.drop('lor_cn', axis = 1, inplace = True)
    return(out_df)

def log_or(p0,p1,p2,p3,p4,h_null = 2):
    p_dict = dict(enumerate([p0,p1,p2,p3,p4]))
    out_dict = dict()
    for i in range(5):
        if i < h_null:
            lor_current = np.log10(p_dict[i] / p_dict[i+1])
        elif i == h_null:
            if i == 0:
                lor_current = np.log10(p_dict[i] / p_dict[i+1])
            elif i == 4:
                lor_current = np.log10(p_dict[i] / p_dict[i-1])
            else:
                lor_current = min(np.log10(p_dict[i] / p_dict[i+1]),np.log10(p_dict[i] / p_dict[i-1]))
        else:
            lor_current = np.log10(p_dict[i] / p_dict[i-1])
        out_dict.update({i:lor_current})
    return(out_dict)

def lor_call(lor_dict,cn,expected_cn='2x',thresh = 6):
    max_lor = max(lor_dict.values())
    max_likely_cn = [k for k,v in lor_dict.items() if v == max(lor_dict.values())][0]
    max_likely_cn = f'{max_likely_cn}x'
    if max_lor < 6 or max_likely_cn == expected_cn:
        return(expected_cn)
    else:
        return(max_likely_cn)

def find_baf_modes(baf_values,sampling = True, sampling_size = 400, log_density_threshold = 0, iterations = 1, plot_scores = False, plot_label = None):
    # Mirrored at 0.5
    baf_values = list(abs(0.5 - np.array(baf_values)) + 0.5)
    aggregate_scores =  np.repeat(0,250)
    for i in range(iterations):
        if sampling:
            baf_samples = random.choices(list(baf_values),k = sampling_size)
        else:
            baf_samples = baf_values
        if np.array(baf_samples).ndim != 2:
            try:
                baf_samples =  np.array(baf_samples).reshape(len(baf_samples),1)
            except ValueError as e:
                print(f'Failed to reshape data to ({len(baf_samples)},1) array')
                print(e)
                return(None)
        kde = KernelDensity(kernel='gaussian',bandwidth=0.01).fit(baf_samples)
        test_values = np.array([*range(250,500,1)]) / 500
        test_values = test_values.reshape(len(test_values),1)
        scores = kde.score_samples(test_values)
        adjusted_scores = scores - log_density_threshold
        aggregate_scores = aggregate_scores + adjusted_scores
    score_df = pd.DataFrame({'baf_values': test_values.reshape(-1), 'adj_average_scores': aggregate_scores / iterations})
    # look for peak regions in scores
    peak_centers = []
    peak_start, peak_end = None, None
    in_peak = False
    for i in range(len(aggregate_scores)):
        if in_peak and aggregate_scores[i] < 0: # sign flip out of peak
            in_peak = False
            peak_end = test_values[i-1][0]
            peak_centers.append([np.mean([peak_start,peak_end]),peak_end - peak_start])
            peak_start, peak_end = None, None
        elif not in_peak and aggregate_scores[i] > 0: # sign flip into peak
            in_peak = True
            peak_start = test_values[i][0]
    if in_peak:
        peak_end = 1
        peak_centers.append([np.mean([peak_start,peak_end]),peak_end - peak_start])
    if plot_scores:
        peaks = score_df[score_df['adj_average_scores']>0]
        troughs = score_df[score_df['adj_average_scores']<=0]
        plt.figure(figsize=(5,1))
        plt.title(plot_label)
        plt.xlabel('BAF values')
        plt.ylabel('log score')
        plt.xlim(0.5,1)
        plt.ylim(-4,4)
        plt.plot(troughs['baf_values'],troughs['adj_average_scores'],'.',color = 'grey')
        plt.plot(peaks['baf_values'],peaks['adj_average_scores'],'r.')
        plt.hlines(y=0,xmin=0,xmax=1,linestyle = 'dotted')
        plt.vlines(x=peak_centers,ymin = [-4],ymax = [4],linestyle = 'dotted', color = 'red')
        plt.show()
        plt.close()
    return(peak_centers)

def classify_baf_patterns(baf_modes):
    '''
    Bands:
        - b1: 0.5-0.56 (shows up in 2x/4x)
        - b2: 0.56-0.6 (2x/3x mosaic)
        - b3: 0.6-0.68 (3x/4x)
        - b4: 0.68-0.80 (4x)
        - b5 : 0.8-0.92 (mosaic 1x)
        - b6: 0.92-1 (1x, present in all ploidy)
    '''
    b1_bands = [x for x in baf_modes if x[0] <= 0.56]
    b2_bands = [x for x in baf_modes if x[0] > 0.56 and x[0] <= 0.6]
    b3_bands = [x for x in baf_modes if x[0] > 0.6 and x[0] <= 0.68]
    b4_bands = [x for x in baf_modes if x[0] > 0.68 and x[0] <= 0.8]
    b5_bands = [x for x in baf_modes if x[0] > 0.8 and x[0] <= 0.92]
    b6_bands = [x for x in baf_modes if x[0] >0.92]
    width_thresh = {'b1':0.15,'b3':0.17,'b4':0.17,'b6':0.15}
    if len(b6_bands) != 1 or b6_bands[0][1] > width_thresh['b6']:
        return('missing 1x band')
    if sum([len(b1_bands),len(b2_bands),len(b3_bands),len(b4_bands),len(b5_bands)]) == 0:
        return('1x')
    if len(b1_bands) == 1 and b1_bands[0][1] <= width_thresh['b1'] and sum([len(b2_bands),len(b3_bands),len(b4_bands),len(b5_bands)]) == 0:
        return('2x')
    if len(b3_bands) == 1 and b3_bands[0][1] <= width_thresh['b3']: 
        if sum([len(b1_bands),len(b2_bands),len(b4_bands),len(b5_bands)]) == 0:
            return('3x')
        elif len(b1_bands) == 1 and b1_bands[0][1] <= width_thresh['b1']:
            return('4x')
    if len(b4_bands) == 1 and b4_bands[0][1] <= width_thresh['b4']:
        if sum([len(b1_bands),len(b2_bands),len(b3_bands),len(b5_bands)]) == 0:
            return('4x')
        elif len(b1_bands) == 1 and b1_bands[0][1] <= width_thresh['b1']:
            return('4x')
    if len(b2_bands) == 1 and sum([len(b1_bands),len(b3_bands),len(b4_bands),len(b5_bands)]) == 0:
        return('2-3x')
    if len(b5_bands) == 1:
        if sum([len(b1_bands),len(b2_bands),len(b3_bands),len(b4_bands)]) == 0:
            return('1-2x')
        elif len(b1_bands) > 0 or len(b2_bands) > 0:
            return('MCC')
    return('noisy/mosaic')

def summary_classify_baf(summary_df, data_df,lower_limit = 20):
    output_list = []
    for chrom, start, end, n in summary_df[['Chr','Start','End','Probe_count']].values:
        if n < lower_limit:
            output_list.append('too few data')
            continue
        baf_values = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['B Allele Freq'].values
        baf_modes = find_baf_modes(baf_values,iterations=10,log_density_threshold = 0.5)
        output_list.append(classify_baf_patterns(baf_modes))
    return(output_list)

def collapse_summary_df(summary_df, data_df,lrr_ref_dict, sampling_limit = 200):
    working_df = summary_df.copy()
    columns = summary_df.columns
    i_cn = np.where(columns == 'section_CN')[0][0]
    out_df_list = []
    for chrom in np.unique(working_df['Chr'].values):
        c_df_vals = working_df[working_df['Chr'] == chrom].values
        for cn,values in groupby(c_df_vals,lambda x: x[i_cn]):
            values_array = [*values]
            if len(values_array) == 1:
                new_row = pd.DataFrame(dict(zip(columns,values_array[0])),index = [0])
                out_df_list.append(new_row)
                continue
            new_row_dict = dict(zip(columns,[None] * len(columns)))
            start_col, end_col = np.where(columns == 'Start')[0][0], np.where(columns == 'End')[0][0]
            collapsed_start, collapsed_end = min([x[start_col] for x in values_array]), max([x[end_col] for x in values_array])
            segment_df = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= collapsed_start) & (data_df['Position'] <= collapsed_end)]
            lrr = segment_df['Adjusted LRR'].values
            lrr_med = np.median(lrr)
            baf = segment_df['Transformed BAF'].values
            n_probes = len(lrr)
            unique_baf = len(set(baf))
            if n_probes > sampling_limit:
                lrr = random.sample(list(lrr),k = sampling_limit)
            lrr_p_dict = state_tests(lrr,lrr_ref_dict)
            new_row_dict.update({'Chr':[chrom],
                                 'Start':[collapsed_start],
                                 'End':[collapsed_end],
                                 'Length':[collapsed_end - collapsed_start],
                                 'Probe_count':[n_probes],
                                 'unique_BAF_count':[unique_baf],
                                 'Copy_Number_Call':[cn],
                                 'LRR_median':[np.median(lrr_med)],
                                 'p(0x)':[lrr_p_dict['0x']],
                                 'p(1x)':[lrr_p_dict['1x']],
                                 'p(2x)':[lrr_p_dict['2x']],
                                 'p(3x)':[lrr_p_dict['3x']],
                                 'p(4x)':[lrr_p_dict['4x']],
                                 'section_CN':[max_p(list(lrr_p_dict.values()))]})
            new_row = pd.DataFrame(new_row_dict, index = [0])
            out_df_list.append(new_row)
    out_df = pd.concat(out_df_list,ignore_index = True)
    return(out_df)

def state_tests(values,state_sample_dict):
    p_dict = dict()
    for cn,state_samples in state_sample_dict.items():
        p = stats.ranksums(state_samples, values).pvalue
        p_dict.update({cn:p})
    return(p_dict)

def count_X(xy_str,output_float=False):
    X_counts = [xy.count('X') for xy in xy_str.split('/')]
    if len(X_counts) == 1:
        X_cn = X_counts[0]
    else:
        X_cn = np.mean(X_counts)
    if not output_float:
        X_cn = str(X_cn) + 'x'
    return(X_cn)

def get_ref_values(summary_df,data_df):
    out_dict = {'0x':{},'1x':{},'2x':{},'3x':{},'4x':{}}
    for cn in ['0x','1x','2x','3x','4x']:
        cn_df = summary_df[summary_df['fixed'] & (summary_df['section_CN'] == cn)]
        if len(cn_df) == 0:
            continue
        cn_lrr_dict = dict()
        for i in cn_df.index:
            chrom,start,end = cn_df.loc[i,'Chr'],cn_df.loc[i,'Start'],cn_df.loc[i,'End']
            lrr = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
            cn_lrr_dict.update({i:lrr})
        out_dict.update({cn:cn_lrr_dict})
    return(out_dict)

def compare_large_sections(summary_df,data_df,ref_vals,model_ref_dict,header_dict,filter_thresh,model_means): # autosome only
    compare_df = summary_df[~summary_df['fixed'] & (summary_df['BAF_class'] != 'too few data') & (summary_df['Chr'] != 'X')].copy()
    rest_df = summary_df[summary_df['fixed'] | (summary_df['BAF_class'] == 'too few data') | (summary_df['Chr'] == 'X')].copy()
    out_df_list = []
    for i in compare_df.index:
        row = compare_df.loc[i]
        chrom, start, end = row['Chr'], row['Start'], row['End']
        expected_cn = header_dict['Chromosome_summary'][chrom]
        if expected_cn not in ref_vals.keys() or len(ref_vals[expected_cn].keys()) == 0:
            ref_region_counts = {k:len(v) for k,v in ref_vals.items()}
            expected_cn = [k for k,v in ref_region_counts.items() if v == max(ref_region_counts.values())][0]
        ref_indices = np.array(list(ref_vals[expected_cn].keys()))
        lrr_values = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
        if len(lrr_values) > 200:
            lrr_values = random.choices(list(lrr_values),k = 200)
        ref_dist = np.abs(ref_indices - i)
        closest_ref_index = ref_indices[np.where(ref_dist == min(ref_dist))[0][0]]
        ref_lrr = ref_vals[expected_cn][closest_ref_index]
        new_row = pd.DataFrame(dict(zip(row.index,row)),index = [row.name])
        new_row['lor'] = row['lor']
        new_section_CN = classify_region(row,ref_lrr,lrr_values,expected_cn,filter_thresh,model_means)
        new_row['section_CN'] = new_section_CN
        new_row['fixed'] = True
        out_df_list.append(new_row)
    if len(out_df_list) > 0:
        out_df = pd.concat(out_df_list,ignore_index= False)
        out_df['lor'] = compare_df['lor'].values
        out_df = pd.concat([out_df,rest_df],ignore_index= False)
        out_df.sort_index(inplace = True)
    else:
        out_df = rest_df.copy()
    return(out_df)

def compare_large_sections_X(summary_df,data_df,ref_vals,model_ref_dict,header_dict,filter_thresh,model_means):
    XY_str = header_dict['Sex_chromosomes']
    X_cn = majority_cn_chrom(summary_df[(summary_df['Chr'] == 'X')],model_means,by_lrr_median = True)
    expected_X_cn = float(X_cn.strip('x'))
    if expected_X_cn.is_integer():
        ref_X_indices = np.array(list(ref_vals[X_cn].keys()))
        if len(ref_X_indices) == 0: # rare cases of miscall due to LOH, set the most abundant CN as ref (usually '2x')
            ref_region_counts = {k:len(v) for k,v in ref_vals.items()}
            X_cn = [k for k,v in ref_region_counts.items() if v == max(ref_region_counts.values())][0]
            ref_X_indices = np.array(list(ref_vals[X_cn].keys()))
    else:
        return(summary_df) # probably need to do more
    compare_df = summary_df[~summary_df['fixed'] & (summary_df['BAF_class'] != 'too few data') & (summary_df['Chr'] == 'X')].copy()
    rest_df = summary_df[summary_df['fixed'] | (summary_df['BAF_class'] == 'too few data') | (summary_df['Chr'] != 'X')].copy()
    if len(compare_df) == 0:
        return(summary_df)
    out_df_list = []
    for i in compare_df.index:
        row = compare_df.loc[i]
        chrom, start, end = row['Chr'], row['Start'], row['End']
        lrr_values = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
        if len(lrr_values) > 200:
            lrr_values = random.choices(list(lrr_values),k = 200)
        ref_dist = np.abs(ref_X_indices - i)
        closest_ref_index = ref_X_indices[np.where(ref_dist == min(ref_dist))[0][0]]
        ref_lrr = ref_vals[X_cn][closest_ref_index]
        new_row = pd.DataFrame(dict(zip(row.index,row)),index = [row.name])
        new_row['lor'] = row['lor']
        new_section_CN = classify_region(row,ref_lrr,lrr_values,X_cn,filter_thresh,model_means)
        new_row['section_CN'] = new_section_CN
        new_row['fixed'] = True
        out_df_list.append(new_row)
    if len(out_df_list) > 0:
        out_df = pd.concat(out_df_list,ignore_index= False)
        out_df['lor'] = compare_df['lor'].values
        out_df = pd.concat([out_df,rest_df],ignore_index= False)
        out_df.sort_index(inplace = True)
    else:
        out_df = rest_df.copy()
    return(out_df)

# ref_cn is first in the filter_thresh dictionary
def classify_region(row,ref_lrr,test_lrr,ref_cn,filter_thresh,model_means):
    cn_means = dict(zip(['0x','1x','2x','3x','4x'],model_means.reshape(-1)))
    lrr_12_dist, lrr_23_dist = cn_means['2x'] - cn_means['1x'], cn_means['3x'] - cn_means['2x']
    lrr_12_limits = (cn_means['1x'] + lrr_12_dist * 0.3,cn_means['2x'] - lrr_12_dist * 0.3)
    lrr_23_limits = (cn_means['2x'] + lrr_23_dist * 0.3,cn_means['3x'] - lrr_23_dist * 0.3)
    lrr_median = row['LRR_median']
    test_cn, baf_class = row['section_CN'], row['BAF_class']
    if test_cn == baf_class or test_cn == ref_cn:
        return(test_cn)
    if test_cn == '1-2x':
        if lrr_median < lrr_12_limits[0]: 
            return('1x')
        elif lrr_median < lrr_12_limits[1]:
            return('1-2x')
        else:
            return('2x')
    elif test_cn == '2-3x':
        if lrr_median < lrr_23_limits[0]:
            return('2x')
        elif lrr_median < lrr_23_limits[1]:
            return('2-3x')
        else:
            return('3x')
    compare_group = f'{ref_cn}-{test_cn}'
    sample_size = len(test_lrr)
    p_thresh = filter_thresh[compare_group][str(sample_size)]['p']
    outlier_thresh = filter_thresh[compare_group][str(sample_size)]['outlier']
    p_val = stats.ranksums(ref_lrr,test_lrr).pvalue
    q1, q9 = np.quantile(ref_lrr,[0.1,0.9])
    r_outlier = max(sum(test_lrr > q9), sum(test_lrr < q1)) / len(test_lrr)
    if p_val > p_thresh or r_outlier < outlier_thresh:
        return(ref_cn)
    else:
        return(test_cn)

def compare_small_sections(summary_df,data_df,ref_vals,header_dict,filter_thresh,model_means,lor_thresh = 3, outlier_thresh = 45):
    XY_str = header_dict['Sex_chromosomes']
    X_cn = majority_cn_chrom(summary_df[(summary_df['Chr'] == 'X')],model_means,by_lrr_median = True)
    expected_X_cn = float(X_cn.strip('x'))
    working_df = summary_df.copy()
    flanked_array = [False]
    chr_cn_array = working_df[['Chr','section_CN','fixed']].values
    for x in zip(chr_cn_array,chr_cn_array[1:],chr_cn_array[2:]):
        c,cn,fixed = np.array(x).transpose()
        if (not c[0] == c[1] == c[2]) or (not all([fixed[0],fixed[2]])) or fixed[1]: # not the same chromosome, 0 and 2 not both fixed, or 1 is fixed
            flanked_array.append(False)
            continue
        if cn[0] == cn[2] and cn[0] != cn[1]: #flanking regions have the same cn that is not the same as middle region
            flanked_array.append(True)
        else:
            flanked_array.append(False)
    flanked_array.append(False)
    working_df['flanked'] = flanked_array
    fixed_df = working_df[working_df['fixed']].copy()
    flanked_df = working_df[working_df['flanked']].copy()
    larger_regions_df = working_df[(~working_df['flanked']) & (~working_df['fixed']) & (working_df['Probe_count'] >= 10)].copy()
    smaller_regions_df = working_df[(~working_df['flanked']) & (~working_df['fixed']) & (working_df['Probe_count'] < 10)].copy()
    out_df_list = []
    for i in flanked_df.index:
        row = flanked_df.loc[i]
        left_row,right_row = fixed_df.loc[i-1], fixed_df.loc[i+1]
        chrom = row['Chr']
        if chrom == 'X':
            expected_cn = X_cn
        else:
            expected_cn = header_dict['Chromosome_summary'][chrom]
            if expected_cn not in ref_vals.keys() or len(ref_vals[expected_cn].keys()) == 0:
                ref_region_counts = {k:len(v) for k,v in ref_vals.items()}
                expected_cn = [k for k,v in ref_region_counts.items() if v == max(ref_region_counts.values())][0]
        new_row = process_flanked(row,left_row,right_row,data_df,filter_thresh, model_means = model_means,expected_cn = expected_cn,lor_thresh = lor_thresh)
        out_df_list.append(new_row)
    for i in larger_regions_df.index:
        row = larger_regions_df.loc[i]
        chrom = row['Chr']
        if chrom == 'X':
            expected_cn = X_cn
        else:
            expected_cn = header_dict['Chromosome_summary'][chrom]
            if expected_cn not in ref_vals.keys() or len(ref_vals[expected_cn].keys()) == 0:
                ref_region_counts = {k:len(v) for k,v in ref_vals.items()}
                expected_cn = [k for k,v in ref_region_counts.items() if v == max(ref_region_counts.values())][0]
        new_row = process_larger(row,fixed_df = fixed_df,data_df = data_df,filter_thresh = filter_thresh, model_means = model_means,expected_cn = expected_cn)
        out_df_list.append(new_row)
    if len(out_df_list) > 0:
        temp_fixed_df = pd.concat(out_df_list, ignore_index=False)
        temp_fixed_df = pd.concat([fixed_df,temp_fixed_df], ignore_index=False)
        temp_fixed_df.sort_index(inplace = True)
    else:
        temp_fixed_df = fixed_df.copy()
    fixed_indices = np.array(list(temp_fixed_df.index))
    for i in smaller_regions_df.index:
        row = smaller_regions_df.loc[i]
        chrom = row['Chr']
        if chrom == 'X':
            expected_cn = X_cn
        else:
            expected_cn = header_dict['Chromosome_summary'][chrom]
            if expected_cn not in ref_vals.keys() or len(ref_vals[expected_cn].keys()) == 0:
                ref_region_counts = {k:len(v) for k,v in ref_vals.items()}
                expected_cn = [k for k,v in ref_region_counts.items() if v == max(ref_region_counts.values())][0]
        new_row = process_smaller(row,fixed_df = temp_fixed_df,data_df = data_df, expected_cn = expected_cn)
        out_df_list.append(new_row)
    if len(out_df_list) > 0:
        out_df = pd.concat(out_df_list,ignore_index= False)
        out_df = pd.concat([out_df,fixed_df],ignore_index=False)
    else:
        out_df = fixed_df.copy()
    out_df.sort_index(inplace = True)
    out_df.drop(['flanked'], axis = 1, inplace = True)
    return(out_df)

def process_flanked(row,left_row,right_row,data_df,filter_thresh,model_means,expected_cn = '2x',lor_thresh = 3):
    cn_means = dict(zip(['0x','1x','2x','3x','4x'],model_means.reshape(-1)))
    lrr_median = row['LRR_median']
    cn_means_dist = {k:abs(lrr_median - v) for k,v in cn_means.items()}
    section_cn = [k for k,v in cn_means_dist.items() if v == min(cn_means_dist.values())][0]
    cn_value = {'0x':0,'1x':1,'2x':2,'3x':3,'4x':4,'1-2x':1.5,'2-3x':2.5}
    chrom, start, end = row['Chr'], row['Start'], row['End']
    lrr_values = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
    new_row = pd.DataFrame(dict(zip(row.index,row)),index = [row.name])
    left_cn, right_cn = left_row['section_CN'], right_row['section_CN']
    left_len, right_len = left_row['Probe_count'], right_row['Probe_count']
    if section_cn == left_cn: # lrr_median does not deviate much from expected
        new_row['section_CN'] = section_cn
    elif left_cn != expected_cn: # small section flanked by larger abnormal regions
        if (left_len + right_len) > (row['Probe_count'] * 5):
            new_row['section_CN'] = left_cn
        elif left_cn in ['1-2x','2-3x']:
            new_row['section_CN'] = left_cn
        elif cn_value[left_cn] < cn_value[section_cn]:
            if np.any([row['lor'][i] > lor_thresh for i in range(cn_value[row['section_CN']])]):
                new_row['section_CN'] = left_cn
            else:
                new_row['section_CN'] = expected_cn
        elif cn_value[left_cn] > cn_value[section_cn]:
            if np.any([row['lor'][i] > lor_thresh for i in range(int(cn_value[row['section_CN']])+1,5)]):
                new_row['section_CN'] = left_cn
            else:
                new_row['section_CN'] = expected_cn
        else:
            new_row['section_CN'] = expected_cn
    else: # small abnormal region flanked by normal regions
        lrr_len = len(lrr_values)
        if lrr_len >= 10:
            left_start, left_end = left_row['Start'], left_row['End']
            right_start, right_end = right_row['Start'], right_row['End']
            left_lrr = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= left_start) & (data_df['Position'] <= left_end)]['Adjusted LRR'].values
            right_lrr = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= right_start) & (data_df['Position'] <= right_end)]['Adjusted LRR'].values
            ref_lrr = np.concatenate([left_lrr,right_lrr])
            q1, q9 = np.quantile(ref_lrr,[0.1,0.9])
            r_outlier = max(sum(lrr_values > q9),sum(lrr_values < q1)) / len(lrr_values)
            normal_lrr_sample = np.random.choice(ref_lrr,size = 200)
            p_val = stats.ranksums(normal_lrr_sample,lrr_values).pvalue
            comp_group = f'{expected_cn}-{section_cn}'
            thresh_dict = filter_thresh[comp_group][str(lrr_len)]
            p_thresh,outlier_thresh = thresh_dict['p'], thresh_dict['outlier']
            if p_val < p_thresh and r_outlier >= outlier_thresh:
                new_row['section_CN'] = section_cn
            else:
                new_row['section_CN'] = expected_cn
        else:
            new_row['section_CN'] = expected_cn
    new_row['fixed'] = True
    return(new_row)

def process_larger(row,fixed_df,data_df,filter_thresh,model_means,expected_cn = '2x'):
    cn_means = dict(zip(['0x','1x','2x','3x','4x'],model_means.reshape(-1)))
    lrr_median = row['LRR_median']
    cn_means_dist = {k:abs(lrr_median - v) for k,v in cn_means.items()}
    section_cn = [k for k,v in cn_means_dist.items() if v == min(cn_means_dist.values())][0]
    cn_value = {'0x':0,'1x':1,'2x':2,'3x':3,'4x':4,'1-2x':1.5,'2-3x':2.5}
    fixed_indices = np.array(list(fixed_df[fixed_df['section_CN'] == expected_cn].index))
    chrom, start, end = row['Chr'], row['Start'], row['End']
    lrr_values = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
    new_row = pd.DataFrame(dict(zip(row.index,row)),index = [row.name])
    if section_cn == expected_cn:
        new_row['section_CN'] = section_cn
        new_row['fixed'] = True
        return(new_row)
    ref_lrr = get_closest_ref_lrr(row,fixed_df,data_df,expected_cn = expected_cn, n = 200)
    q1, q9 = np.quantile(ref_lrr,[0.1,0.9])
    r_outlier = max(sum(lrr_values > q9),sum(lrr_values < q1)) / len(lrr_values)
    p_val = stats.ranksums(ref_lrr,lrr_values).pvalue
    comp_group = f'{expected_cn}-{section_cn}'
    thresh_dict = filter_thresh[comp_group][str(len(lrr_values))]
    p_thresh,outlier_thresh = thresh_dict['p'], thresh_dict['outlier']
    if p_val < p_thresh and r_outlier >= outlier_thresh:
        new_row['section_CN'] = section_cn
    else:
        new_row['section_CN'] = expected_cn
    new_row['fixed'] = True
    return(new_row)

def get_closest_ref_lrr(row,fixed_df,data_df,expected_cn = '2x', n = 200):
    fixed_indices = np.array(list(fixed_df[fixed_df['section_CN'] == expected_cn].index))
    ref_dist = np.abs(fixed_indices - row.name)
    ranked_fixed_indices = np.array([*zip(fixed_indices,ref_dist)])
    sorted_indices = np.array(sorted(ranked_fixed_indices,key=lambda x: x[1]))
    sorted_indices = [x[0] for x in sorted_indices]
    ref_lrr = []
    for i in sorted_indices:
        chrom, start,end = fixed_df.loc[i,'Chr'],fixed_df.loc[i,'Start'], fixed_df.loc[i,'End']
        ref_lrr_i = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
        ref_lrr.extend(ref_lrr_i)
        if len(ref_lrr) < n:
            continue
        else:
            ref_lrr = ref_lrr[:n]
            break
    return(ref_lrr)

def process_smaller(row,fixed_df,data_df,expected_cn = '2x'):
    cn_value = {'0x':0,'1x':1,'2x':2,'3x':3,'4x':4,'1-2x':1.5,'2-3x':2.5}
    fixed_indices = np.array(list(fixed_df.index))
    chrom, start, end = row['Chr'], row['Start'], row['End']
    lrr_values = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= start) & (data_df['Position'] <= end)]['Adjusted LRR'].values
    new_row = pd.DataFrame(dict(zip(row.index,row)),index = [row.name])
    # need to test edge cases
    ref_dist = fixed_indices - row.name
    try:
        closest_left_i = fixed_indices[np.where(ref_dist == max(ref_dist[ref_dist<0]))[0][0]]
        left_cn = fixed_df.loc[closest_left_i,'section_CN']
        left_start, left_end = fixed_df.loc[closest_left_i,'Start'], fixed_df.loc[closest_left_i,'End']
        left_lrr = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= left_start) & (data_df['Position'] <= left_end)]['Adjusted LRR'].values
        left_p = stats.ranksums([lrr_values,left_lrr]).pvalues
    except:
        left_p = 0
        left_cn = expected_cn
    try:
        closest_right_i = fixed_indices[np.where(ref_dist == min(ref_dist[ref_dist>0]))[0][0]]
        right_cn = fixed_df.loc[closest_right_i,'section_CN']
        right_start, right_end = fixed_df.loc[closest_right_i,'Start'], fixed_df.loc[closest_right_i,'End']
        right_lrr = data_df[(data_df['Chr'] == chrom) & (data_df['Position'] >= right_start) & (data_df['Position'] <= right_end)]['Adjusted LRR'].values
        right_p = stats.ranksums([lrr_values,right_lrr]).pvalues
    except:
        right_p = 0
        right_cn = expected_cn
    if left_p >= right_p:
        new_row['section_CN'] = left_cn
    else:
        new_row['section_CN'] = right_cn
    new_row['fixed'] = True
    return(new_row)

def majority_cn_chrom(df,model_means, fixed_only = False, by_lrr_median = False, majority_thresh = 0.4):
    total_probes = sum(df['Probe_count'].values)
    cn_len_dict = dict()
    if fixed_only:
        working_df = df[df['fixed'] ==  True].copy()
    else:
        working_df = df.copy()
    if not by_lrr_median:
        for n_p,cn in working_df[['Probe_count','section_CN']].values:
            if cn not in cn_len_dict.keys():
                cn_len_dict.update({cn:n_p})
            else:
                cn_len_dict[cn] += n_p
        if len(working_df) > 0 and max(cn_len_dict.values()) >= total_probes * majority_thresh:
            top_cn = [k for k,v in cn_len_dict.items() if v == max(cn_len_dict.values())][0]
            return(top_cn)
    weighted_median_list = []
    for n_p,lrr in df[['Probe_count','LRR_median']].values:
        weighted_median_list.extend([lrr for i in range(int(n_p))])
    lrr_median = np.median(weighted_median_list)
    cn_means = dict(zip(['0x','1x','2x','3x','4x'],model_means.reshape(-1)))
    cn_means_dist = {k:abs(lrr_median - v) for k,v in cn_means.items()}
    top_cn = [k for k,v in cn_means_dist.items() if v == min(cn_means_dist.values())][0]
    return(top_cn)

def find_abnormal_regions(summary_df, X_count):
    X_cn = f'{X_count}x'
    # autosomes
    autosomal_abnormal = summary_df[(summary_df['Chr'] != 'X') & (summary_df['section_CN'] != '2x')]
    # X chromosome
    if X_cn not in ['1x','2x']:
        X_cn = '2x'
    x_abnormal = summary_df[(summary_df['Chr'] == 'X') & (summary_df['section_CN'] != X_cn)]
    all_abnormal = pd.concat([autosomal_abnormal,x_abnormal],ignore_index=True)
    all_abnormal.drop(['Copy_Number_Call','p(0x)','p(1x)','p(2x)','p(3x)','p(4x)','lor','fixed'],axis = 1,inplace = True)
    return(all_abnormal)
    
def header_to_string(header):
    working_header = header.copy()
    output_list = []
    if 'nheader' in working_header:
        del working_header['nheader']
    for k,v in working_header.items():
        if k != 'Chromosome_summary':
            output_str = f'# {k}={v}\n'
            output_list.append(output_str)
        else:
            output_str = f'# {k}:\n'
            output_list.append(output_str)
            for chrom, cn in v.items():
                output_str = f'## {chrom}: {cn}\n'
                output_list.append(output_str)
    final_str = ''.join(output_list)
    return(final_str)

def save_summary(summary_table,header,file):
    header_str = header_to_string(header)
    with open(file,'w') as f:
        f.write(header_str)
    summary_table.to_csv(file,sep = '\t', mode = 'a', index = None)
    return

def main():
    arg_list = sys.argv[1:]
    short_opts = 's:d:m:o:'
    long_opts = ['summary=','data=','model=','output=']
    try:
        opt_list = getopt.getopt(arg_list, short_opts, long_opts)[0]
    except getopt.error as error:
        sys.exit(error)
    summary_file, data_file, model_file, output_file = None, None, None, None
    for current_arg, current_val in opt_list:
        if current_arg in ['-s','--summary']:
            summary_file = current_val
        elif current_arg in ['-d','--data']:
            data_file = current_val
        elif current_arg in ['-m','--model']:
            model_file = current_val
        elif current_arg in ['-o','--output']:
            output_file = current_val
    sys.stderr.write(f'data_file: {data_file}\n')
    sys.stderr.write(f'summary_file: {summary_file}\n')

    data_df = read_result_data(data_file)
    header = read_header(summary_file)
    model_names_dict, model_params, filter_thresh = read_info_json(model_file)
    Y_count = count_Y(header['Sex_chromosomes'])[0]
    model_name = model_names_dict[header['Microarray_type']]
    filter_thresh = filter_thresh[model_name]
    models, labels = make_model_from_dict(model_params,model_names_dict[header['Microarray_type']])
    lrr_model, lrr_labels = models['lrr_model'], labels['lrr_model']
    lrr_ref_dict = sample_distributions(lrr_model,lrr_labels)
    summary_df = read_table(summary_file,skiprows=header['nheader'])
    print('Finding anchor regions')
    s_step_1 = find_anchors(summary_df,header,data_df,lrr_ref_dict,lrr_model.means_)
    print('Retrieving reference lrr values')
    ref_vals = get_ref_values(s_step_1,data_df)
    print('Comparing large regions')
    s_step_2 = compare_large_sections(s_step_1,data_df,ref_vals,lrr_ref_dict,header,filter_thresh,lrr_model.means_)
    s_step_2 = compare_large_sections_X(s_step_2,data_df,ref_vals,lrr_ref_dict,header,filter_thresh,lrr_model.means_)
    print('Comparing small regions')
    s_step_3 = compare_small_sections(s_step_2,data_df,ref_vals,header,filter_thresh,lrr_model.means_)
    print('Collapsing adjacent regions with identical copy number')
    s_step_4 = collapse_summary_df(s_step_3,data_df,lrr_ref_dict)
    s_step_4 = collapse_summary_df(s_step_4,data_df,lrr_ref_dict)
    s_step_4['lor'] = s_step_4.apply(lambda x: log_or(x['p(0x)'],x['p(1x)'],x['p(2x)'],x['p(3x)'],x['p(4x)']),axis = 1)
    s_step_4['BAF_class'] = summary_classify_baf(s_step_4,data_df)
    
    X_chrom = s_step_4[s_step_4['Chr'] == 'X'].copy()
    X_cn = majority_cn_chrom(X_chrom,lrr_model.means_)
    if X_cn == '1-2x':
        new_xy_str = 'X' + 'Y' * Y_count + '/XX' + 'Y' * Y_count
    elif X_cn == '2-3x':
        new_xy_str = 'XX' + 'Y' * Y_count + '/XXX' + 'Y' * Y_count
    else:
        X_count = int(X_cn.strip('x'))
        new_xy_str = 'X' * X_count + 'Y' * Y_count
    header['Sex_chromosomes'] = new_xy_str
    for c in header['Chromosome_summary']:
        cn = majority_cn_chrom(s_step_4[s_step_4['Chr'] == c],lrr_model.means_)
        header['Chromosome_summary'].update({c:cn})
    print('Saving outputs')
    abnormal_out_df = find_abnormal_regions(s_step_4,X_count)
    abnormal_out_name = output_file + '_abnormal.txt'
    save_summary(abnormal_out_df,header,abnormal_out_name)
    full_out_df = s_step_4.drop(['Copy_Number_Call','lor','fixed'], axis = 1)
    full_out_name = output_file + '_full_table.txt'
    save_summary(full_out_df,header,full_out_name)
    return

if __name__ == '__main__':
    main()




















