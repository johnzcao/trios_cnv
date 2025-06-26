def parse_info_file(file,n):
    with open(file,'r') as f:
        lines = f.readlines()
    target_info = lines[n].strip().split('\t')
    return(target_info)
    
def read_file(file_name):
    if file_name.endswith('.gz'):
        gz = True
    else:
        gz = False
    n_skip = 0
    n_snps = None
    if gz:
        f = gzip.open(file_name)
        while True:
            l = f.readline().decode()
            n_skip += 1
            if 'Content' in l:
                cont = l.strip().split('\t')[1]
                print(f'Content: {cont}')
            if 'Total SNPs' in l:
                n_snps = int(l.strip().split('\t')[1])
                print(f'Total SNPs: {n_snps}')
            if '[Data]' in l:
                break
    else: 
        f = open(file_name)
        while True:
            l = f.readline()
            n_skip += 1
            if 'Content' in l:
                cont = l.strip().split('\t')[1]
                print(f'Content: {cont}')
            if 'Total SNPs' in l:
                n_snps = int(l.strip().split('\t')[1])
                print(f'Total SNPs: {n_snps}')
            if '[Data]' in l:
                break
    f.close()
    
    if gz: 
        df = pd.read_table(file_name,compression='gzip', skiprows=n_skip, dtype=str)
    else:
        df = pd.read_table(file_name, skiprows=n_skip, dtype=str)
    df['Position'] = pd.to_numeric(df['Position'])
    df['Log R Ratio'] = pd.to_numeric(df['Log R Ratio'])
    df['B Allele Freq'] = pd.to_numeric(df['B Allele Freq'])
    if 'GC Score' in df.columns:
        df['GC Score'] = pd.to_numeric(df['GC Score'])
    else:
        df['GC Score'] = [1] * len(df)
    # deal with XY homology region
    # fornow I just preserve the information about XY homology and change the Chr value into 'X'
    df['XY Homology'] = (df['Chr'] == 'XY')
    df.loc[df['Chr'] == 'XY', 'Chr'] = 'X'
    # remove entries where Chr is '0'
    df = df[df['Chr'] != '0']
    df['Chr'] = pd.Categorical(df['Chr'],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])
    df.reset_index(drop = True, inplace = True)
    return(df,cont)

def read_models_json(file,content = None):
    models_dict,labels_dict = dict(),dict()
    with open(file,'r') as f:
        models = json.load(f)
    if content is None:
        return(models['model_names'])
    model_name = models['model_names'][content]
    model_params = models[model_name]
    lrr_params, baf_params = reorder_states(model_params['lrr_model']),reorder_states(model_params['baf_model'])
    lrr_model,baf_model = make_model(lrr_params),make_model(baf_params)
    models_dict.update({'lrr_model':lrr_model,'baf_model':baf_model})
    lrr_labels = {int(k):v for k,v in lrr_params['Labels'].items()}
    baf_labels = {int(k):v for k,v in baf_params['Labels'].items()}
    labels_dict.update({'lrr_model':lrr_labels,'baf_model':baf_labels})
    return(models_dict,labels_dict)

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

def filter_and_sort(df, gc_thresh = 0.15):
    # GC score filter
    df_filt = df.copy()[df['GC Score'] >= gc_thresh]
    no_call_rate = (((len(df) - len(df_filt)) / len(df))) * 100
    print(f'\tFiltered out probes with GC Score < {gc_thresh} (Filtered dataframe n = {len(df_filt)}, removed {no_call_rate:.2f}% of all probes).')
    # sorting by chromosome then position
    df_filt.sort_values(by = ['Chr','Position'], ascending = [True, True], inplace = True)
    df_filt.reset_index(drop = True, inplace = True)
    return(df_filt)

def find_potential_outlier(df,val_col,pos_col = 'Position',gap_size = 0.02, plot_groups = False):
    index = df.index
    pos = df[pos_col].values
    values = df[val_col].values
    if val_col == 'B Allele Freq':
        values = np.abs(0.5 - values) + 0.5
    indexed_list = [*zip(index,pos,values)]
    v_sorted_list_dict = dict(enumerate(sorted(indexed_list, key = lambda v: v[2])))
    sorted_values = np.array([v[2] for k,v in v_sorted_list_dict.items()])
    value_diff = sorted_values[1:] - sorted_values[:-1]
    group_id = 0
    group_member = [v_sorted_list_dict[0]]
    group_dict = dict()
    for i,d in enumerate(value_diff):
        if d < gap_size:
            group_member.append(v_sorted_list_dict[i+1])
        else:
            group_dict.update({group_id:group_member})
            group_id += 1
            group_member = [v_sorted_list_dict[i+1]]
    group_dict.update({group_id:group_member})
    if plot_groups:
        groups = group_dict.keys()
        plt.figure(figsize=(10,2))
        for k,v in group_dict.items():
            pos = [x[1] for x in v]
            val = [x[2] for x in v]
            plt.plot(pos,val,'.')
        plt.legend(groups,fontsize = 8)
        plt.show()
        plt.close()
    out_df = df.copy()
    out_df[f'candidate outliers ({val_col})'] = False
    cutoff = len(sorted_values) * 0.02
    for i,members in group_dict.items():
        if len(members) < cutoff:
            for index, pos, val in members:
                out_df.loc[index,f'candidate outliers ({val_col})'] = True
        else:
            continue
    return(out_df)

def filter_outlier(df,n_neighbors = 20,max_dist = 0.02):
    filtered_df = df.copy()
    min_index = min(df.index)
    max_index = max(df.index)
    outlier_index = df[df['candidate outliers (Log R Ratio)'] | df['candidate outliers (B Allele Freq)']].index
    for i in outlier_index:
        is_outlier_lrr = False
        is_outlier_baf = False
        lrr_outlier = df.loc[i,'candidate outliers (Log R Ratio)']
        baf_outlier = df.loc[i,'candidate outliers (B Allele Freq)']
        left_bound = max(i - n_neighbors,min_index)
        right_bound = min(i + n_neighbors + 1, max_index)
        if lrr_outlier:
            value = df.loc[i,'Log R Ratio']
            neighbors = df.loc[left_bound:right_bound,'Log R Ratio'].values
            n_clust = sum(np.abs(value - neighbors) <= max_dist) # will be at least 1, since lrr_outlier is included in neighbors
            if n_clust < 3:
                is_outlier_lrr = True
        if baf_outlier:
            value = df.loc[i,'B Allele Freq']
            neighbors = df.loc[left_bound:right_bound,'B Allele Freq'].values
            neighbors = np.abs(0.5 - neighbors) + 0.5
            n_clust = sum(np.abs(value - neighbors) <= max_dist)
            if n_clust < 3:
                is_outlier_baf = True
        if is_outlier_lrr or is_outlier_baf:
            filtered_df.drop(i,axis = 0, inplace = True)
    return(filtered_df)

def filter_outlier_chrom(df,window_size = 1_000_000,lrr_gap_size = 0.1,baf_gap_size = 0.05):
    chrom_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
    out_df_list = []
    for chrom in chrom_list:
        c_df = df[df['Chr'] == chrom].copy()
        if len(c_df) == 0:
            continue
        if chrom == 'Y':
            out_df_list.append(c_df)        
        min_pos,max_pos = min(c_df['Position'].values), max(c_df['Position'].values)
        left_bound,right_bound = min_pos, min_pos + window_size
        while left_bound < max_pos:
            #print(left_bound)
            window_df = c_df[(c_df['Position'] >= left_bound) & (c_df['Position'] < right_bound)].copy()
            #print(len(window_df))
            if len(window_df) == 0:
                left_bound += window_size
                right_bound += window_size
                continue
            elif len(window_df) <= 50:
                out_df_list.append(window_df)
                left_bound += window_size
                right_bound += window_size
                continue
            window_df = find_potential_outlier(window_df, 'Log R Ratio',gap_size=lrr_gap_size)
            window_df = find_potential_outlier(window_df, 'B Allele Freq',gap_size=baf_gap_size)
            filtered_window_df = filter_outlier(window_df)
            out_df_list.append(filtered_window_df)
            left_bound += window_size
            right_bound += window_size
    out_df = pd.concat(out_df_list,ignore_index=True)
    out_df.drop(['candidate outliers (Log R Ratio)','candidate outliers (B Allele Freq)'],axis = 1, inplace = True)
    return(out_df)

def lrr_adjust(df,lrr_model,lrr_labels):
    target_means = dict(zip(lrr_labels.values(),[x[0] for x in lrr_model.means_]))
    autosome_baf = autosome_baf_peaks(df)
    lrr_2x,lrr_3x,lrr_4x = [],[],[]
    for c in autosome_baf:
        for r,pattern_dict in autosome_baf[c].items():
            if pattern_dict['pattern'] == '2x':
                lrr_2x.extend(pattern_dict['lrr'])
            elif pattern_dict['pattern'] == '3x':
                lrr_3x.extend(pattern_dict['lrr'])
            elif pattern_dict['pattern'] == '4x':
                lrr_4x.extend(pattern_dict['lrr'])
    if len(lrr_2x) > 0:
        target_med = target_means['2x']
        sample_med = np.median(lrr_2x)
    else:
        if len(lrr_3x) > 0:
            target_med = target_means['3x']
            sample_med = np.median(lrr_3x)
        else:
            if len(lrr_4x) > 0:
                target_med = target_means['4x']
                sample_med = np.median(lrr_4x)
            else:
                target_med = None
    if target_med is None:
        df['Adjusted LRR'] = df['Log R Ratio']
        return(df)
    diff = target_med - sample_med
    df['Adjusted LRR'] = df['Log R Ratio'].apply(lambda x: x + diff)
    return(df,autosome_baf)

def find_baf_modes(baf_values,sampling_size = 400, log_density_threshold = 0, iterations = 1, plot_scores = False, plot_label = None):
    # Mirrored at 0.5
    baf_values = list(abs(0.5 - np.array(baf_values)) + 0.5)
    aggregate_scores =  np.repeat(0,250)
    for i in range(iterations):
        baf_samples = random.choices(list(baf_values),k = sampling_size)
        if np.array(baf_samples).ndim != 2:
            try:
                baf_samples =  np.array(baf_samples).reshape(len(baf_samples),1)
            except ValueError as e:
                print(f'Failed to reshape data to ({len(baf_values)},1) array')
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

def classify_baf_patterns(baf_modes): # working solution: only defining normal bands, all else is blanket labeled "noisy/mosaic"
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
        return('noisy/mosaic')
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
    return('noisy/mosaic')

def autosome_baf_peaks(df,window=10_000_000,chroms = 'all',iterations = 100):
    if chroms == 'all':
        chroms = [str(x) for x in range(1,23)]
    all_baf_modes = {}
    for c in chroms:
        chrom_dict = {}
        c_df = df[df['Chr'] == c].copy()
        if len(c_df) == 0:
            continue
        max_pos = max(c_df['Position'].values)
        left_edge = 0
        right_edge = left_edge + window
        while left_edge < max_pos:
            window_baf = c_df[(c_df['Position'] > left_edge) & (c_df['Position'] <= right_edge)]['B Allele Freq'].values
            window_baf = window_baf[(window_baf > 0) & (window_baf < 1) & ~np.isnan(window_baf)]
            window_lrr = c_df[(c_df['Position'] > left_edge) & (c_df['Position'] <= right_edge)]['Log R Ratio'].values
            if len(window_baf) < 200:
                left_edge = right_edge
                right_edge = left_edge + window
                continue
            window_name = str(left_edge) + '-' + str(right_edge)
            window_list = find_baf_modes(window_baf,iterations = iterations)
            window_baf_pattern = classify_baf_patterns(window_list)
            chrom_dict.update({window_name:{'pattern':window_baf_pattern,'lrr':window_lrr}})
            left_edge = right_edge
            right_edge = left_edge + window
        all_baf_modes.update({c:chrom_dict})
    return(all_baf_modes)

def autosomes_summary(baf_modes):
    out_dict = dict()
    for c in baf_modes:
        patterns = [v['pattern'] for v in baf_modes[c].values()]
        c_mode = statistics.mode(patterns)
        out_dict.update({c:c_mode})
    return(out_dict)

def dup_intervals(file):
    df = pd.read_table(file, dtype={'chrom':object,'start':int,'end':int})
    interval_dict = dict()
    for chrom, c_df in df.groupby('chrom'):
        c_df.sort_values(by='start',inplace = True)
        intervals = c_df[['start','end']].values
        interval_dict.update({chrom : intervals})
    return(interval_dict)

def seg_dup_filter(df,seg_dup_dict):
    new_df = pd.DataFrame()
    for chrom,c_df in df.groupby('Chr',observed = False):
        positions = c_df['Position'].values
        c_ints = seg_dup_dict[chrom].copy()
        in_interval = np.array(sum([positions >= i[0] for i in c_ints]) + sum([positions <= i[1] for i in c_ints]) - len(c_ints),dtype = bool)
        filtered_c_df = c_df[~in_interval]
        new_df = pd.concat([new_df,filtered_c_df],ignore_index=True)
    new_df.reset_index(inplace = True, drop = True)
    return(new_df)

def transform_BAF(df,chr_col = 'Chr',pos_col = 'Position', baf_col = 'B Allele Freq', dist_limit = 100_000):
    pos_vals = list(df[[chr_col,pos_col]].apply(lambda x: pos_transform(x.iloc[0],x.iloc[1]) ,axis=1))
    baf_vals = list(df[baf_col])
    values_dict = dict(enumerate(zip(pos_vals,baf_vals)))
    left_bound, current_index, right_bound = 0,0,0
    baf_window = {right_bound: (values_dict[right_bound][0],values_dict[right_bound][1])} # Initial value for the dictionary
    output_dict = dict()
    while current_index < len(values_dict):
        current_pos, current_baf = values_dict[current_index]
        current_lims = [current_pos - dist_limit, current_pos + dist_limit]
        # add all entries to the right that's within the distance limit
        while right_bound + 1 < len(values_dict):
            next_pos, next_val = values_dict[right_bound + 1]
            if abs(next_pos - current_pos) < dist_limit:            
                right_bound += 1
                baf_window.update({right_bound:(next_pos,next_val)})
            else:
                break
        # remove all entries to the left that's outside the distance limit
        while abs(values_dict[left_bound][0] - current_pos) > dist_limit:
            try:
                del baf_window[left_bound]
            except:
                pass
            left_bound += 1
        output_dict.update({current_index:(current_pos,baf_transform(baf_window,current_index))})
        current_index += 1
    df['Transformed BAF'] = [abs(x[1]-0.5)+0.5 for x in output_dict.values()]
    return(df)
        
def pos_transform(c,pos):
    if c == 'X':
        return(pos + 23_000_000_000)
    elif c == 'Y':
        return(pos + 24_000_000_000)
    else:
        return(pos + int(c) * 1_000_000_000)

def baf_transform(baf_dict,current_ind,lim = 0.2):
    upper, lower = 1 - lim, lim
    current_pos,current_val = baf_dict[current_ind]
    if current_val >= lower and current_val <= upper:
        return(current_val)
    dist_dict = {abs(current_pos - k):v for k,v in baf_dict.values()}
    keys_by_dist = list(dist_dict.keys())
    keys_by_dist.sort()
    for k in keys_by_dist:
        if dist_dict[k] >= lower and dist_dict[k] <= upper:
            return((dist_dict[k]))
    return(current_val)

def proba_dict(x,y):
    d = dict()
    for i,xi in enumerate(x):
        for j,yi in enumerate(y):
            name = str(i)+','+str(j)
            d.update({name:xi*yi})
    return(d)

def categorize_proba(proba,categorize_dict):
    out_dict = dict()
    for k,v in categorize_dict.items():
        out_dict.update({k:sum([p for c,p in proba.items() if c in v])})
    return(out_dict)

def most_likely(proba_sum):
    amb_removed = proba_sum.copy()
    del amb_removed['amb']
    max_cat = max(zip(amb_removed.values(),amb_removed.keys()))[1]
    return(max_cat)

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

def state_tests(values,state_sample_dict):
    p_dict = dict()
    for cn,state_samples in state_sample_dict.items():
        p = stats.ranksums(state_samples, values).pvalue
        p_dict.update({cn:p})
    return(p_dict)

def segment_ranksums(df,lrr_model,lrr_labels,baf_model,baf_labels,categorize_dict,sampling_limit = 200):
    lrr_ref_dict = sample_distributions(lrr_model,lrr_labels)
    baf_ref_dict = sample_distributions(baf_model,baf_labels,bounded = True)
    input_list = df[['CN call','Position','Adjusted LRR','Transformed BAF']].values
    output_list = []
    groups = groupby(input_list, lambda x:x[0])
    for cn, vals in groups:
        vals = [*vals]
        n = len(vals)
        if n > sampling_limit:
            vals = random.sample(vals,k = sampling_limit)
        lrr_test_vals = np.array([x[2] for x in vals])
        baf_test_vals = np.array([x[3] for x in vals])
        lrr_p_dict = state_tests(lrr_test_vals,lrr_ref_dict)
        baf_p_dict = state_tests(baf_test_vals,baf_ref_dict)
        category_p = categorize_proba(proba_dict(lrr_p_dict.values(),baf_p_dict.values()),categorize_dict)
        group_call = most_likely(category_p)
        output_list.extend(np.repeat(group_call,n))
    return(output_list)

def summary_table(df, lrr_model, lrr_labels,sampling_limit = 200):
    lrr_ref_dict = sample_distributions(lrr_model,lrr_labels)
    chrom = df['Chr'].values[0]
    input_list = df[['grouped CN call','Position','Adjusted LRR','Transformed BAF']].values
    output_summary_list = []
    groups = groupby(input_list, lambda x:x[0])
    # initiate first groups
    current_cn, current_vals = '',[]
    while True:
        try:
            current_cn, current_vals = next(groups)
        except StopIteration:
            break
        current_vals = [*current_vals]
        unique_tbafs  = list(set([x[3] for x in current_vals])) 
        n_tbafs = len(unique_tbafs)
        current_lrrs = [x[2] for x in current_vals]
        current_pos = [x[1] for x in current_vals]
        start, end = min(current_pos), max(current_pos)
        length = end - start
        probe_count = len(current_vals)
        lrr_med = np.median(current_lrrs)
        if len(current_lrrs) > sampling_limit:
            lrr_p_dict = state_tests(random.sample(current_lrrs,k = sampling_limit),lrr_ref_dict)
        else:
            lrr_p_dict = state_tests(current_lrrs,lrr_ref_dict)
        new_row = pd.DataFrame({'Chr':[chrom],
                                'Start':[start],
                                'End':[end],
                                'Length':[length],
                                'Probe_count':[probe_count],
                                'unique_BAF_count':[n_tbafs],
                                'Copy_Number_Call':[current_cn],
                                'LRR_median':[lrr_med],
                                'p_values(0x,1x,2x,3x,4x)':[','.join([np.format_float_scientific(x,precision = 2) for x in list(lrr_p_dict.values())])]})
        output_summary_list.append(new_row)
    output_df = pd.concat(output_summary_list,ignore_index=True)
    return(output_df)

def plot_prediction(df,save_name, chrom = 'all', title = None, gap = 2.5, plt_dim = (15,5),XY_hom = False):
    colors = {'2x LOH':'turquoise',
              '3x LOH':'maroon',
              '4x LOH':'forestgreen',
              'ambiguous (1x LRR with 2x BAF)':'gold',
              'ambiguous (1x LRR with 3x BAF)':'silver',
              'ambiguous (2x LRR with 3x BAF)':'lightblue',
              'ambiguous (3x LRR with 2x BAF)':'tan',
              '0x':'indigo',
              '1x':'lightcoral',
              '2x':'dodgerblue',
              '3x':'darkorange',
              '4x':'limegreen'}
    y_lim_lrr, y_lim_baf = [-4,4], [-0.1,1.1]
    fig,(ax_lrr,ax_baf) = plt.subplots(2,figsize = plt_dim)
    fig.subplots_adjust(bottom=0.3)
    if title is not None:
        fig.suptitle(title,fontsize = 16)
    ax_lrr.set_ylim(y_lim_lrr)
    ax_baf.set_ylim(y_lim_baf)
    ax_lrr.set_ylabel('Log R Ratio')
    ax_baf.set_ylabel('B Allele Freq')
    if chrom != 'all':
        c_df = df[df['Chr'] == chrom].copy()
        if chrom == 'X' and not XY_hom:
            c_df = c_df[c_df['XY Homology'] == False].copy()
        c_df['color'] = c_df['CN call'].apply(lambda x: colors[x])
        if len(c_df) == 0:
            print(f'\tNo data for chromosome {chrom}.')
            return
        X = c_df['Position']
        Y = c_df['Adjusted LRR']
        grouped_by_calls = c_df.groupby('grouped CN call')
        for name,group in grouped_by_calls:
            X = group['Position']
            Y = group['Adjusted LRR']
            ax_lrr.scatter(X,Y,c = colors[name] ,s = 4, label = name)
            Y = group['B Allele Freq']
            ax_baf.scatter(X,Y,c = colors[name] ,s = 4, label = name)
            ax_baf.set_xlabel(f'Chromosome {chrom}')
            ax_baf.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),ncol=6)
        plt.savefig(save_name)
        return
    # adjusted X axis calculation
    chr_size = {'1':249250621,'2':243199373,'3':198022430,'4':191154276,'5':180915260,
                '6':171115067,'7':159138663,'8':146364022,'9':141213431,'10':135534747,
                '11':135006516,'12':133851895,'13':115169878,'14':107349540,'15':102531392,
                '16':90354753,'17':81195210,'18':78077248,'19':59128983,'20':63025520,
                '21':48129895,'22':51304566,'X':155270560,'Y':59373566}
    chr_plot_start = dict()
    sum_sizes = 0
    for c in chr_size:
        chr_plot_start.update({c : sum_sizes})
        sum_sizes += (chr_size[c] + gap * 10000000)
    chr_plot_start.update({'end': sum_sizes})
    chr_pos = [p + chr_plot_start[c] for c,p in df[['Chr','Position']].values]
    df['Adjusted Position'] = chr_pos
    # x ticks position calculation
    l = list(chr_plot_start.values())
    x_ticks_pos = []
    for i in range(1,len(l)):
        x_ticks_pos.append((l[i] + l[i-1] - gap * 10000000)/2)
    grouped_by_calls = df.groupby('CN call')
    for name,group in grouped_by_calls:
        X = group['Adjusted Position']
        Y = group['Adjusted LRR']
        ax_lrr.scatter(X,Y,c = colors[name] ,s = 4, label = name)
        Y = group['B Allele Freq']
        ax_baf.scatter(X,Y,c = colors[name] ,s = 4, label = name)
        ax_baf.set_xlabel(f'Chromosome {chrom}')
    ax_lrr.vlines([x - gap  * 10000000 / 2 for x in list(chr_plot_start.values())[1:24]],ymin = -4 ,ymax = 4,linestyles='dotted',color = 'gray', alpha = 0.5)
    ax_baf.vlines([x - gap  * 10000000 / 2 for x in list(chr_plot_start.values())[1:24]],ymin = -4 ,ymax = 4,linestyles='dotted',color = 'gray', alpha = 0.5)
    ax_lrr.set_xticks(x_ticks_pos, chr_size.keys())
    ax_baf.set_xticks(x_ticks_pos, chr_size.keys())
    ax_baf.set_xlabel('Chromosome')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),ncol=5)
    plt.savefig(save_name)
    return

def main():
    time_format = '%Y-%m-%d %H:%M:%S'
    arg_list = sys.argv[1:]
    short_opts = 'f:o:m:c:i:d:pgh'
    long_opts = ['help','include_XY_homology','min_probes=','file_list=','entry_number=','config=','ignore_ambiguous']
    try:
        opt_list = getopt.getopt(arg_list, short_opts, long_opts)[0]
    except getopt.error as error:
        sys.exit(error)
    file_name, save_name, plot_name, iscn = None, None, None, None
    file_list, entry_number = None, None
    segmental_dup_table = 'segmental_duplication_hg19.txt'
    model_file = None
    target_chrom = 'all'
    incl_XY_hom, gz, make_plot = False, False, False
    min_probes = 10
    fdr_thresh = 0.05
    ignore_ambiguous = False
    XY_str = None
    categorize_dict = {'0x':['0,0','0,1','0,2'],
                       '1x':['1,0'],
                       '2x':['2,0','2,1'],
                       '3x':['3,0','3,2'],
                       '4x':['4,0','4,1','4,2'],
                       'amb':['1,1','1,2','2,2','3,1']}
    
    if any(['--config' in x for x in opt_list]):
        config_file = opt_list[['--config' in x for x in opt_list].index(True)][1]
        with open(config_file,'r') as f:
            lines = f.readlines()
        lines = [l.strip().split('=') for l in lines]
        param_dict = {x[0].strip():x[1].strip() for x in lines}
        for k,v in param_dict.items():
            if k == 'model_file':
                model_file = v
            elif k == 'segmental_dup_table':
                segmental_dup_table = v
            elif k == 'target_chrom':
                target_chrom = v
            elif k == 'save_as_gz':
                gz = (v == 'True')
            elif k == 'plot_results':
                make_plot = (v == 'True')
            elif k == 'include_XY_homology':
                incl_XY_hom = (v == 'True')
            elif k == 'min_probes':
                min_probes = int(v)
            elif k == 'fdr_thresh':
                fdr_thresh = float(v)
            elif k == 'ignore_ambiguous':
                ignore_ambiguous = True
    for current_arg, current_val in opt_list:
        if current_arg == '-f':
            file_name = current_val
            if not Path(file_name).exists():
                sys.exit(f'Cannot find directory: {file_name}')
            print(f'{datetime.strftime(datetime.now(),time_format)}\tInput file: {os.path.abspath(file_name)}')
        elif current_arg == '-o':
            save_name = current_val
        elif current_arg == '-c':
            target_chrom = current_val
            print(f'{datetime.strftime(datetime.now(),time_format)}\tProcess only chromosome {target_chrom}')
        elif current_arg == '-m':
            model_file = current_val
        elif current_arg == '-i':
            iscn = current_val
        elif current_arg == '-d':
            segmental_dup_table = current_val
        elif current_arg == '-p':
            make_plot = True
        elif current_arg == '-g':
            gz = True
        elif current_arg == '--include_XY_homology':
            incl_XY_hom = True
        elif current_arg == '--min_probes':
            min_probes = current_val
        elif current_arg == '--file_list':
            file_list = current_val
        elif current_arg == '--entry_number':
            entry_number = int(current_val)
        elif current_arg == '--ignore_ambiguous':
            ignore_ambiguous = True
        if (file_list is not None) and (entry_number is not None):
            iscn,file_name,save_name,XY_str = parse_info_file(file_list,entry_number)
    ######## XY_str is absent for single file operation ##########
    #nX = XY_str.split('/')[0].count('X')
    #X_ploidy = str(nX) + 'x'
    log_file = open(save_name + '.log','w')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tInput file: {os.path.abspath(file_name)}\n')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tTarget chromosome(s): {target_chrom}\n')
    pd.set_option('display.float_format', lambda x: '%.7f' % x)
    # Processing steps
    #1. Read file
    print(f'{datetime.strftime(datetime.now(),time_format)}\tReading file..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tReading file..\n')
    df, content = read_file(file_name)
    models,labels = read_models_json(model_file,content)
    lrr_model, baf_model = models.values()
    lrr_labels, baf_labels = labels.values()
    #2. Filter by quality
    print(f'{datetime.strftime(datetime.now(),time_format)}\tFiltering file..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tFiltering file..\n')
    df = filter_and_sort(df,gc_thresh=0.3)
    if target_chrom != 'all':
        df = df[(df['Chr']==target_chrom) | (df['Chr']=='X') | (df['Chr']=='Y')].copy()
        df.reset_index(drop = True, inplace = True)
    #3. Remove noise
    print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving noise..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving noise..\n')
    original_len = len(df)
    df = filter_outlier_chrom(df,lrr_gap_size=0.1,baf_gap_size = 0.05)
    removed_len = original_len - len(df)
    print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoved {removed_len} probes ({removed_len/original_len}%)')
    #4. LRR adjust
    print(f'{datetime.strftime(datetime.now(),time_format)}\tAdjusting LRR')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tAdjusting LRR')
    trans_df, autosome_baf = lrr_adjust(df,lrr_model, lrr_labels)
    #5 whole chromosome summary
    autosome_dict = autosomes_summary(autosome_baf)
    if not incl_XY_hom:
        print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving XY homology probes..')
        log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving XY homology probes..\n')
        trans_df = trans_df[trans_df['XY Homology'] == False].copy()
        trans_df.reset_index(inplace = True, drop = True)
    print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving probes in segmental duplications..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving probes in segmental duplications..\n')
    dups = dup_intervals(segmental_dup_table)
    trans_df = seg_dup_filter(trans_df,dups)
    trans_df = transform_BAF(trans_df)
    #6. Prediction by BAF and LRR model probabilities, followed by low confidence filter
    print(f'{datetime.strftime(datetime.now(),time_format)}\tPredicting CN by BAF and LRR..')
    chrom_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    output_df_list = []
    summary_df_list = []
    for c in chrom_list:
        print(c)
        c_df = trans_df[trans_df['Chr'] == c].copy()
        c_df = c_df[~np.isnan(c_df['Log R Ratio']) & ~np.isnan(c_df['Transformed BAF'])]
        lrr_proba = lrr_model.predict_proba(c_df[['Log R Ratio']].values)
        baf_proba = baf_model.predict_proba(c_df[['Transformed BAF']].values)
        c_df['LRR_proba(0x,1x,2x,3x,4x)'] = [','.join([(np.format_float_scientific(x,precision = 2)) for x in lst]) for lst in lrr_proba]
        c_df['BAF_proba(1x,2x,3x)'] = [','.join([(np.format_float_scientific(x,precision = 2)) for x in lst]) for lst in baf_proba]
        predictions = np.array([most_likely(categorize_proba(proba_dict(x,y),categorize_dict)) for x,y in zip(lrr_proba,baf_proba)])
        c_df['CN call'] = predictions
        print(f'\tFiltering low confidence predictions')
        log_file.write(f'\tFiltering low confidence predictions\n')
        c_df['grouped CN call'] = segment_ranksums(c_df,lrr_model,lrr_labels,baf_model,baf_labels,categorize_dict)
        c_df_summary = summary_table(c_df, lrr_model, lrr_labels,sampling_limit = 200)
        output_df_list.append(c_df)
        summary_df_list.append(c_df_summary)
    output_df = pd.concat(output_df_list,axis = 0, ignore_index = True)
    summary_df = pd.concat(summary_df_list,axis = 0, ignore_index = True)
    if gz:
        out_name = save_name + '.txt.gz'
    else:
        out_name = save_name + '.txt'
    print(f'{datetime.strftime(datetime.now(),time_format)}\tSaving file as: {out_name}')
    output_df.to_csv(out_name,sep = '\t',index = False)
    
    summary_out_name = save_name + '_summary.txt'
    print(f'{datetime.strftime(datetime.now(),time_format)}\tMaking summary table: {summary_out_name}')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tMaking summary table: {summary_out_name}\n')
    
    with open(summary_out_name,'w') as f:
        f.write(f'# Input_file={file_name}\n')
        f.write(f'# Microarray_type={content}\n')
        f.write(f'# Original_ISCN={iscn}\n')
        f.write(f'# Sex_chromosomes={XY_str}\n')
        f.write(f'# Processed_chromosome={target_chrom}\n')
        f.write('# LRR_HMM_Model:\n')
        f.write(f'## State_labels={list(lrr_labels.values())}\n')
        f.write(f'## Means={lrr_model.means_.tolist()}\n')
        f.write(f'## Covars={lrr_model.covars_.tolist()}\n')
        f.write(f'## Transmat=[{lrr_model.transmat_.tolist()}]\n')
        f.write(f'## Start_prob={lrr_model.startprob_.tolist()}\n')
        f.write('# BAF_HMM_Model:\n')
        f.write(f'## State_labels={list(baf_labels.values())}\n')
        f.write(f'## Means={baf_model.means_.tolist()}\n')
        f.write(f'## Covars={baf_model.covars_.tolist()}\n')
        f.write(f'## Transmat={baf_model.transmat_.tolist()}\n')
        f.write(f'## Start_prob={baf_model.startprob_.tolist()}\n')
        f.write(f'# Autosome summary:\n')
        for c,s in autosome_dict.items():
            f.write(f'## {c}: {s}\n')
    summary_df.to_csv(summary_out_name,mode = 'a',sep = '\t',index = False)
    if make_plot:
        plot_name = save_name + '.png'
        print(f'{datetime.strftime(datetime.now(),time_format)}\tPloting results and save as {plot_name}')
        log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tPloting results and save as {plot_name}\n')
        plot_prediction(output_df,plot_name, chrom = target_chrom,XY_hom = incl_XY_hom)
    log_file.close()

help_msg = '''usage: python copy_number_predict.py -f <file.txt[.gz]> -o <output.txt> [options]
Mandatory arguments:
    -f: input file
    -o: output file (name stem only)
    -m: file with model parameters, can be setup in configure file (not needed if defined by --config)
    -d: Segmental duplication table, can be setup in configure file (not needed if defined by --config)
    
Options:
    --config: load config file to set up all parameters, except sample specific arguments (-f, -o, and -i)
    -i: ISCN value from metadata
    -c: Chromosome, default to 'all'
    -p: Plot prediction and save as file
    -g: save tables with .gz format
    -h or --help: display help message
    --include_XY_homology: use to keep XY homology probes in prediction
    --min_probes: minimum number of probes for an abnormal region to include in the summary table, default 10
    --ignore_ambiguous: Don't keep ambiguous results in summary and plot
'''

if __name__ == '__main__':
    import os,sys,getopt,json
    arg_list = sys.argv[1:]
    if ('--help' in arg_list) or ('-h' in arg_list) or len(arg_list) == 0:
        print(help_msg)
        sys.exit(0)
    import gzip,random,statistics
    from matplotlib import pyplot as plt
    from pathlib import Path
    from datetime import datetime
    import pandas as pd
    import numpy as np
    from hmmlearn import hmm
    from sklearn.neighbors import KernelDensity
    from itertools import groupby
    import scipy.stats as stats
    main()