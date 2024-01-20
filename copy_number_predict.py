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
            if 'Total SNPs' in l:
                n_snps = int(l.strip().split('\t')[1])
                print(f'\tTotal SNPs in file: {n_snps}')
            if '[Data]' in l:
                break
    else: 
        f = open(file_name)
        while True:
            l = f.readline()
            n_skip += 1
            if 'Total SNPs' in l:
                n_snps = int(l.strip().split('\t')[1])
                print(f'\tTotal SNPs in file: {n_snps}')
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
    df['GC Score'] = pd.to_numeric(df['GC Score'])
    # deal with XY homology region (collapse into X chromosome)
    df['XY Homology'] = (df['Chr'] == 'XY')
    df.loc[df['Chr'] == 'XY', 'Chr'] = 'X'
    # remove entries where Chr is '0'
    df = df[df['Chr'] != '0']
    df['Chr'] = pd.Categorical(df['Chr'],['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])
    df.reset_index(drop = True, inplace = True)
    return(df)

def filter_and_sort(df, gc_thresh = 0.15):
    # GC score filter
    df_filt = df.copy()[df['GC Score'] >= gc_thresh]
    no_call_rate = (((len(df) - len(df_filt)) / len(df))) * 100
    print(f'\tFiltered out probes with GC Score < {gc_thresh} (Filtered dataframe n = {len(df_filt)}, removed {no_call_rate:.2f}% of all probes).')
    # sorting by chromosome then position
    df_filt.sort_values(by = ['Chr','Position'], ascending = [True, True], inplace = True)
    df_filt.reset_index(drop = True, inplace = True)
    return(df_filt)

# Functions to remove probes in segmental duplication regions
def dup_intervals(file):
    df = pd.read_table(file, dtype={'chrom':object,'start':int,'end':int})
    interval_dict = dict()
    for chrom, c_df in df.groupby('chrom'):
        c_df.sort_values(by='start',inplace = True)
        intervals = [pd.Interval(x[0],x[1],closed = 'both') for x in c_df[['start','end']].values]
        interval_dict.update({chrom : intervals})
    return(interval_dict)

def check_interval(pos, intervals):
    for i in intervals:
        if pos in i:
            return(False)
    return(True)

# assuming sorted df and intervals
def filter_chrom(df,intervals):
    pos_list = df['Position'].values
    out_list = []
    for p in pos_list:
        intervals = [i for i in intervals if i.right > p]
        out_list.append(check_interval(p,intervals))
    return(out_list)

def seg_dup_filter(df,seg_dup_dict):
    new_df = pd.DataFrame()
    for chrom,c_df in df.groupby('Chr'):
        chrom_intervals = seg_dup_dict[chrom].copy()
        new_c_df = c_df[filter_chrom(c_df,chrom_intervals)]
        new_df = pd.concat([new_df,new_c_df],ignore_index=True)
    new_df.reset_index(drop= True,inplace = True)
    return(new_df)

def find_singlet(l,max_dist = 0.3):
    if type(l) != dict:
        indexed_list = [*enumerate(l)]
    else: 
        indexed_list = [(k,v) for k,v in l.items()]
    sorted_list_dict = dict(enumerate(sorted(indexed_list, key = lambda x: x[1])))
    out_dict = dict()
    for k in sorted_list_dict:
        if k == 0:
            current_index, current_val = sorted_list_dict[k]
            next_index, next_val = sorted_list_dict[k+1]
            if np.abs(current_val - next_val) <= max_dist:
                out_dict.update({current_index : False})
            else: 
                out_dict.update({current_index : True})
            continue
        if k == len(sorted_list_dict)-1: # last one
            last_val, last_index = current_val, current_index
            current_val, current_index = next_val, next_index
            if np.abs(current_val - last_val) <= max_dist:
                out_dict.update({current_index : False})
            else: 
                out_dict.update({current_index : True})
            continue
        last_val, last_index = current_val, current_index
        current_val, current_index = next_val, next_index
        next_index, next_val = sorted_list_dict[k+1]
        if np.abs(current_val - next_val) > max_dist and np.abs(current_val - last_val) > max_dist:
            out_dict.update({current_index : True})
        else:
            out_dict.update({current_index : False})
    return(out_dict)



def baf_denoise(df,chr_col = 'Chr',pos_col = 'Position', baf_col = 'B Allele Freq', window_size = 1_000_000):
    out_df = pd.DataFrame()
    for c_name,c_df in df.groupby(chr_col):
        if len(c_df) == 0:
            continue
        c_df.reset_index(inplace = True, drop = True)
        print(f'\tChr{c_name}')
        pos_vals = list(c_df[pos_col])
        baf_vals = list(c_df[baf_col].apply(lambda x: np.abs(0.5 - x) + 0.5))
        values_dict = dict(enumerate(zip(pos_vals,baf_vals)))
        singlet_index_list = []
        start = np.floor(min(pos_vals)/1000000000) * 1000000000
        end = start + window_size
        while start < max(pos_vals):
            window_values_dict = {k:v[1] for k,v in values_dict.items() if ((v[0] >= start) and (v[0] < end))}
            if len(window_values_dict) < 5:
                start = end
                end = end + window_size
                continue
            potential_singlet = find_singlet(window_values_dict)
            if any(potential_singlet.values()):
                singlet_index = [k for k,v in potential_singlet.items() if v == True][0]
                singlet_pos = values_dict[singlet_index][0]       
                # look forward and backwards one window size to validate singlet
                singlet_flanking = {k:v[1] for k,v in values_dict.items() if ((v[0] >= singlet_pos - window_size) and (v[0] < singlet_pos + window_size))}
                singlet_validation = find_singlet(singlet_flanking)
                if singlet_validation[singlet_index] == True:
                    singlet_index_list.append(singlet_index)
            start = end
            end = end + window_size
        c_df = c_df.drop(index = singlet_index_list)
        out_df = pd.concat([out_df,c_df],ignore_index = True)
    out_df.reset_index(drop = True, inplace = True)
    return(out_df)

# BAF transformation functions
def transform_BAF(df,chr_col = 'Chr',pos_col = 'Position', baf_col = 'B Allele Freq', dist_limit = 100_000):
    pos_vals = list(df[[chr_col,pos_col]].apply(lambda x: pos_transform(x[0],x[1]) ,axis=1))
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

def call_ploidy(prediction, model_labels):
    l = len(prediction)
    counts = dict()
    for i in model_labels:
        counts.update({i:list(prediction).count(i)})
    top_state,second_state=[0,0],[0,0]
    for i,v in counts.items():
        if v > top_state[1]:
            second_state = top_state
            top_state = [i,v]
            continue
        if v > second_state[1]:
            second_state = [i,v]
            continue
    out_dict = {'most likely': (model_labels[top_state[0]],top_state[1]/l*100,l),
                'second likely':(model_labels[second_state[0]],second_state[1]/l*100,l)}
    return(out_dict)

def check_Y_presence(df,Y_sample_regions):
    out_list = []
    for start,end in Y_sample_regions.values():
        sample_df = df[(df['Chr'] == 'Y') & (df['Position'] > start) & (df['Position'] < end)]
        sample_med = np.median(sample_df['Log R Ratio'].values)
        if sample_med > -1:
            out_list.append(True)
        else:
            out_list.append(False)
    return(statistics.mode(out_list))

def lrr_adjust(df, baf_model, baf_labels, lrr_model, lrr_labels, sample_regions):
    X_sample_regions, Y_sample_regions, XY_sample_regions = sample_regions['X'], sample_regions['Y'], sample_regions['XY']
    Y_presence = check_Y_presence(df,Y_sample_regions)
    target_means = dict(zip(lrr_labels.values(),[x[0] for x in lrr_model.means_]))
    X_df = df[df['Chr'] == 'X'].copy()
    XY_hom_predict_baf, X_sample_predict_baf, XY_hom_predict_lrr, X_sample_predict_lrr = dict(), dict(), dict(), dict()
    XY_hom_lrr, X_sample_lrr = dict(), dict()
    for k,v in XY_sample_regions.items():
        sub_df = X_df[(X_df['Position'] > v[0]) & (X_df['Position'] < v[1]) & X_df['XY Homology']].copy()
        if len(sub_df) == 0:
            continue
        XY_hom_lrr.update({k:sub_df['Log R Ratio'].values})
        pred_states = baf_model.predict(sub_df[['Transformed BAF']].values)
        CN_call = call_ploidy(pred_states,baf_labels)
        XY_hom_predict_baf.update({k:CN_call['most likely']})
        pred_states = lrr_model.predict(sub_df[['Log R Ratio']].values)
        CN_call = call_ploidy(pred_states,lrr_labels)
        XY_hom_predict_lrr.update({k:CN_call['most likely']})
    for k,v in X_sample_regions.items():
        sub_df = X_df[(X_df['Position'] > v[0]) & (X_df['Position'] < v[1])].copy()
        if len(sub_df) == 0:
            continue
        X_sample_lrr.update({k:sub_df['Log R Ratio'].values})
        pred_states = baf_model.predict(sub_df[['Transformed BAF']].values)
        CN_call = call_ploidy(pred_states,baf_labels)
        X_sample_predict_baf.update({k:CN_call['most likely']})
        pred_states = lrr_model.predict(sub_df[['Log R Ratio']].values)
        CN_call = call_ploidy(pred_states,lrr_labels)
        X_sample_predict_lrr.update({k:CN_call['most likely']})
    XY_consensus = statistics.mode([v[0] for k,v in XY_hom_predict_baf.items()])
    X_consensus = statistics.mode([v[0] for k,v in X_sample_predict_baf.items()])
    if XY_consensus == '2x' and X_consensus == '2x': # normal female or XXYY
        if not Y_presence:
            X_ploidy, Y_ploidy = '2x','0x'
        else: 
            X_ploidy, Y_ploidy = '2x', '2x'
    if XY_consensus == '2x' and X_consensus == '1x': # normal male
        if Y_presence:
            X_ploidy, Y_ploidy = '1x','1x'
        else:
            X_ploidy, Y_ploidy = None,None
    if XY_consensus == '2x' and X_consensus == '3x': # XXXY
        if Y_presence:
            X_ploidy, Y_ploidy = '3x','1x'
        else:
            X_ploidy, Y_ploidy = None,None
    if XY_consensus == '3x' and X_consensus == '3x': # XXX
        if not Y_presence:
            X_ploidy, Y_ploidy = '3x','0x'
        else: 
            X_ploidy, Y_ploidy = None, None
    if XY_consensus == '3x' and X_consensus == '2x': # XXY
        if Y_presence:
            X_ploidy, Y_ploidy = '2x','1x'
        else:
            X_ploidy, Y_ploidy = None,None
    if XY_consensus == '3x' and X_consensus == '1x': # XYY or XXY with LOH
        if Y_presence:
            X_ploidy, Y_ploidy = '1x','2x'
        else:
            X_ploidy, Y_ploidy = None,None
    if XY_consensus == '1x' and X_consensus == '1x': # monosomy X female or LOH
        if not Y_presence:
            X_ploidy, Y_ploidy = '1x','0x'
        else: 
            X_ploidy, Y_ploidy = None, None
    if X_ploidy is None:
        print('\tUnable to decide X/Y ploidy, ambiguous results')
        return()
    adjust_regions_names = [k for k,v in X_sample_predict_baf.items() if v[0] == X_ploidy]
    adjust_regions_lrr = [X_sample_lrr[n] for n in adjust_regions_names]
    target_med = target_means[X_ploidy]
    lrr_med = np.median([item for sublist in adjust_regions_lrr for item in sublist])
    diff = target_med - lrr_med
    df['Adjusted LRR'] = df['Log R Ratio'].apply(lambda x: x + diff)
    return(X_ploidy, Y_ploidy, df)

def copy_number_calls(LRR_predict,BAF_predict):
    if LRR_predict == '0x':
        return('0x')
    elif LRR_predict == '1x':
        if BAF_predict == '1x':
            return('1x')
        elif BAF_predict == '2x':
            return('ambiguous (1x LRR with 2x BAF)')
        elif BAF_predict == '3x':
            return('ambiguous (1x LRR with 3x BAF)')
    elif LRR_predict == '2x':
        if BAF_predict == '1x':
            return('2x')
        elif BAF_predict == '2x':
            return('2x')
        elif BAF_predict == '3x':
            return('ambiguous (2x LRR with 3x BAF)')
    elif LRR_predict == '3x':
        if BAF_predict == '1x':
            return('3x')
        elif BAF_predict == '2x':
            return('ambiguous (3x LRR with 2x BAF)')
        elif BAF_predict == '3x':
            return('3x')
    elif LRR_predict == '4x':
        if BAF_predict == '1x':
            return('4x')
        elif BAF_predict == '2x':
            return('4x')
        elif BAF_predict == '3x':
            return('4x')

def CN_call_cleanup(df,dist_limit = 100_000):
    input_list = df[['CN call','Position','Transformed BAF','LRR_predict_label', 'BAF_predict_label']].values
    output_list = []
    groups = groupby(input_list, lambda x:x[0])
    # initiate first groups
    current_cn, current_vals = next(groups)
    current_vals = [list(x) for x in current_vals]
    next_cn, next_vals = next(groups)
    next_vals = [list(x) for x in next_vals]
    # dealing with short list with low confidence
    if len(current_vals) < 5 and len(next_vals) > 5:
        output_list = output_list + [next_cn] * len(current_vals)
    elif len(current_vals) < 5 and len(next_vals) < 5:
        output_list = output_list + [x[0] for x in current_vals]
    #dealing with ambiguous calls
    elif current_cn.startswith('ambiguous') and not next_cn.startswith('ambiguous'):
        grouped_vals = detect_gaps(current_vals,dist_limit) # Each element contains unique tBAF at [0] and original data at [1]
        set_list = []
        for tBAFs, sub_vals in grouped_vals:
            if len(sub_vals) < 5 or len(tBAFs) < 3:
                set_list = set_list + ([next_cn] * len(sub_vals))
            else:
                set_list = set_list + [x[0] for x in sub_vals]
        output_list = output_list + set_list
    elif current_cn.startswith('ambiguous') and next_cn.startswith('ambiguous'):
        output_list = output_list + [x[0] for x in current_vals]
    # condition for non-ambiguous calls
    else:
        output_list = output_list + [current_cn] * len(current_vals)
    # loop through the rest of the groups
    for cn,vals in groups:
        prev_cn,prev_vals = current_cn, current_vals
        current_cn, current_vals = next_cn, next_vals
        next_cn, next_vals = cn, [list(x) for x in vals]
        #print(f'current_vals:{len(current_vals)}')
        # Non-abmiguous cases
        if current_cn in ['1x','2x','3x','4x']:
            output_list = output_list + [current_cn] * len(current_vals) 
            continue
        # check noise
        if current_cn == '0x' and len(current_vals) < 5:
            if not next_cn.startswith('ambiguous'):
                output_list = output_list + [next_cn] * len(current_vals)
            elif not prev_cn.startswith('ambiguous'):
                output_list = output_list + [prev_cn] * len(current_vals)
            else:
                output_list = output_list + [next_vals[0][3]] * len(current_vals)
            continue
        elif current_cn == '0x' and len(current_vals) >= 5:
            output_list.append([x[0] for x in current_vals])
            continue
        # dealing with short calls
        if (len(current_vals) < 5 or len(list(set([x[2] for x in current_vals]))) < 3) :
            if not next_cn.startswith('ambiguous'):
                output_list = output_list + [next_cn] * len(current_vals)
            elif not prev_cn.startswith('ambiguous'):
                output_list = output_list + [prev_cn] * len(current_vals)
            else:
                output_list = output_list + [current_cn] * len(current_vals) 
        #dealing with ambiguous calls
        elif current_cn.startswith('ambiguous') and not next_cn.startswith('ambiguous'):
            grouped_vals = detect_gaps(current_vals,dist_limit) # Each element contains unique tBAF at [0] and original data at [1]
            set_list = []
            for tBAFs, sub_vals in grouped_vals:
                if len(sub_vals) < 5 or len(tBAFs) < 3:
                    set_list = set_list + [next_cn] * len(sub_vals)
                else:
                    set_list = set_list + [x[0] for x in sub_vals]
            output_list = output_list + set_list
        elif current_cn.startswith('ambiguous') and next_cn.startswith('ambiguous'):
            output_list = output_list + [current_cn] * len(current_vals) 
        else:
            output_list = output_list + [current_cn] * len(current_vals) 
    # deal with the last group
    prev_cn,prev_vals = current_cn, current_vals
    current_cn, current_vals = next_cn, next_vals
    # dealing with short list with low confidence
    if len(current_vals) < 5 and len(prev_vals) > 5:
        output_list = output_list + [prev_cn] * len(current_vals)
    elif len(current_vals) < 5 and len(prev_vals) < 5:
        output_list = output_list + [x[0] for x in current_vals]
    #dealing with ambiguous calls
    elif current_cn.startswith('ambiguous') and not prev_cn.startswith('ambiguous'):
        grouped_vals = detect_gaps(current_vals,dist_limit) # Each element contains unique tBAF at [0] and original data at [1]
        set_list = []
        for tBAFs, sub_vals in grouped_vals:
            if len(sub_vals) < 5 or len(tBAFs) < 3:
                set_list = set_list + ([prev_cn] * len(sub_vals))
            else:
                set_list = set_list + [x[0] for x in sub_vals]
        output_list = output_list + set_list
    elif current_cn.startswith('ambiguous') and prev_cn.startswith('ambiguous'):
        output_list = output_list + [x[0] for x in current_vals]
    # condition for non-ambiguous calls
    else:
        output_list = output_list + [current_cn] * len(current_vals)
    df['CN call'] = output_list
    return(df)

def detect_gaps(input_list, limit = 100_000):
    pos = [x[1] for x in input_list]
    gaps = [e-s > 100000 for s,e in zip(pos,pos[1:])]
    breaks = [0] + [i + 1 for i,x in enumerate(gaps) if x == True] + [len(input_list)]
    if len(breaks) == 2:
        unique_tBAF = set([x[2] for x in input_list])
        return([[unique_tBAF,input_list]])
    groups = [*zip(breaks,breaks[1:])]
    out_list = []
    for s,e in groups:
        g = input_list[s:e]
        unique_tBAF = set([x[2] for x in g])
        out_list.append([unique_tBAF,g])
    return(out_list)
    
# plot prediction results
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
        grouped_by_calls = c_df.groupby('CN call')
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
    chr_size = {'1':249250621,'2':243199373,'3':198022430,'4':191154276,'5':180915260,
                '6':171115067,'7':159138663,'8':146364022,'9':141213431,'10':135534747,
                '11':135006516,'12':133851895,'13':115169878,'14':107349540,'15':102531392,
                '16':90354753,'17':81195210,'18':78077248,'19':59128983,'20':63025520,
                '21':48129895,'22':51304566,'X':155270560,'Y':59373566}
    
    # adjusted X axis calculation
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

def select_controls(df, expected_copy, model_mean = 0, model_var = 0.015, size = 1000):
    control_df = df[df['CN call'] == expected_copy].copy()
    if len(control_df) < size:
        control_vals = np.random.normal(loc = model_mean,scale = np.sqrt(model_var),size = size)
    else:
        control_vals = control_df.sample(n=size)['Adjusted LRR'].values
    return(control_vals)

def ranksum_stats(control_vals, test_vals, alt = 'two-sided',n_tests = 19, max_samples = 50):
    if len(test_vals) > max_samples:
        test_vals = np.random.choice(test_vals,size = max_samples, replace = False)
    rank_p_vals = []
    for i in range(n_tests):
        c = np.random.choice(control_vals,size = len(test_vals), replace = False)
        rank_p_vals.append(stats.ranksums(test_vals,c,alternative=alt).pvalue)
    return(np.median(rank_p_vals))

def fdr(ps):
    ranked_ps = stats.rankdata(ps)
    fdr = [p * len(ps) / r for p,r in zip(ps, ranked_ps)]
    fdr = [1 if x > 1 else x for x in fdr]
    return(fdr)

def summary_table(df, X_cn, minimum_probes = 10, lrr_2x_mean = 0, lrr_2x_var = 0.015):
    values_list = df[['Chr','Position','XY Homology','Adjusted LRR','LRR_predict_label','CN call']].values
    grouped_values = groupby(values_list,lambda x: [x[0]] + [x[5]]) # Chr and CN call are keys
    out_df = pd.DataFrame(columns=['Chr','Start','End','Length','Copy_Number_Call','Probe_count','LRR_median','p_value'])
    for (c, CN_call), vals in grouped_values:
        if c == 'X':
            expected_cn = X_cn
            control_vals = select_controls(df,X_cn)
        elif c == 'Y':
            continue
        else:
            expected_cn = '2x'
            control_vals = select_controls(df,'2x',lrr_2x_mean,lrr_2x_var)
        if CN_call == expected_cn:
            continue
        vals = [x.tolist() for x in vals]
        pos_list = [x[1] for x in vals if x[2] == False] #discard XY homology probes
        if CN_call in ['0x','1x','2x','3x','4x']:
            lrr_cn = int(CN_call[0])
        else:
            lrr_cn = int(statistics.mode([x[4] for x in vals])[0])
        lrr_list = [x[3] for x in vals]
        if len(pos_list) < minimum_probes:
            continue
        if lrr_cn < int(expected_cn[0]):
            pval = ranksum_stats(control_vals, lrr_list, alt = 'less')
        elif lrr_cn > int(expected_cn[0]):
            pval = ranksum_stats(control_vals, lrr_list, alt = 'greater')
        else:
            pval = ranksum_stats(control_vals, lrr_list, alt = 'two-sided')
        out_df.loc[len(out_df.index)] = [c,min(pos_list),max(pos_list),max(pos_list)-min(pos_list),CN_call,len(pos_list),np.median(lrr_list),pval]
    out_df['FDR'] = fdr(out_df['p_value'])
    return(out_df)

help_msg = '''usage: python copy_number_predict.py -f <file.txt[.gz]> -o <output.txt> [options]
Mandatory arguments:
    -f: input file
    -o: output file (name stem only)
    
Options:
    -i: ISCN value from metadata
    -l: Log R Ratio model, has a default
    -b: B Allele Freq model, has a default
    -c: Chromosome, default to 'all'
    -d: Segmental duplication table
    -p: Plot prediction and save as file
    -g: save tables with .gz format
    -h or --help: display help message
    --include_XY_homology: use to keep XY homology probes in prediction
    --min_probes: minimum number of probes for an abnormal region to include in the summary table, default 10
'''

def main():
    time_format = '%Y-%m-%d %H:%M:%S'
    arg_list = sys.argv[1:]
    short_opts = 'f:o:l:b:c:i:d:pgh'
    long_opts = ['help','include_XY_homology','min_probes=']
    try:
        opt_list = getopt.getopt(arg_list, short_opts, long_opts)[0]
    except getopt.error as error:
        sys.exit(error)
    if (('--help','') in opt_list) or (('-h','') in opt_list) or len(arg_list) == 0:
        print(help_msg)
        sys.exit(0)
    
    file_name, save_name, plot_name, iscn = None, None, None, None
    segmental_dup_table = 'segmental_duplication_hg19.txt'
    lrr_model_file, baf_model_file = 'lrr_5_states_hmm_w_labels.pkl','baf_3_states_hmm_w_labels.pkl'
    target_chrom = 'all'
    excl_XY_hom, gz, make_plot = True, False, False
    min_probes = 10

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
        elif current_arg == '-l':
            lrr_model_file = current_val
        elif current_arg == '-b':
            baf_model_file = current_val
        elif current_arg == '-i':
            iscn = current_val
        elif current_arg == '-d':
            segmental_dup_table = current_val
        elif current_arg == '-p':
            make_plot = True
        elif current_arg == '-g':
            gz = True
        elif current_arg == '--include_XY_homology':
            excl_XY_hom = False
        elif current_arg == '--min_probes':
            min_probes = current_val
        
    XY_hom_splits = {'H1': (1, 1500000),
                     'H2': (1500000, 2300000),
                     'H3': (2300000, 12000000),
                     'H4': (85000000, 91000000),
                     'H5': (91000000, 95000000),
                     'H6': (150000000, 160000000)}
    X_samples = {'S1': (15000000, 17000000),
                 'S2': (30000000, 31000000),
                 'S3': (43000000, 45000000),
                 'S4': (68000000, 70000000),
                 'S5': (95000000, 97000000),
                 'S6': (110000000, 112000000),
                 'S7': (123000000, 125000000),
                 'S8': (133000000, 135000000),
                 'S9': (143000000, 145000000)}
    Y_samples = {'Y1':(8_000_000,9_000_000),
                 'Y2':(15_000_000,16_000_000),
                 'Y3':(22_000_000,23_000_000)}
    sampling_dicts = {'X':X_samples,'Y':Y_samples,'XY':XY_hom_splits}
    
    log_file = open(save_name + '.log','w')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tInput file: {os.path.abspath(file_name)}\n')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tTarget chromosome(s): {target_chrom}\n')
    
    with open('baf_3_states_hmm_w_labels.pkl','rb') as f: 
        baf_model, baf_model_labels = pickle.load(f)
    with open('lrr_5_states_hmm_w_labels.pkl','rb') as f: 
        lrr_model, lrr_model_labels = pickle.load(f)
    pd.set_option('display.float_format', lambda x: '%.7f' % x)
    
    # Processing steps
    #1. Read file
    print(f'{datetime.strftime(datetime.now(),time_format)}\tReading file..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tReading file..\n')
    df = read_file(file_name)
    
    #2. Filter by quality
    print(f'{datetime.strftime(datetime.now(),time_format)}\tFiltering file..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tFiltering file..\n')
    df = filter_and_sort(df)
    if target_chrom != 'all':
        df = df[(df['Chr']==target_chrom) | (df['Chr']=='X') | (df['Chr']=='Y')].copy()
        df.reset_index(drop = True, inplace = True)
    
    #3. Remove singlet BAF
    print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving singlet BAF..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving singlet BAF..\n')
    df = baf_denoise(df)
    
    #4. Transform BAF for X
    print(f'{datetime.strftime(datetime.now(),time_format)}\tTransforming BAF..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tTransforming BAF..\n')
    trans_df = transform_BAF(df)
    
    #5. LRR adjust by X chromosome and removing segmental duplications
    print(f'{datetime.strftime(datetime.now(),time_format)}\tAdjusting LRR by X chromosome')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tAdjusting LRR by X chromosome\n')
    X_ploidy, Y_ploidy, trans_df = lrr_adjust(trans_df, baf_model, baf_model_labels, lrr_model, lrr_model_labels, sample_regions = sampling_dicts)
    print(f'\tX copy number: {X_ploidy}\n\tY copy number: {Y_ploidy}')
    log_file.write(f'\tX copy number: {X_ploidy}\n\tY copy number: {Y_ploidy}\n')
    if excl_XY_hom:
        print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving XY homology probes..')
        log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving XY homology probes..\n')
        trans_df = trans_df[trans_df['XY Homology'] == False].copy()
        trans_df.reset_index(inplace = True, drop = True)
        trans_df = transform_BAF(trans_df)
    print(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving probes in segmental duplications..')
    log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tRemoving probes in segmental duplications..\n')
    dups = dup_intervals(segmental_dup_table)
    trans_df = seg_dup_filter(trans_df,dups)
    
    #6. Prediction by BAF and LRR and joint interpretation, followed by low confidence filter
    print(f'{datetime.strftime(datetime.now(),time_format)}\tPredicting CN by BAF and LRR..')
    if target_chrom != 'all':
        c_df = trans_df[trans_df['Chr'] == target_chrom].copy() 
        c_df['LRR_predict'] = lrr_model.predict(c_df[['Adjusted LRR']].values)
        c_df['LRR_predict_label'] = c_df['LRR_predict'].apply(lambda x: lrr_model_labels[x])
        c_df['BAF_predict'] = baf_model.predict(c_df[['Transformed BAF']].values)
        c_df['BAF_predict_label'] = c_df['BAF_predict'].apply(lambda x: baf_model_labels[x])  
        print(f'{datetime.strftime(datetime.now(),time_format)}\tPerforming joint interpretation..')
        log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tPerforming joint interpretation..\n')
        predicts = c_df[['LRR_predict_label','BAF_predict_label']].values
        c_df['CN call'] = [copy_number_calls(L,B) for L,B in predicts]
        output_df = CN_call_cleanup(c_df,dist_limit = 100_000)
    else:
        chrom_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
        output_df = pd.DataFrame()
        for c in chrom_list:
            print(f'{datetime.strftime(datetime.now(),time_format)}\tProcessing Chr{c}')
            log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tProcessing Chr{c}\n')
            c_df = trans_df[trans_df['Chr'] == c].copy() 
            c_df = c_df[~np.isnan(c_df['Adjusted LRR']) & ~np.isnan(c_df['Transformed BAF'])]
            c_df['LRR_predict'] = lrr_model.predict(c_df[['Adjusted LRR']].values)
            c_df['LRR_predict_label'] = c_df['LRR_predict'].apply(lambda x: lrr_model_labels[x])
            c_df['BAF_predict'] = baf_model.predict(c_df[['Transformed BAF']].values)
            c_df['BAF_predict_label'] = c_df['BAF_predict'].apply(lambda x: baf_model_labels[x])
            print(f'\tPerforming joint interpretation..')
            log_file.write(f'\tPerforming joint interpretation..\n')
            predicts = c_df[['LRR_predict_label','BAF_predict_label']].values
            c_df['CN call'] = [copy_number_calls(L,B) for L,B in predicts]
            print(f'\tFiltering low confidence predictions')
            log_file.write(f'\tFiltering low confidence predictions\n')
            c_df = CN_call_cleanup(c_df,dist_limit = 100_000)
            output_df = pd.concat([output_df,c_df],ignore_index = True)
    output_df.reset_index(inplace = True, drop = True)
    output_df.drop(['LRR_predict','BAF_predict'],axis = 1,inplace = True)
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
        f.write(f'# Original_ISCN={iscn}\n')
        N_X, N_Y = int(X_ploidy[0]), int(Y_ploidy[0])
        f.write(f'# Sex_chromosomes={"X" * N_X + "Y" * N_Y}\n')
        f.write(f'# Processed_chromosome={target_chrom}\n')
        f.write('# LRR_HMM_Model:\n')
        f.write(f'## State_labels={list(lrr_model_labels.values())}\n')
        f.write(f'## Means={lrr_model.means_.tolist()}\n')
        f.write(f'## Covars={lrr_model.covars_.tolist()}\n')
        f.write(f'## Transmat=[{lrr_model.transmat_.tolist()}]\n')
        f.write(f'## Start_prob={lrr_model.startprob_.tolist()}\n')
        f.write('# BAF_HMM_Model:\n')
        f.write(f'## State_labels={list(baf_model_labels.values())}\n')
        f.write(f'## Means={baf_model.means_.tolist()}\n')
        f.write(f'## Covars={baf_model.covars_.tolist()}\n')
        f.write(f'## Transmat={baf_model.transmat_.tolist()}\n')
        f.write(f'## Start_prob={baf_model.startprob_.tolist()}\n')
    sum_df = summary_table(output_df, X_ploidy, minimum_probes = min_probes)
    sum_df.to_csv(summary_out_name,mode = 'a',sep = '\t',index = False)
    
    if make_plot:
        plot_name = save_name + '.png'
        print(f'{datetime.strftime(datetime.now(),time_format)}\tPloting results and save as {plot_name}')
        log_file.write(f'{datetime.strftime(datetime.now(),time_format)}\tPloting results and save as {plot_name}\n')
        XY_incl = (not excl_XY_hom)
        plot_prediction(output_df,plot_name, chrom = target_chrom,XY_hom = XY_incl)
    
    log_file.close()

if __name__ == '__main__':
    import os,sys,getopt
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import gzip
    import statistics
    from scipy import stats
    from hmmlearn import hmm
    import pickle
    from matplotlib import pyplot as plt
    from itertools import cycle,islice,groupby
    from datetime import datetime
    main()
    
    
