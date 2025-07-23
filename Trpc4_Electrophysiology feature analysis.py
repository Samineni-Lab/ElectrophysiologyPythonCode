# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:26:13 2025

@author: Owner
"""

# === Core Python Libraries ===
import os
import copy
from datetime import datetime

# === Scientific Computing ===
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

# === Data Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette

# === Machine Learning & Preprocessing ===
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === Electrophysiology Data Handling ===
import pyabf
import efel
#from scipy import nanmean

def find_first_exceeding_index(numbers, threshold):
    for index, value in enumerate(numbers):
        if value > threshold:
            return index
    return None

def find_first_below_index(numbers, threshold, start_index):
    for index in range(start_index, len(numbers)):
        if numbers[index] < threshold:
            return index
    return None

def linier_fit_after_rheobase(df, column, cur_list, rb_index):
    x = np.array(cur_list[rb_index:])
    y = np.array(df[column].values[rb_index:])
    
    # linierfit with only non zero values
    mask = y != 0
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if y_filtered.size > 1: # check more than 2 points
        coefficients = np.polyfit(x_filtered, y_filtered, 1)
        slope = coefficients[0]
        intercept = coefficients[1]       
    elif y_filtered.size == 1:
        slope = y_filtered[0] / x_filtered[0]
        intercept = 0
    else:
        slope = 0
        intercept = 0
        
    return slope, intercept

def return_ind_df_first_value(df, idx):
    row = df.loc[idx]
    first_values = {col: row[col].iloc[0] if isinstance(row[col], pd.Series) else row[col] 
                    for col in row.index}
    return first_values

def return_ind_df_mean_value(df, idx):
    row = df.loc[idx]
    mean_values = {col: row[col].mean() if isinstance(row[col], pd.Series) else row[col] 
                    for col in row.index}
    return mean_values

def return_aft_ind_mean_value(df,ind):
    mean_dic = {}
    for col in df.columns:
        col_mean_list = []
        for value in df[col]:
            if isinstance(value, np.ndarray):
                col_mean_list.append(np.mean(value))  # average array
            else:
                col_mean_list.append(value)  # scolor
        mean_dic[col] = col_mean_list
    mean_dic_aft_ind = {}
    for key, value_list in mean_dic.items():
        if ind < len(value_list):
            mean_dic_aft_ind[key] = np.mean(remove_leading_zeros(value_list[ind:]))  # average after index with removing initial 0
        else:
            mean_dic_aft_ind[key] = 0
    return mean_dic_aft_ind
       
def remove_leading_zeros(lst):
    # count 0 from start
    count = 0
    for value in lst:
        if value == 0:
            count += 1
        else:
            break  # break loop if not 0
    list_wo_zero = lst[count:]
    if len(list_wo_zero) == 0:
        list_wo_zero = [0]
    return list_wo_zero

def min_index_of_max_value(lst):
    max_value = max(lst)
    min_index = lst.index(max_value)
    return min_index

#%% Setup parameters and directory

feature_list = ['ADP_peak_amplitude',
                'AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AHP_depth_diff', 'AHP_depth', 'AHP_depth_slow', 'AHP_slow_time', 'AHP_time_from_peak',
                'AP1_amp', 'AP1_begin_voltage', 'AP1_width', 'AP2_AP1_begin_width_diff', 'AP2_AP1_diff', 'AP2_amp', 'AP2_begin_voltage', 'AP2_width', 'APlast_amp', 'APlast_width',
                'AP_amplitude', 'AP_amplitude_change', 'AP_begin_voltage',
                'AP_fall_rate_change', 'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change', 'time_to_first_spike', 'time_to_last_spike',
                'irregularity_index', 'number_initial_spikes', 'adaptation_index2', 'ISI_values', 'ISI_CV', 'doublet_ISI', 'inv_second_ISI', 'inv_last_ISI', 'ISI_log_slope_skip',
                'spike_count_stimint','spike_width2', 'phaseslope_max', 'AP_phaseslope',
                'decay_time_constant_after_stim', 'depolarized_base', 'maximum_voltage_from_voltagebase', 'voltage_after_stim', 'voltage_base',
                'postburst_adp_peak_values', 'postburst_fast_ahp_values', 'postburst_slow_ahp_values',
                'interburst_voltage', 'initburst_sahp_vb', 'strict_burst_number', 'strict_burst_mean_freq', 'single_burst_ratio']

feature_list_sub = ['sag_amplitude', 'sag_ratio1', 'sag_time_constant', 'ohmic_input_resistance_vb_ssse', 'time_constant', 'decay_time_constant_after_stim',
                    'voltage_after_stim', 'voltage_base', 'voltage_deflection_vb_ssse']

feature_rheo_for_analys = [
                
                'AP1_amp', 'AP1_begin_voltage', 'AP1_width', 'AP_peak_downstroke', 'AP_peak_upstroke', 'time_to_first_spike',
                'phaseslope_max', 'AP_phaseslope']

feature_rheo_W_for_analys = [
                
                'AP1_amp', 'AP1_begin_voltage', 'AP1_width', 'AP2_AP1_begin_width_diff', 'AP2_AP1_diff', 'AP2_amp', 'AP2_begin_voltage', 'AP2_width', 'APlast_amp', 'APlast_width',
                'AP_fall_rate_change', 'AP_peak_downstroke', 'AP_peak_upstroke', 'AP_rise_rate_change', 'time_to_first_spike', 'time_to_last_spike',
                'number_initial_spikes', 'ISI_values',
                'spike_count_stimint','spike_width2', 'phaseslope_max', 'AP_phaseslope']

feature_mean_avb_rheo_for_analys =  ['ADP_peak_amplitude',
                 'AHP_slow_time', 'AHP_time_from_peak',
                'AP_amplitude', 'AP_amplitude_change', 'AP_begin_voltage',
                'AP_fall_rate_change', 'AP_peak_downstroke', 'AP_peak_upstroke',
                'irregularity_index', 'adaptation_index2',
                'spike_width2', 'phaseslope_max', 'AP_phaseslope', 'ISI_CV', 'doublet_ISI', 'ISI_log_slope_skip',
                'depolarized_base',  'voltage_after_stim', 
                'postburst_adp_peak_values', 'postburst_fast_ahp_values', 'postburst_slow_ahp_values',
                'interburst_voltage',  'strict_burst_number', 'strict_burst_mean_freq', 'single_burst_ratio']

feature_ratio_blw_abv_rheo = ['decay_time_constant_after_stim']

feature_spike_decay = ['spike_count_stimint']

change_ratio = ['AP_amplitude_change','AP_rise_rate_change']

change_abs = ['AHP_depth_diff','AP2_AP1_begin_width_diff','AP2_AP1_diff']

need_new = ['first_second_ISI_ratio', 'max_spike_current_ratio_to_max_current']

feature_sub_first_swp = ['sag_ratio1', 'sag_time_constant']

feature_sub_mean_all = ['ohmic_input_resistance_vb_ssse', 'time_constant', 'decay_time_constant_after_stim']



directory = r"/Users/yufenzhang/Library/CloudStorage/Box-Box/EP data/patch Seq/Cell_Collected_EphysFiles_cluster5_20250716"
extension = '.abf'
seq_info_path = r"/Users/yufenzhang/Library/CloudStorage/Box-Box/EP data/patch Seq/Python_data/20250715_Mata_Info_For_tSNE.xlsx"

# change everytime of the "test output" name to create a new folder for all the files
output_directory = r'C:\Users\Owner\Box\patch Seq\Python_data/Test_output2'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created folder: {output_directory}")
    

#%% Load data from .abf files 
data_summary = pd.DataFrame()

group_num = 0
group_list = {}

dvdt_threshold = 20  # Derivative threshold

for child_directory in os.listdir(directory):
    child_path = os.path.join(directory, child_directory)
    group_num += 1
    
    if os.path.isdir(child_path):
        group_list[child_directory] = group_num
        file_list = [filename for filename in os.listdir(child_path) if filename.endswith(extension)]
        
        for file in file_list:
            file_path = os.path.join(child_path, file)
            try:
                # Prin current file name for debug
                print(f"Processing file: {file}")

                # load .abf files
                abf = pyabf.ABF(file_path)

            
                # identifying current stim starts
                list_current = []
            
                for swp in abf.sweepList:
                    abf.setSweep(swp)
                    current = abf.sweepC
                    list_current.append(current)
           
                    current_array = np.array(list_current)
                    current_std_devs = np.std(current_array, axis=0)

                c_threshold = 1
                start_index = find_first_exceeding_index(current_std_devs, c_threshold)
                if start_index is not None:
                    end_index = find_first_below_index(current_std_devs, c_threshold, start_index)
                    window_start = start_index*1000/abf.dataRate
                    window_end = end_index*1000/abf.dataRate
                else:
                    end_index = None
                window_start = 0
                window_end = 1000
             
            # find Zero current sweep
                nparray_list_current = np.array(list_current)
                nparray_list_current_ave = np.mean(nparray_list_current, axis=1)
                nparray_list_current_ave_abs = np.abs(nparray_list_current_ave)
                zero_sweep_index = np.unravel_index(np.argmin(nparray_list_current_ave_abs), nparray_list_current_ave.shape)
                zero_sweep_index_int = zero_sweep_index[0]
                
                # for pool data for each sweep      # 初始化 DataFrame 来存储特征
                feature_frame = pd.DataFrame(columns=feature_list)
                feature_frame_sub = pd.DataFrame(columns=feature_list_sub)
                
                # Current step values
                currents = [] # Current value between t1 and t2 (ms) for each step
                t1 = int(window_start*abf.dataPointsPerMs) 
                t2 = int(window_end*abf.dataPointsPerMs)
                
                vm_max_list = []
                vm_base_list = []
                
                # determine the AP threshold   
                for sweepNumber in abf.sweepList:
                    #sweepNumber= 7 # 0,1,2,,,
                    sweep = abf.setSweep(sweepNumber=sweepNumber, channel=0)
                    
                    time = abf.sweepX*1000  # in miliseconds
                    voltage = abf.sweepY  # or filtered voltage
                    current = abf.sweepC
                    
                    current_mean = np.average(abf.sweepC[t1:t2])
                    currents.append(current_mean)
                    
                    trace = {'T': abf.sweepX*1000, 
                             'V': abf.sweepY,
                             'I': abf.sweepC,
                             'stim_start': [window_start],
                             'stim_end': [window_end],
                             'stimulus_current' : [current_mean],
                             'AP_phaseslope_range' : [2]} 
                    traces = [trace]
                    
                    vm_max = np.max(trace['V'])
                    vm_base = np.mean(trace['V'][:100])
                    vm_max_list.append(vm_max)
                    vm_base_list.append(vm_base)
                    
                max_voltage_max = max(vm_max_list)
                mean_voltage_base = sum(vm_base_list) / len(vm_base_list)
                
                spike_threshold = (3*max_voltage_max + 2*mean_voltage_base)/5 # set the threshold at 3/5 between max and base
                
                # main
                for sweepNumber in abf.sweepList:
                    #sweepNumber= 7 # 0,1,2,,,
                    sweep = abf.setSweep(sweepNumber=sweepNumber, channel=0)
                    
                    time = abf.sweepX*1000  # in miliseconds
                    voltage = abf.sweepY  # or filtered voltage
                    current = abf.sweepC
                    
                    current_mean = np.average(abf.sweepC[t1:t2])
                    currents.append(current_mean)
                    
                    trace = {'T': abf.sweepX*1000, 
                             'V': abf.sweepY,
                             'I': abf.sweepC,
                             'stim_start': [window_start],
                             'stim_end': [window_end],
                             'stimulus_current' : [current_mean],
                             'AP_phaseslope_range' : [2]} 
                    traces = [trace]
                    
                    # Detection parameters use efel 
                    efel.api.set_setting('Threshold', spike_threshold)  # Voltage threshold 
                    efel.api.set_setting('DerivativeThreshold', dvdt_threshold) # dV/dt threshold 
                    efel.api.set_setting('strict_stiminterval', True) # limited period to analysis
                    
                    # Define the output features
   
                    if sweepNumber < zero_sweep_index_int:
                        feature_values = efel.get_feature_values(traces,feature_list_sub,raise_warnings=None)[0]                                    
                    else:
                        feature_values = efel.get_feature_values(traces,feature_list,raise_warnings=None)[0]
                        
                    copy_dict = copy.deepcopy(feature_values) # for debug
                                                               
                    for key, value in feature_values.items():
                        if value is None:
                            feature_values[key] = 0 # covert None to 0
                        elif np.isnan(value).any():
                            feature_values[key] = 0 # some feature returns np.nan
                        elif len(value) == 0:
                            feature_values[key] = 0 # some feature returns []
                        else:
                            feature_array = np.array(value)
                            #feature_values[key] = np.mean(feature_array) # Average
                            feature_values[key] = feature_array
                            
                    new_row = pd.DataFrame([feature_values])
                    
                    # modify values
                    if sweepNumber >= zero_sweep_index_int:
                        # voltage_base
                        if new_row['depolarized_base'].dtype == 'object':
                            new_row['depolarized_base'] = new_row['depolarized_base'].mean().mean() - new_row['voltage_base']
                        # voltage_after_stim
                        new_row['voltage_after_stim'] -= new_row['voltage_base']
                        #doublet ISI -> doublet ISI / mean(ISI values)
                        if new_row['doublet_ISI'].dtype == 'object':
                            if new_row['ISI_values'].mean().mean() == 0:
                                new_row['ISI_first_second_ratio'] = [0]
                                new_row['doublet_ISI'] = [1]
                            else:
                                new_row['ISI_first_second_ratio'] = new_row['ISI_values'][0][0] / new_row['doublet_ISI']
                                new_row['doublet_ISI'] = new_row['doublet_ISI'] / new_row['ISI_values'].mean().mean()
                        else:
                            new_row['ISI_first_second_ratio'] = [0]
                        
                    if sweepNumber < zero_sweep_index_int:
                        feature_frame_sub = pd.concat([feature_frame_sub, new_row], ignore_index=True)
                    else:
                        feature_frame = pd.concat([feature_frame, new_row], ignore_index=True)
                        feature_frame = feature_frame.apply(pd.to_numeric, errors='ignore')
                        
                # rheobase
                positive_mask = feature_frame['spike_count_stimint'].gt(0)
                if positive_mask.any():
                    rheobase_index = positive_mask.idxmax()
                else:
                    rheobase_index = 0
                    
                rheobase_index_W = min(rheobase_index*2, len(feature_frame)-1)
                if rheobase_index*2>(len(feature_frame)-1):
                    print(f'{file} requires more sweep to analyse w*rheobase; w*rheobase was set at max sweep')
                
                rheobase = currents[rheobase_index + zero_sweep_index_int]
                
                
                feature_summary = {'rheobase' : rheobase}
                
                rheo_feature = return_ind_df_first_value(feature_frame, rheobase_index)
                for col in feature_rheo_for_analys:
                    if isinstance(rheo_feature[col], np.ndarray):
                        feature_summary[f'{col}_rheo'] = float(rheo_feature[col][0])
                    else:
                        feature_summary[f'{col}_rheo'] = rheo_feature[col]
                        
                rheo_W_feature = return_ind_df_mean_value(feature_frame, rheobase_index_W)
                for col in feature_rheo_W_for_analys:
                    if isinstance(rheo_W_feature[col], np.ndarray):
                        feature_summary[f'{col}_rheo_W'] = float(rheo_W_feature[col][0])
                    else:
                        feature_summary[f'{col}_rheo_W'] = rheo_W_feature[col]
                        
                all_swp_feature = return_aft_ind_mean_value(feature_frame, rheobase_index+1)
                for col in feature_mean_avb_rheo_for_analys:
                    if isinstance(all_swp_feature[col], np.ndarray):
                        feature_summary[f'{col}_aft_rheo'] = float(all_swp_feature[col][0])
                    else:
                        feature_summary[f'{col}_aft_rheo'] = all_swp_feature[col]
                        
                sub_feature = {
                                col: np.mean([val[0] if isinstance(val, list) or isinstance(val, np.ndarray) else val for val in feature_frame_sub[col]])
                                for col in feature_frame_sub.columns
                            }
                    
                sub_feature_sag = feature_frame_sub.iloc[0]

                #max spike loc
                spike_count_df = pd.DataFrame(feature_frame['spike_count_stimint'])
                spike_count_list = [item[0] for item in spike_count_df['spike_count_stimint']]
                max_spike_loc = min_index_of_max_value(spike_count_list)
                feature_summary['max_spike_loc_ratio'] = max_spike_loc / len(feature_frame)
                
                    
                feature_summary['file_name'] = file
                feature_summary['group'] = group_num
                feature_summary['group_name'] = child_directory
                    
                new_data = pd.DataFrame([feature_summary])
                data_summary = pd.concat([data_summary, new_data], ignore_index=True)
                
                print(f"Processing {file} completed successfully")  # File process sucess
    
            except Exception as e:
                            # Catch error file and jump to next file
                            print(f"Error processing file {file}: {e}")
                            continue
#%%
data_df = data_summary.iloc[:,:-3]
labels = data_summary['file_name']
labels_list = labels.tolist()
nan_rows = data_summary[data_summary.isna().any(axis=1)]
nan_indices = nan_rows.index.tolist()
if len(nan_indices) == 0:
    standarded_data = StandardScaler().fit_transform(data_df) 
    
#%% dendrogram

linkage_matrix = linkage(standarded_data, method='ward')

cluster_range = []
threshold_range = 100
for i in range(threshold_range):
    clusters = fcluster(linkage_matrix, i, criterion='distance')
    num_clusters = len(set(clusters))
    cluster_range.append(num_clusters)
increment = [cluster_range[i-1] - cluster_range[i] for i in range(1, len(cluster_range))]
increment.insert(-1, 0)

plt.figure(figsize=(10, 6))
plt.plot(range(threshold_range), cluster_range, marker='o', linestyle='-',color='blue', label='Cluster Range')
plt.plot(range(threshold_range), increment, marker='x', linestyle='-',color='y', label='Increment')
plt.gca().invert_xaxis()
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Number of Clusters')
plt.ylim(0, 50)
plt.title('Number of Clusters vs Threshold')
plt.grid(True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  
output_file_name = f'Cluster_Threshold_{timestamp}.pdf'
output_file_path = os.path.join(output_directory, output_file_name)
plt.savefig(output_file_path, format='pdf', bbox_inches='tight')
plt.show()

#%% Manually set dendrogram groups
den_threshold = 16
 #adjust it after check "Number of Clusters vs Threshold" plot!!!!

#%% dendrogram cont.
set_link_color_palette(None)
fig, ax = plt.subplots(figsize=(20, 12), dpi=300)

dendrogram(linkage_matrix, labels=labels_list, 
           leaf_rotation=90,  # file name rotation degrees
           leaf_font_size=10,
           color_threshold=den_threshold)

plt.title('Circular Dendrogram')
plt.tight_layout()

clusters = fcluster(linkage_matrix, den_threshold, criterion='distance')  # Adjust 'threshold' as needed
clusters_series = pd.Series(clusters, name='Cluster')

data_summary['dendrogram_cluster'] = clusters_series

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
output_file_name = f'Dendrogram_{timestamp}.pdf'
output_file_path = os.path.join(output_directory, output_file_name)
plt.savefig(output_file_path, format='pdf', bbox_inches='tight')
plt.show()

#%% Heatmap: This plot not save
sort_column = 'group'
#'group' manually classified
#'dendrogram_cluster' classification based on dendrogram 

sorted_indices = data_summary[sort_column].argsort()
sorted_values = data_summary[sort_column].sort_values().values
sorted_standarded_data = standarded_data[sorted_indices]
sorted_standarded_data_df = pd.DataFrame(sorted_standarded_data, columns=data_df.columns)

df = sorted_standarded_data_df.T
df.index = sorted_standarded_data_df.columns 

group_labels =['Initial','Single',
               'Gap', 'Tonic',
                'Phasic',
               'Delayed']
group_ranges = []
for group in group_labels:
    group_indices = data_summary[data_summary['group_name'] == group].index
    group_ranges.append((group_indices.min(), group_indices.max()))


clustergrid = sns.clustermap(df,
                             cmap='GnBu',
                vmin=-1.5,
                vmax=3,
                figsize=(25, 8),
                col_cluster=False,
                cbar_pos=None,
                dendrogram_ratio=(0.1, 0.2),
                xticklabels=False, 
                yticklabels=True)

# Divider between each group
for start, end in group_ranges:
    clustergrid.ax_heatmap.axvline(x=start, color='black', linewidth=2)
    clustergrid.ax_heatmap.axvline(x=end + 1, color='black', linewidth=2)

# Add group label
for i, (start, end) in enumerate(group_ranges):
    clustergrid.ax_heatmap.text((start + end) / 2, -0.5, group_labels[i], ha='center', va='bottom', fontsize=11, fontweight='bold')

# Get the reordered features (row clustering order)
reordered_rows = clustergrid.dendrogram_row.reordered_ind
reordered_features = sorted_standarded_data_df.columns[reordered_rows]

# Add a border around the heatmap
for spine in clustergrid.ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
    
clustergrid.ax_heatmap.set_xticklabels([])
plt.colorbar(clustergrid.ax_heatmap.collections[0], ax=clustergrid.ax_heatmap, orientation='vertical', fraction=0.05, pad=0.15,shrink=0.6)

plt.show()
# def mean_of_arrays(arrays):
#     return np.mean(np.stack(arrays), axis=0)

class_indices = {cls: np.where(data_summary[sort_column] == cls)[0] for cls in data_summary[sort_column].unique()}
class_means = {cls: np.mean(standarded_data[indices], axis=0) for cls, indices in class_indices.items()} # not sorted data because class indices are base on not sorted data
class_means_array = np.array(list(class_means.values()))

unique_class_tick = data_summary[sort_column].unique()

plt.figure(figsize=(15, 3),dpi=200)
ax = sns.heatmap(class_means_array, vmin=-1, vmax=3,cmap='GnBu', cbar_kws={"shrink": 0.8, "aspect": 20, "pad": 0.02}, annot=False)
ax.set_yticks(np.arange(len(unique_class_tick))+0.5)
ax.set_yticklabels(unique_class_tick, rotation=0, fontsize=6)
ax.set_xticks(np.arange(len(data_df.columns))+0.5)
ax.set_xticklabels(data_df.columns, rotation=90, fontsize=6)
plt.title('mean of each class')
plt.show()

#%% p-value plots

sort_column = 'group' # use 'group' or 'dendrogram_cluster'

# Step 1: Sort data_summary first
data_summary_sorted = data_summary.sort_values(by=sort_column).reset_index(drop=True)

# Step 2: Extract numeric data and standardize
numeric_data = data_summary_sorted.select_dtypes(include='number')
numeric_data = numeric_data.drop(columns=['group', 'dendrogram_cluster'])
sorted_standarded_data = StandardScaler().fit_transform(numeric_data)

# Step 3: Get class array and class groups
class_array = data_summary_sorted[sort_column].values
unique_class_tick = np.unique(class_array)
class_data = {cls: sorted_standarded_data[class_array == cls] for cls in unique_class_tick}

# Step 4: Compute p-values
anova_results = []
kruskal_results = []

for i in range(sorted_standarded_data.shape[1]):
    # if some parameter are same, skip stat test 
    if len(np.unique(sorted_standarded_data[:,i])) == 1:
        continue
    
    class_arrays = [class_data[cls][:, i] for cls in unique_class_tick]
    _, p1 = stats.f_oneway(*class_arrays)
    _, p2 = stats.kruskal(*class_arrays)
    anova_results.append(p1)
    kruskal_results.append(p2)

# Step 5: Plot
p_values_matrix = np.array([anova_results, kruskal_results])
stat_tick = ['ANOVA', 'Kruskal_W']
columns = numeric_data.columns

plt.figure(figsize=(20, 3), dpi=200)
ax = sns.heatmap(p_values_matrix, annot=False, fmt=".3f", cmap="autumn",
                 cbar_kws={"label": "p-value"},
                 mask=(p_values_matrix > 0.05), linewidths=.5)
ax.set_yticks(np.arange(len(stat_tick)) + 0.5)
ax.set_yticklabels(stat_tick, rotation=90, fontsize=9)
ax.set_xticks(np.arange(len(columns)) + 0.5)
ax.set_xticklabels(columns, rotation=90, fontsize=6)

for (i, j), val in np.ndenumerate(p_values_matrix):
    ax.text(j + 0.5, i + 0.5, f"{val:.3f}", ha='center', va='center', fontsize=6, rotation=90)

# Save and show
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_file_name = f'statitics_{timestamp}.png'
output_file_path = os.path.join(output_directory, output_file_name)
plt.savefig(output_file_path, format='png')
print(f"File saved as {output_file_path}")
plt.show()


#%% tsne
def merge_df(df1, df2, left_on=[], right_on=[]):
    if len(left_on) == 0 or len(right_on) == 0:
        raise ValueError("No corresponding column name provided for merging （left_on or right_on）")
    new_df = pd.merge(
        df1,      
        df2,      
        how='left',  
        left_on=left_on, 
        right_on=right_on
    )

    return new_df


# This is the data you want to append
seq_info = pd.read_excel(seq_info_path)
merged_data = merge_df(data_summary, seq_info,
                       left_on=['file_name', 'group_name'],
                       right_on=['file_name', 'label'])

merged_data = merged_data.iloc[:, :-2] # drop last two nan columns in seq info


merged_data = merged_data.sort_values(by='file_name').reset_index(drop=True)
# t-SNE data prepare
label = merged_data['group_name']  # use 'group' or 'dendrogram_cluster'
'''
data = merged_data.iloc[:, :-6]  
'''
data = merged_data.select_dtypes(include='number')
data = data.drop(columns=['group', 'dendrogram_cluster'])

seq_status = merged_data['Seq']


# Check the number of samples in each group
print("Group counts:\n", label.value_counts())


standarded_data = StandardScaler().fit_transform(data)
model_tsne = TSNE(n_components=2, random_state=501, perplexity=18, learning_rate=200, 
                  n_iter=4000, early_exaggeration=15, init='pca')
tsne_data = model_tsne.fit_transform(standarded_data)

# Normalized t-SNE results
x_min, x_max = tsne_data.min(0), tsne_data.max(0)
tsne_data = (tsne_data - x_min) / (x_max - x_min)

# Create a data frame containing t-SNE results and labels
tsne_df = pd.DataFrame(data=tsne_data, columns=["Dim1", "Dim2"])
tsne_df['label'] = label.values
tsne_df['file_name'] = merged_data['file_name'].values
tsne_df['Seq'] = seq_status.values

# Colors
unique_labels = label.unique()

colors = {
    "#A6761D",
    "#D95F02",       
   "#7570B3",   
   "#E7298A",     
   "#66A61E"}

color_map = dict(zip(unique_labels, colors))  # Assign group names to colors

plt.figure(figsize=(6, 6), dpi=100)

for unique_label in unique_labels:
    label_mask = tsne_df['label'] == unique_label
    seq_mask = tsne_df['Seq'] == 'Y'  # Highlight Seq == 'Y' 
    
    plt.scatter(tsne_df['Dim1'][label_mask ], tsne_df['Dim2'][label_mask],
                s=50, label=f"{unique_label}", c=color_map[unique_label], alpha=1.0)
    
    plt.scatter(tsne_df['Dim1'][label_mask & ~seq_mask], tsne_df['Dim2'][label_mask & ~seq_mask],
                s=50, c=color_map[unique_label], alpha=1.0)

plt.legend(loc='best')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Clustering Visualization ')


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

output_pdf = os.path.join(output_directory, f'tSNE_Seq_Clustering_{timestamp}.pdf')
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"t-SNE image saved as: {output_pdf}")

output_excel = os.path.join(output_directory, f'tSNE_Seq_Coordinates_{timestamp}.xlsx')
with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    tsne_df.to_excel(writer, sheet_name='tSNE_Coordinates', index=False)
    print(f"Seq t-SNE coordinates saved as: {output_excel}")


plt.show()


#%% Optional Step: Drop all non significant column for tsne by p-value

high_pvalue_cols = np.where((p_values_matrix > 0.05).any(axis=0))[0]
high_pvalue_colnames = data_summary.columns[high_pvalue_cols]
merged_data = merged_data.sort_values(by='file_name').reset_index(drop=True)
drop_non_significant = merged_data.drop(columns=high_pvalue_colnames, errors='ignore')



label = drop_non_significant['group_name']  

data = drop_non_significant.select_dtypes(include='number')
data = data.drop(columns=['group', 'dendrogram_cluster'])
seq_status = drop_non_significant['Seq']

# Check the number of samples in each group
print("Group counts:\n", label.value_counts())


standarded_data = StandardScaler().fit_transform(data)
model_tsne = TSNE(n_components=2, random_state=501, perplexity=18, learning_rate=200, 
                  n_iter=4000, early_exaggeration=15, init='pca')
tsne_data = model_tsne.fit_transform(standarded_data)

# Normalized t-SNE results
x_min, x_max = tsne_data.min(0), tsne_data.max(0)
tsne_data = (tsne_data - x_min) / (x_max - x_min)

# Create a data frame containing t-SNE results and labels
tsne_df = pd.DataFrame(data=tsne_data, columns=["Dim1", "Dim2"])
tsne_df['label'] = label.values
tsne_df['file_name'] = merged_data['file_name'].values
tsne_df['Seq'] = seq_status.values

# Colors
unique_labels = label.unique()

colors = {
    "#A6761D",
    "#D95F02",       
   "#7570B3",   
   "#E7298A",     
   "#66A61E"}

color_map = dict(zip(unique_labels, colors))  # Assign group names to colors

plt.figure(figsize=(6, 6), dpi=100)

for unique_label in unique_labels:
    label_mask = tsne_df['label'] == unique_label
    seq_mask = tsne_df['Seq'] == 'Y'  # Highlight Seq == 'Y' 
    
    plt.scatter(tsne_df['Dim1'][label_mask ], tsne_df['Dim2'][label_mask],
                s=50, label=f"{unique_label}", c=color_map[unique_label], alpha=1.0)
    
    plt.scatter(tsne_df['Dim1'][label_mask & ~seq_mask], tsne_df['Dim2'][label_mask & ~seq_mask],
                s=50, c=color_map[unique_label], alpha=1.0)

plt.legend(loc='best')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Clustering Visualization ')


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

output_pdf = os.path.join(output_directory, f'tSNE_Seq_Clustering_wo_nonsignificant_{timestamp}.pdf')
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"t-SNE image saved as: {output_pdf}")

output_excel = os.path.join(output_directory, f'tSNE_Seq_Coordinates_wo_nonsignificant_{timestamp}.xlsx')
with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    tsne_df.to_excel(writer, sheet_name='tSNE_Coordinates', index=False)
    print(f"Seq t-SNE coordinates saved as: {output_excel}")


plt.show()

#%% save data summary

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
xlsx_filename = f'data_summary_{timestamp}.xlsx'
xlsx_file_path = os.path.join(output_directory, xlsx_filename)
data_summary.to_excel(xlsx_file_path, index=False)

