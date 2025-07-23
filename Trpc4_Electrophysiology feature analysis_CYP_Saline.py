# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:40:43 2025

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



directory = r"/Users/yufenzhang/Library/CloudStorage/Box-Box/EP data/trpc4 project/CYP_NS_FiringPattern/Ephys_files_Code_extract/5_compare by each firing pattern_no initial"
extension = '.abf'
seq_info_path = r"C:\Users\Owner\Box\patch Seq\Python_data\20250715_Mata_Info_For_tSNE.xlsx"


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
            
#%% rearrange group numbers
hardcode_order = {"saline_Single": 1,
                  "CYP_Phasic": 10,
                  "saline_Delayed": 5,
                  "CYP_Delayed": 6,
                  "saline_Gap": 3,
                  "CYP_Gap": 4,
                  "saline_Tonic":7,
                  "CYP_Tonic":8, 
                  "saline_Phasic": 9,
                  "CYP_Single":2}
# In this data excel, there are some nan value (may not happened in other data)
data_summary['group_name'] = data_summary['group_name'].fillna('CYP_Single')
# Assign group names and group number and sort by groupo number
data_summary['group'] = data_summary['group_name'].map(hardcode_order)
data_summary = data_summary.sort_values(by='group').reset_index(drop=True)


#%% generate standard data for plots
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


#%% Heatmap

def extract_features_by_p_value(data, top_n_smallest=10): # choose top 10 smallest p value from each group

    feature_data = data.iloc[:, :-4]
    labels = data['group']
    
    significant_features = []
    for group in range(1, len(np.unique(labels))+1, 2):
        
        saline_data = feature_data[labels == group]
        cyp_data = feature_data[labels == group+1]
        
        t_p_values = []
        valid_features = []
        
        for col in feature_data.columns:
            group1 = saline_data[col].dropna()
            group2 = cyp_data[col].dropna()
        
            # skip too small data set or same data
            if len(group1) < 2 or len(group2) < 2:
                continue
            if np.var(group1) == 0 or np.var(group2) == 0:
                continue
        
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test 更稳健
            t_p_values.append(p_val)
            valid_features.append(col)
  ## choose the significant P value          
        # ttest_results_df = pd.DataFrame({
        #     'feature': valid_features,
        #     'p_value': t_p_values
        # })
        # significant_features_df = ttest_results_df[ttest_results_df['p_value'] < 0.05]
        # significant_features.append(significant_features_df)
 ##  choose the significant P value      
        ttest_results_df = pd.DataFrame({
            'feature': valid_features,
            'p_value': t_p_values
        })
        
        # select smallest n p-values
        significant_features_df = ttest_results_df.sort_values(by='p_value').head(top_n_smallest)
        
        significant_features.append(significant_features_df)
        
    
    all_significant_feature_names = set()
    
    for df in significant_features:
        all_significant_feature_names.update(df['feature'].tolist())
    
    all_significant_feature_names = list(all_significant_feature_names)
    
    print(all_significant_feature_names)
    top_var_data = feature_data[all_significant_feature_names]
    return top_var_data



# Set RGB Color for heat map（236, 130, 16） into range (0,1)
orange_rgb = (236/255, 130/255, 16/255)
custom_cmap = LinearSegmentedColormap.from_list("custom_orange", ["white", orange_rgb])


labels = data_summary['file_name']
group_labels =['saline_Initial','CYP_Initial','saline_Single','CYP_Single',
               'saline_Gap', 'CYP_Gap','saline_Tonic','CYP_Tonic', 'saline_Phasic',
               'CYP_Phasic', 'saline_Delayed','CYP_Delayed']


'''
If you want to plot to n smallest p-value parameteres use following line:
top_var_data = extract_features_by_p_value(data_summary, top_n_smallest=10)

Otherwise to plot all features use:
top_var_data = data_df
'''
top_var_data = extract_features_by_p_value(data_summary, top_n_smallest=10)
# top_var_data = data_df


scaler = StandardScaler()
standarded_data_HM_Cluster = scaler.fit_transform(top_var_data)
standarded_data_HM_Cluster_df = pd.DataFrame(standarded_data_HM_Cluster, columns=top_var_data.columns)


group_ranges = []
for group in group_labels:
    group_indices = data_summary[data_summary['group_name'] == group].index
    group_ranges.append((group_indices.min(), group_indices.max()))



clustergrid = sns.clustermap(
    standarded_data_HM_Cluster_df.T,
    cmap= custom_cmap , # Set color to heatmap
    vmin=-1.5, # Min of color bar
    vmax=1.5, # Max of color bar
    figsize=(25, 8), #(w, h)
    col_cluster=False,
    cbar_pos=None,
    dendrogram_ratio=(0.1, 0.2),
    xticklabels=False, 
    yticklabels=True
)

# Divider between each group
for start, end in group_ranges:
    clustergrid.ax_heatmap.axvline(x=start, color='black', linewidth=2)
    clustergrid.ax_heatmap.axvline(x=end + 1, color='black', linewidth=2)

# Add group label
for i, (start, end) in enumerate(group_ranges):
    clustergrid.ax_heatmap.text((start + end) / 2, -0.5, group_labels[i], ha='center', va='bottom', fontsize=11, fontweight='bold')

# Get the reordered features (row clustering order)
reordered_rows = clustergrid.dendrogram_row.reordered_ind
reordered_features = standarded_data_HM_Cluster_df.columns[reordered_rows]

# Add a border around the heatmap
for spine in clustergrid.ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
    

clustergrid.ax_heatmap.set_xticklabels([])
plt.colorbar(clustergrid.ax_heatmap.collections[0], ax=clustergrid.ax_heatmap, orientation='vertical', fraction=0.05, pad=0.15,shrink=0.6)
   

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file_name = f'reordered_data_summary_HeatMap_{timestamp}.xlsx'
output_file_path = os.path.join(output_directory, output_file_name)

with pd.ExcelWriter(output_file_path) as writer:
    data_summary.to_excel(writer, sheet_name='Reordered Data')
    pd.DataFrame(reordered_features, columns=['Features']).to_excel(writer, sheet_name='Reordered Features')

output_file_name = f'heatmap_FeatureCluster_{timestamp}.pdf'
output_file_path = os.path.join(output_directory, output_file_name)
plt.savefig(output_file_path, format='pdf', bbox_inches='tight')

plt.show()





#%% save data summary

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
xlsx_filename = f'data_summary_{timestamp}.xlsx'
xlsx_file_path = os.path.join(output_directory, xlsx_filename)
data_summary.to_excel(xlsx_file_path, index=False)






