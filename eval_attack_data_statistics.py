# Evaluate the static indicators of the attacked datasets, and perform statistical analysis


import argparse
import json
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Set the font to a similar Times New Roman serif font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIXGeneral']

# Global font size increase
plt.rcParams['font.size'] = 18          # Default font size
plt.rcParams['axes.titlesize'] = 18     # Title font size
plt.rcParams['axes.labelsize'] = 20     # Axis label font size
plt.rcParams['xtick.labelsize'] = 18    # x-axis tick font size
plt.rcParams['ytick.labelsize'] = 18    # y-axis tick font size
plt.rcParams['legend.fontsize'] = 18    # Legend font size

def query_attack_main(draw_pic=True): # Evaluate the static indicators of the attacked datasets, and perform statistical analysis

    model_list = [
    "contriever", 
    "bge_m3",  
    "qwen3", 
    "linq",
    "gte",
    "reasonir",  
    "diver",
    "bge_reasoner",
    "qwen3_4B",
    "qwen3_0.6B",
    ]
    
    # Model display name mapping
    model_display_names = {
        "contriever": "Contriever",
        "bge_m3": "BGE-M3",
        "qwen3": "Qwen3",
        "linq": "Linq",
        "gte": "GTE",
        "reasonir": "ReasonIR",
        "diver": "DIVER",
        "bge_reasoner": "ReasonEmbed",
        "qwen3_4B": "Qwen3-4B",
        "qwen3_0.6B": "Qwen3-0.6B",
    }

    query_attack_method_list = [
        "none",
        "mispelling",
        "ordering",
        "synonym",
        "paraphrase",
        "naturality",
    ]
    
    # Attack method display name mapping
    attack_method_display_names = {
        "mispelling": "Misspelling",
        "ordering": "Reordering",
        "synonym": "Synonymizing",
        "paraphrase": "Paraphrasing",
        "naturality": "Naturalizing",
    }
    
    # Attack methods except none (for plotting)
    attack_methods_for_plot = [m for m in query_attack_method_list if m != "none"]

    seed_list = [1999, 5, 27, 2016, 2026]

    dataset_list = [
        # "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "msmarco",
    ]

    dataset_display_names = {
        "nq": {"test": "NQ",},
        "hotpotqa": {"test": "HotpotQA"},
        "nfcorpus": {"test": "NFCorpus"},
        "fiqa": {"test": "FiQA"},
        "msmarco": {"dev": "MS MARCO (Dev)",}, #"trec_dl19": "TREC DL 2019", "trec_dl20": "TREC DL 2020"
    }

    # Create the folder to save the pictures
    os.makedirs("figs", exist_ok=True)

    # ========== First step: Collect all the data of the datasets ==========
    all_data = {}  # {f"{dataset}_{split}": {model: {attack_method: (mean, std)}}}
    all_metrics = {}  # {f"{dataset}_{split}": target_metric}
    
    # Reverse mapping: display name -> model name
    display_to_model = {v: k for k, v in model_display_names.items()}
    # Reverse mapping: display name -> attack method name
    display_to_attack = {v: k for k, v in attack_method_display_names.items()}
    
    for dataset in dataset_list:
        if dataset == 'msmarco':
            split_list = ['dev', ] # 'trec_dl19', 'trec_dl20'
        else:
            split_list = ['test']

        for split in split_list:
            # if dataset == 'msmarco' and split == 'dev':
            #     target_metric = 'MRR@10'
            # else:
            target_metric = 'NDCG@10' # Latest change unified to use NDCG@10
            
            key = f"{dataset}_{split}"
            all_metrics[key] = target_metric
            
            # ========== Try to load from Excel cache ==========
            excel_path = f'excel_results/{dataset}_{split}_decrease_rate.xlsx'
            if os.path.exists(excel_path):
                print(f"Loading cached data from: {excel_path}")
                df = pd.read_excel(excel_path, index_col='Attack Method')
                
                # Rebuild the data dictionary from the DataFrame
                data = {model: {} for model in model_list}
                for attack_display_name in df.index:
                    attack_method = display_to_attack.get(attack_display_name)
                    if attack_method is None:
                        continue
                    for model in model_list:
                        display_name = model_display_names[model]
                        mean_col = display_name
                        std_col = f'{display_name}-std'
                        if mean_col in df.columns and std_col in df.columns:
                            mean_val = df.loc[attack_display_name, mean_col] / 100  # Convert from percentage to decimal
                            std_val = df.loc[attack_display_name, std_col] / 100
                            data[model][attack_method] = (mean_val, std_val)
                
                all_data[key] = data
                continue  # Skip calculation, directly process the next split
            
            # ========== Excel does not exist, recalculate ==========
            data = {model: {} for model in model_list}
            
            for model in model_list:
                # Get the clean target_metric
                clean_target_metric_list = []
                for seed in seed_list:
                    scores_save_path = f'output_attack/scores/{model}/{dataset}_{split}/attack_method_none_seed_{seed}_attacked_num_50/metrics_scores_and_asr.json'
                    try:
                        with open(scores_save_path, 'r') as f:
                            scores = json.load(f)
                        clean_target_metric_list.append(scores[target_metric])
                    except FileNotFoundError:
                        print(f"File not found: {scores_save_path}")
                        continue
                
                if not clean_target_metric_list:
                    print(f"No clean data for model: {model}, dataset: {dataset}")
                    continue
                
                clean_target_metric_mean = np.mean(clean_target_metric_list)
                print(f"model: {model}, dataset: {dataset}, split: {split}, clean_target_metric_mean: {clean_target_metric_mean:.4f}")

                # Calculate the decrease_rate for each attack method
                for query_attack_method in attack_methods_for_plot:
                    # Calculate the decrease_rate for each seed
                    decrease_rate_list = []
                    for seed_idx, seed in enumerate(seed_list):
                        scores_save_path = f'output_attack/scores/{model}/{dataset}_{split}/attack_method_{query_attack_method}_seed_{seed}_attacked_num_50/metrics_scores_and_asr.json'
                        try:
                            with open(scores_save_path, 'r') as f:
                                scores = json.load(f)
                            attacked_val = scores[target_metric]
                            # Calculate the decrease_rate using the clean value of the corresponding seed
                            if seed_idx < len(clean_target_metric_list):
                                clean_val = clean_target_metric_list[seed_idx]
                                dr = (clean_val - attacked_val) / clean_val
                                decrease_rate_list.append(dr)
                        except FileNotFoundError:
                            print(f"File not found: {scores_save_path}")
                            continue
                    
                    if not decrease_rate_list:
                        print(f"No attacked data for model: {model}, dataset: {dataset}, attack_method: {query_attack_method}")
                        continue
                    
                    # Calculate the mean and standard deviation of the decrease_rate list
                    decrease_rate_mean = np.mean(decrease_rate_list)
                    decrease_rate_std = np.std(decrease_rate_list)
                    
                    data[model][query_attack_method] = (decrease_rate_mean, decrease_rate_std)
                    print(f"  attack_method: {query_attack_method}, decrease_rate: {decrease_rate_mean * 100:.2f}% ± {decrease_rate_std * 100:.2f}%")
            
            all_data[key] = data
            
            # ========== Save as Excel file ==========
            # Build the DataFrame
            excel_columns = []
            for model in model_list:
                display_name = model_display_names[model]
                excel_columns.append(display_name)
                excel_columns.append(f'{display_name}-std')
            
            excel_data = []
            for attack_method in attack_methods_for_plot:
                row = []
                for model in model_list:
                    if attack_method in data[model]:
                        mean_val = data[model][attack_method][0] * 100  # Convert to percentage
                        std_val = data[model][attack_method][1] * 100
                    else:
                        mean_val = 0
                        std_val = 0
                    row.append(mean_val)
                    row.append(std_val)
                excel_data.append(row)
            
            df = pd.DataFrame(excel_data, columns=excel_columns, 
                            index=[attack_method_display_names[m] for m in attack_methods_for_plot])
            
            # Save Excel file
            os.makedirs("excel_results", exist_ok=True)
            df.to_excel(excel_path, index_label='Attack Method')
            print(f"Saved Excel: {excel_path}")

    if draw_pic==False:
        return all_data
    
    # ========== Second step: Merge subplots and draw ==========
    # Filter out the datasets that need to be merged (nq and hotpotqa test split)
    combined_datasets = [('nq', 'test'), ('hotpotqa', 'test')]
    # Filter out the datasets that have no data
    combined_datasets = [(d, s) for d, s in combined_datasets if f"{d}_{s}" in all_data]
    
    if len(combined_datasets) >= 2:
        # Create 1 row 2 columns subplots, without sharing the y-axis
        fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), sharey=False)
        
        x = np.arange(len(attack_methods_for_plot))
        width = 0.1
        n_models = len(model_list)
        
        # Define the fill styles and colors
        patterns = ['', '/', '\\', 'x', '+', 'o', '.', '-']
        # colors = plt.cm.tab10.colors[:n_models]
        colors = plt.cm.Set3.colors[:n_models]
        
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']  # Subplot labels
        
        for idx, (dataset, split) in enumerate(combined_datasets):
            ax = axes[idx]
            key = f"{dataset}_{split}"
            data = all_data[key]
            target_metric = all_metrics[key]
            
            # Draw bars for each model
            for i, model in enumerate(model_list):
                means = []
                stds = []
                for attack_method in attack_methods_for_plot:
                    if attack_method in data[model]:
                        means.append(data[model][attack_method][0] * 100)
                        stds.append(data[model][attack_method][1] * 100)
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - n_models / 2 + 0.5) * width
                bars = ax.bar(x + offset, means, width, yerr=stds, 
                            label=model_display_names[model] if idx == 0 else "",  # Add label only in the first subplot
                            capsize=3, color=colors[i],
                            hatch=patterns[i % len(patterns)], 
                            edgecolor='dimgray', linewidth=0.4)
            
            # Set the subplot labels and titles
            ax.set_xlabel(f'{subplot_labels[idx]} {target_metric} Drop Rate on {dataset_display_names[dataset][split]}')
            ax.set_xticks(x)
            ax.set_xticklabels([attack_method_display_names[m] for m in attack_methods_for_plot], rotation=0)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            
            # Add horizontal grid lines
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=0.5)
            ax.set_axisbelow(True)  # Grid lines below the bars
            
            # Set the y-axis label for each subplot, labelpad controls the distance between the label and the axis
            ax.set_ylabel('Drop Rate (%)', labelpad=1)
        
        # Add a shared legend, placed in the center above the entire figure, one row horizontally
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), 
                  ncol=len(model_list), frameon=False, columnspacing=1.0,
                  fontsize=18, handlelength=2, handleheight=1.5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, wspace=0.1)  # top leaves space for the legend, wspace controls the spacing between subplots
        
        # Save the merged figure
        combined_name = '_'.join([f'{d}_{s}' for d, s in combined_datasets])
        # Save both PDF (vector image) and PNG (bitmap image)
        plt.savefig(f'figs/combined_{combined_name}_decrease_rate.pdf', 
                   bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f'figs/combined_{combined_name}_decrease_rate.png', dpi=1200, 
                   bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"Saved: figs/combined_{combined_name}_decrease_rate.pdf & .png")

    # ========== Second step (additional): Generate 2x2 four dataset merged figure ==========
    combined_datasets_2x2 = [('nq', 'test'), ('msmarco', 'dev'), ('hotpotqa', 'test'), ('fiqa', 'test')]
    # Filter out the datasets that have no data
    combined_datasets_2x2 = [(d, s) for d, s in combined_datasets_2x2 if f"{d}_{s}" in all_data]
    
    if len(combined_datasets_2x2) >= 4:
        # Create 2 rows 2 columns subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 9.5), sharey=False)
        axes = axes.flatten()  # Flatten into a one-dimensional array, for easy indexing
        
        x = np.arange(len(attack_methods_for_plot))
        width = 0.1
        n_models = len(model_list)
        
        # Define the fill styles and colors
        patterns = ['', '/', '\\', 'x', '+', 'o', '.', '-']
        # colors = plt.cm.tab10.colors[:n_models]
        colors = plt.cm.Set3.colors[:n_models]
        # colors = [ '#ED342F', '#2356A7', '#178642','#FD763F','#F69A9B', '#83B6E1','#A0D293', '#EECA40'] # My custom color set
        
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']  # Subplot labels
        
        for idx, (dataset, split) in enumerate(combined_datasets_2x2):
            ax = axes[idx]
            key = f"{dataset}_{split}"
            data = all_data[key]
            target_metric = all_metrics[key]
            
            # Draw bars for each model
            for i, model in enumerate(model_list):
                means = []
                stds = []
                for attack_method in attack_methods_for_plot:
                    if attack_method in data[model]:
                        means.append(data[model][attack_method][0] * 100)
                        stds.append(data[model][attack_method][1] * 100)
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - n_models / 2 + 0.5) * width
                bars = ax.bar(x + offset, means, width, yerr=stds, 
                            label=model_display_names[model] if idx == 0 else "",  # Add label only in the first subplot
                            capsize=3, color=colors[i],
                            hatch=patterns[i % len(patterns)], 
                            edgecolor='dimgray', linewidth=0.3,
                            error_kw={'elinewidth': 1, 'capthick': 1})
            
            # Set the subplot labels and titles
            ax.set_xlabel(f'{subplot_labels[idx]} {target_metric} Drop Rate on {dataset_display_names[dataset][split]}')
            ax.set_xticks(x)
            ax.set_xticklabels([attack_method_display_names[m] for m in attack_methods_for_plot], rotation=0)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            
            # Add horizontal grid lines
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=0.5)
            ax.set_axisbelow(True)  # Grid lines below the bars
            
            # Set the y-axis label for each subplot
            ax.set_ylabel('Drop Rate (%)', labelpad=1)
        
        # Add a shared legend, placed in the center above the entire figure, one row horizontally
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), 
                  ncol=len(model_list), frameon=False, columnspacing=0.6,
                  fontsize=18, handlelength=1.9 , handleheight=1.5, handletextpad=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, wspace=0.12, hspace=0.18)  # top leaves space for the legend, hspace controls the spacing between rows
        
        # Save the 2x2 merged figure
        combined_name_2x2 = '_'.join([f'{d}_{s}' for d, s in combined_datasets_2x2])
        plt.savefig(f'figs/combined_2x2_{combined_name_2x2}_decrease_rate.pdf', 
                   bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f'figs/combined_2x2_{combined_name_2x2}_decrease_rate.png', dpi=1200, 
                   bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"Saved: figs/combined_2x2_{combined_name_2x2}_decrease_rate.pdf & .png")
    
    # ========== Third step: Still save the separate figure (optional) ==========
    for key, data in all_data.items():
        # Parse the dataset and split from the key
        parts = key.rsplit('_', 1)
        dataset, split = parts[0], parts[1]
        target_metric = all_metrics[key]
        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        x = np.arange(len(attack_methods_for_plot))
        width = 0.1
        n_models = len(model_list)
        
        patterns = ['', '/', '\\', 'x', '+', 'o', '.', '-']
        # colors = plt.cm.tab10.colors[:n_models]
        colors = plt.cm.Set3.colors[:n_models]
        
        for i, model in enumerate(model_list):
            means = []
            stds = []
            for attack_method in attack_methods_for_plot:
                if attack_method in data[model]:
                    means.append(data[model][attack_method][0] * 100)
                    stds.append(data[model][attack_method][1] * 100)
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, 
                        label=model_display_names[model], capsize=3,
                        color=colors[i],
                        hatch=patterns[i % len(patterns)], 
                        edgecolor='dimgray', linewidth=0.4)
        
        ax.set_ylabel('Drop Rate (%)')
        ax.set_xlabel(f'{target_metric} Drop Rate on {dataset_display_names[dataset][split]}')
        ax.set_xticks(x)
        ax.set_xticklabels([attack_method_display_names[m] for m in attack_methods_for_plot], rotation=0)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                ncol=4, frameon=False, columnspacing=1.2,
                fontsize=14, handlelength=2.0, handleheight=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        # Save both PDF (vector image) and PNG (bitmap image)
        plt.savefig(f'figs/{dataset}_{split}_decrease_rate.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f'figs/{dataset}_{split}_decrease_rate.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"Saved: figs/{dataset}_{split}_decrease_rate.pdf & .png")

    return all_data

def document_attack_main():
    # Record the static indicators of the datasets after corpus poisoning attack, perform statistical analysis, and do not draw pictures
    model_list = [
            
    "qwen3_0.6B",
    "qwen3_4B",
 

    "contriever", 
    "bge_m3",  
    "qwen3", 
    "linq",
    "gte",
    "reasonir",  
    "diver",
    "bge_reasoner"
    ]
    
    # Model display name mapping
    model_display_names = {
        "contriever": "Contriever",
        "bge_m3": "BGE-M3",
        "qwen3": "Qwen3",
        "linq": "Linq",
        "gte": "GTE",
        "reasonir": "ReasonIR",
        "diver": "DIVER",
        "bge_reasoner": "ReasonEmbed",
        "qwen3_4B": "Qwen3-4B",
        "qwen3_0.6B": "Qwen3-0.6B",
        "diver_0.6B": "DIVER-0.6B",
        "diver_1.7B": "DIVER-1.7B",
    }

    seed_list = [1999, 5, 27, 2016, 2026]

    dataset_list = [
        "nq",
        "msmarco",
        "hotpotqa",
    ]
    
    attack_num_list = [10, 50]
    # attack_num_list = [50]
    document_attack_method = 'supervised_poisoning'

    # The returned data structure: {f"{dataset}_{split}": {model: {attack_num: (mean, std)}}}
    all_data = {}
    
    for dataset in dataset_list:

        if dataset == 'msmarco':
            split = 'dev'
        else:
            split = 'test'
        
        # Initialize the data of the dataset_split
        key = f"{dataset}_{split}"
        all_data[key] = {}
        
        for model in model_list:
            all_data[key][model] = {}
            for attack_num in attack_num_list:
                # Get the ASR@20 after attack
                attacked_asr_20_list = []
                for seed in seed_list:
                    scores_save_path = f'output_attack/scores/{model}/{dataset}_{split}/attack_method_{document_attack_method}_seed_{seed}_attacked_num_{attack_num}/metrics_scores_and_asr.json'
                    try:
                        with open(scores_save_path, 'r') as f:
                            scores = json.load(f)
                        attacked_asr_20_list.append(scores['ASR@20'])
                    except FileNotFoundError:
                        print(f"File not found: {scores_save_path}")
                        continue
                
                if not attacked_asr_20_list:
                    print(f"No attacked data for model: {model}, dataset: {dataset}, attack_num: {attack_num}")
                    continue
                
                attacked_asr_20_mean = np.mean(attacked_asr_20_list) * 100  # Convert to percentage
                attacked_asr_20_std = np.std(attacked_asr_20_list) * 100

                # Store the data in the returned data
                all_data[key][model][attack_num] = (attacked_asr_20_mean, attacked_asr_20_std)

                print(f"model: {model}, dataset: {dataset}, split: {split}, attack_num: {attack_num}, ASR@20: {attacked_asr_20_mean:.2f}% ± {attacked_asr_20_std:.2f}%")

    return all_data

if __name__ == "__main__":
    query_attack_main()
    document_attack_main()
