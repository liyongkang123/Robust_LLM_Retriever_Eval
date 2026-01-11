'''
Calculate the relationship between cosine and ASR@20,  
'''

import numpy as np
import torch
import os
import json
from itertools import chain
from tqdm import tqdm
import glob
from eval_attack_data_statistics import document_attack_main
from draw_isoscore import model_performance
from isotropy_utils.existing_scores import cosine_score
from transformers import set_seed
from beir.retrieval.search.dense.util import cos_sim, dot_score, pickle_load, save_embeddings

def model_embedding_average_cosine_similarity_and_asr_20():
    # This function is to model the relationship between average cosine similarity and ASR@20

    asr_20_data = document_attack_main() # This is the average of asr@20 for each seed
    model_performance_data = model_performance()

    model_list = [
    # "contriever", 
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
        "bge_reasoner": "ReasonEmbed"
    }

    dataset_list = [ 
        "nq", 
        "hotpotqa", 
        "msmarco",
        ]
    dataset_display_names = {
        "nq": {"test": "NQ",},
        "hotpotqa": {"test": "HotpotQA"},
        "nfcorpus": {"test": "NFCorpus"},
        "fiqa": {"test": "FiQA"},
        "msmarco": {"dev": "MS MARCO", "trec_dl19": "TREC DL 2019", "trec_dl20": "TREC DL 2020"}, # (Dev)
    }

    # Read the model's embedding, read the embedding according to the model name and dataset name
    embedding_save_root = '/scratch-shared/yli4/LLM_Robust/Clean'

    seed_list = [1999, 5, 27, 2016, 2026 ] # 5, 27, 2016, 2026

    sample_size = 100000 # 100k samples, it is usually enough

    if os.path.exists('data/average_cosine_similarity_data_100k.json'):
        with open('data/average_cosine_similarity_data_100k.json', 'r') as f:
            average_cosine_similarity_data = json.load(f)
    else:
        average_cosine_similarity_data = {} # {"dataset_split": {model: [average_cosine_similarity_mean, average_cosine_similarity_std]}}
        for dataset in dataset_list:
            
            if dataset == 'msmarco':
                split = 'dev'
            else:
                split = 'test'
            key = f"{dataset}_{split}"  # Use string as key, JSON compatible
            average_cosine_similarity_data[key] = {}
            
            for model in model_list:
                average_cosine_similarity_list = []
                for seed in seed_list:
                    set_seed(seed)
                    embedding_save_path = os.path.join(embedding_save_root, f'output/embeddings/{model}/{dataset}')
                    corpus_embeddings_files = sorted(glob.glob(os.path.join(embedding_save_path, 'corpus.*.pkl')))
                    

                    # Allocate 100k samples to each shard, then sample separately, concatenate, and then calculate isoscore
                    num_shards = len(corpus_embeddings_files)
                    base_size, remainder = divmod(sample_size, num_shards)

                    # The first remainder shards get an extra sample
                    shard_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_shards)]
                    


                    shards = chain(map(pickle_load, corpus_embeddings_files))
                    if len(corpus_embeddings_files) > 1:
                        shards = tqdm(shards, desc="Loading shards into index", total=len(corpus_embeddings_files))
                    corpus_embeddings_sampled_all = []
                    for shard_idx, (corpus_embeddings, corpus_ids) in enumerate(shards):
                        corpus_embeddings = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
                        # Sample according to id
                        # Sample according to shard index
                        n_samples = min(shard_sizes[shard_idx], corpus_embeddings.shape[0])  # Prevent overflow
                        indices = np.random.choice(corpus_embeddings.shape[0], n_samples, replace=False)
                        corpus_embeddings_sampled = corpus_embeddings[indices]
                        corpus_embeddings_sampled_all.append(corpus_embeddings_sampled)

                    corpus_embeddings_sampled_all = np.concatenate(corpus_embeddings_sampled_all, axis=0)
                    print('corpus_embeddings_sampled_all shape:', corpus_embeddings_sampled_all.shape)

                    # Calculate average cosine similarity
                    points_numpy = np.transpose(corpus_embeddings_sampled_all)
                    average_cosine_similarity = cosine_score(points_numpy)
                    print('seed:', seed, 'average_cosine_similarity:', average_cosine_similarity)
                    
                    average_cosine_similarity_list.append(average_cosine_similarity)
                average_cosine_similarity_mean = np.mean(average_cosine_similarity_list)
                average_cosine_similarity_std = np.std(average_cosine_similarity_list)
                average_cosine_similarity_data[key][model] = [average_cosine_similarity_mean, average_cosine_similarity_std]  # Use list instead of tuple, JSON compatible
                print('model:', model, 'dataset:', dataset, 'split:', split, 'seed_list:', seed_list, 'average_cosine_similarity_mean:', average_cosine_similarity_mean, 'average_cosine_similarity_std:', average_cosine_similarity_std)
        
        # Save the calculated average_cosine_similarity_data to the json file
        with open('data/average_cosine_similarity_data_100k.json', 'w') as f:
            json.dump(average_cosine_similarity_data, f, indent=2)
        print('Saved: data/average_cosine_similarity_data_100k.json')

    # Start drawing the figure
    # Draw scatter plot, the horizontal axis is average_cosine_similarity, the vertical axis is ASR@20, use matplotlib to draw the figure, try to find the correlation between them, draw one figure for each dataset
    import matplotlib.pyplot as plt
    plt.rcParams['xtick.labelsize'] = 18    # x-axis tick font size
    plt.rcParams['ytick.labelsize'] = 18    # y-axis tick font size
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from scipy.stats import linregress
    
    attack_num = 50  # Select the attack_num to use, optional 10 or 50
    
    for dataset in dataset_list:
        if dataset == 'msmarco':
            split = 'dev'
        else:
            split = 'test'
        
        key = f"{dataset}_{split}"  # Use string key
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        avg_cosines = []
        asr_20s = []
        labels = []
        performance =[]
        
        for model in model_list:
            if key not in average_cosine_similarity_data or model not in average_cosine_similarity_data[key]:
                continue
            if key not in asr_20_data or model not in asr_20_data[key]:
                continue
            if attack_num not in asr_20_data[key][model]:
                continue
                
            average_cosine_similarity_mean, average_cosine_similarity_std = average_cosine_similarity_data[key][model]
            asr_20_mean, asr_20_std = asr_20_data[key][model][attack_num]
            
            avg_cosines.append(average_cosine_similarity_mean)
            asr_20s.append(asr_20_mean)
            labels.append(model_display_names[model])
            performance.append(model_performance_data[key][model])
        # Calculate the correlation coefficient
        avg_cosines_arr = np.array(avg_cosines)
        asr_20s_arr = np.array(asr_20s)
        
        if len(avg_cosines) >= 3:  # At least 3 points are needed to calculate the correlation
            # Pearson correlation coefficient (linear correlation)
            pearson_r, pearson_p = pearsonr(avg_cosines_arr, asr_20s_arr)
            # Spearman rank correlation coefficient (monotonic correlation, more robust to outliers)
            spearman_r, spearman_p = spearmanr(avg_cosines_arr, asr_20s_arr)
            # Kendall rank correlation coefficient (another rank correlation)
            kendall_tau, kendall_p = kendalltau(avg_cosines_arr, asr_20s_arr)
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(avg_cosines_arr, asr_20s_arr)
            r_squared = r_value ** 2
            
            # Calculate the partial correlation coefficient: controlling performance, the correlation between Avg Cosine and ASR
            performance_arr = np.array(performance)
            # Method: calculate the partial correlation coefficient through the regression residual
            # 1. Avg Cosine 关于 performance 的残差
            slope_cosine_perf, intercept_cosine_perf, _, _, _ = linregress(performance_arr, avg_cosines_arr)
            residual_cosine = avg_cosines_arr - (slope_cosine_perf * performance_arr + intercept_cosine_perf)
            # 2. The residual of ASR about performance
            slope_asr_perf, intercept_asr_perf, _, _, _ = linregress(performance_arr, asr_20s_arr)
            residual_asr = asr_20s_arr - (slope_asr_perf * performance_arr + intercept_asr_perf)
            # 3. The correlation coefficient between the residuals is the partial correlation coefficient
            partial_r, partial_p = pearsonr(residual_cosine, residual_asr)
            
            print(f"\n=== {dataset.upper()} ({split}) Correlation Analysis ===")
            print(f"Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.4f}")
            print(f"Spearman ρ = {spearman_r:.4f}, p-value = {spearman_p:.4f}")
            print(f"Kendall τ = {kendall_tau:.4f}, p-value = {kendall_p:.4f}")
            print(f"Linear Regression: R² = {r_squared:.4f}, slope = {slope:.4f}")
            print(f"Partial r (controlling performance) = {partial_r:.4f}, p-value = {partial_p:.4f}")
        else:
            pearson_r, spearman_r, kendall_tau, r_squared = None, None, None, None
            slope, intercept = None, None
        
        # Draw scatter plot, the color changes according to the performance (the higher the performance, the darker the color)
        scatter = ax.scatter(avg_cosines, asr_20s, s=100, alpha=0.9, zorder=5,
                            c=performance, cmap='Blues', edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NDCG@10 (%)', fontsize=16)
        
        # Draw linear regression fitting line (commented out, because the data is not necessarily linear)
        # if slope is not None:
        #     x_line = np.linspace(min(avg_cosines) - 0.01, max(avg_cosines) + 0.01, 100)
        #     y_line = slope * x_line + intercept
        #     ax.plot(x_line, y_line, 'r--', alpha=0.7, linewidth=3, label='Linear fit', zorder=3)
        
        # Add model name label to each point
        for i, label in enumerate(labels):
            if label == "ReasonIR":
                xytext = (-30, -16)  # ReasonIR label is in the lower left
            elif label == "Qwen3":
                xytext = (5, -3)  # Qwen3 label is in the正右方
            elif label == "BGE-M3":
                xytext = (-13, -16.5)  # BGE-M3 label is in the正下方
            elif label == "ReasonEmbed":
                xytext = (-45, -16)  # ReasonEmbed label is in the正下方
            elif label == "DIVER":
                xytext = (-30, -16)  # DIVER label is in the正下方
            else:
                xytext = (5, 5)  # Other labels are in the upper right
            ax.annotate(label, (avg_cosines[i], asr_20s[i]), 
                       textcoords="offset points", xytext=xytext, fontsize=16)
        
        ax.set_xlabel('Avg. Cosine Similarity', fontsize=20)
        ax.set_ylabel(f'ASR@20 on {dataset_display_names[dataset][split]} (%)', fontsize=20)
        # ax.set_title(f'IsoScore vs ASR@20 on {dataset_display_names[dataset][split]}', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add correlation coefficient text box, color according to significance
        if pearson_r is not None:
            # Determine significance and label
            p_sig = "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else " n.s.")
            sp_sig = "**" if spearman_p < 0.01 else ("*" if spearman_p < 0.05 else " n.s.")
            partial_sig = "**" if partial_p < 0.01 else ("*" if partial_p < 0.05 else " n.s.")
            
            textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.3f}){p_sig}'
            # textstr += f'\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.3f}){sp_sig}'  # Spearman is commented out
            textstr += f'\nPartial r = {partial_r:.3f} (p={partial_p:.3f}){partial_sig}'
            # textstr += f'\nR² = {r_squared:.3f}'  # R² is commented out
            
            # Set the color according to the Pearson p value: significant in light green, not significant in light red
            if pearson_p < 0.05:
                box_color = 'lightgreen'
            else:
                box_color = 'mistyrose'  # Light red
            
            # Remove the border, set zorder=1 to put the text box below the scatter (zorder=5)
            props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='none')
            # hotpotQA dataset text box position up
            text_y = 0.2 if dataset in ['hotpotqa', 'nq'] else 0.15
            ax.text(0.43, text_y, textstr, transform=ax.transAxes, fontsize=16,
                   verticalalignment='top', bbox=props, zorder=1)
        
        plt.tight_layout()
        plt.savefig(f'figs/average_cosine_similarity_asr_20_{dataset}_{split}_attack{attack_num}.pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved: figs/average_cosine_similarity_asr_20_{dataset}_{split}_attack{attack_num}.pdf")
    
    return
    

def model_embedding_average_cosine_similarity_and_query_variations():
    # This function is to model the impact of query variations and Average Cosine Similarity
    # Contains 5 query variations, each drawn one sub-figure, placed on the same figure (1 row 5 columns)
    
    from eval_attack_data_statistics import query_attack_main
    
    # 1. Get the data
    # Format: {f"{dataset}_{split}": {model: {attack_method: (mean, std)}}}
    query_variations_data = query_attack_main(draw_pic=False)
    
    model_list = [
    # "contriever", 
    "bge_m3",  
    "qwen3", 
    "linq",
    "gte",
    "reasonir",  
    "diver",
    "bge_reasoner"
    ]
    
    model_display_names = {
        "contriever": "Contriever",
        "bge_m3": "BGE-M3",
        "qwen3": "Qwen3",
        "linq": "Linq",
        "gte": "GTE",
        "reasonir": "ReasonIR",
        "diver": "DIVER",
        "bge_reasoner": "ReasonEmbed"
    }

    dataset_list = [ 
        "nq", 
        "hotpotqa", 
        "msmarco",
        ]
        
    dataset_display_names = {
        "nq": {"test": "NQ",},
        "hotpotqa": {"test": "HotpotQA"},
        "nfcorpus": {"test": "NFCorpus"},
        "fiqa": {"test": "FiQA"},
        "msmarco": {"dev": "MS MARCO", "trec_dl19": "TREC DL 2019", "trec_dl20": "TREC DL 2020"},
    }

    # Query variations list
    query_attack_method_list = [
        "mispelling",
        "ordering",
        "synonym",
        "paraphrase",
        "naturality",
    ]
    attack_method_display_names = {
        "mispelling": "Misspelling",
        "ordering": "Reordering",
        "synonym": "Synonymizing",
        "paraphrase": "Paraphrasing",
        "naturality": "Naturalizing",
    }
    
    # 2. Read the Average Cosine Similarity data
    if os.path.exists('data/average_cosine_similarity_data_100k.json'):
        with open('data/average_cosine_similarity_data_100k.json', 'r') as f:
            avg_cosine_data = json.load(f)
    else:
        print("Warning: data/average_cosine_similarity_data_100k.json not found. Please run model_embedding_average_cosine_similarity_and_asr_20 first to generate it.")
        return 

    # 3. Draw the figure (1x5 Subplots per dataset)
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, linregress
    
    plt.rcParams['xtick.labelsize'] = 20    
    plt.rcParams['ytick.labelsize'] = 20
    
    # Used to collect all the plotting data
    plot_data = {}
    
    for dataset in dataset_list:
        if dataset == 'msmarco':
            split = 'dev'
        else:
            split = 'test'
        
        key = f"{dataset}_{split}"
        if key not in query_variations_data:
            print(f"Skipping {key}: No query variation data found.")
            continue
        
        plot_data[key] = {}  # Initialize the dictionary of the dataset
            
        # Create a large figure with 1 row 5 columns
        fig, axes = plt.subplots(1, 5, figsize=(25, 3.5), sharey=False)
        
        # Define the marker shape of each model
        model_markers = {
            "bge_m3": "o",        # Circular
            "qwen3": "s",         # Square
            "linq": "^",          # Up triangle
            "gte": "D",           # Diamond
            "reasonir": "v",      # Down triangle
            "diver": "p",         # Pentagon
            "bge_reasoner": "*",  # Star
        }
        
        # Used to collect the handles of the legend
        legend_handles = []
        legend_labels_list = []
        
        for idx, method in enumerate(query_attack_method_list):
            ax = axes[idx]
            
            avg_cosines = []
            drop_rates = []
            model_keys = []  # Save the key of the model to get the marker
            
            for model in model_list:
                # Check data availability
                if key not in avg_cosine_data or model not in avg_cosine_data[key]:
                    continue
                if model not in query_variations_data[key]:
                    continue
                if method not in query_variations_data[key][model]:
                    continue
                
                cosine_mean, _ = avg_cosine_data[key][model]
                drop_rate, _ = query_variations_data[key][model][method]
                
                # drop_rate original data is usually a small number (e.g. 0.05), convert to percentage
                avg_cosines.append(cosine_mean)
                drop_rates.append(drop_rate * 100) 
                model_keys.append(model)
            
            # Calculate the correlation
            avg_cosines_arr = np.array(avg_cosines)
            drop_rates_arr = np.array(drop_rates)
            
            pearson_r, pearson_p, slope, intercept = None, None, None, None
            if len(avg_cosines) >= 3:
                pearson_r, pearson_p = pearsonr(avg_cosines_arr, drop_rates_arr)
                slope, intercept, r_value, p_value, std_err = linregress(avg_cosines_arr, drop_rates_arr)

            # Collect the plotting data of this method
            plot_data[key][method] = {
                'avg_cosines': avg_cosines,
                'drop_rates': drop_rates,
                'model_keys': model_keys,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
            }            
            # Scatter Plot - Use different marker shapes
            for i, model in enumerate(model_keys):
                scatter = ax.scatter(avg_cosines[i], drop_rates[i], s=200, alpha=0.9, zorder=5,
                                   marker=model_markers[model], color='steelblue', 
                                   edgecolors='black', linewidths=0.5)
                # Collect the legend information in the first subplot
                if idx == 0:
                    legend_handles.append(scatter)
                    legend_labels_list.append(model_display_names[model])
            
            # Regression Line (commented out)
            # if pearson_r is not None:
            #      x_line = np.linspace(min(avg_cosines) - 0.01, max(avg_cosines) + 0.01, 100)
            #      y_line = slope * x_line + intercept
            #      ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2, zorder=3)

            ax.set_title(attack_method_display_names[method], fontsize=20)
            ax.set_xlabel('Avg. Cosine Similarity', fontsize=20, labelpad=-1.8)
            if idx == 0:
                ax.set_ylabel(f'Drop Rate on {dataset_display_names[dataset][split]} (%)', fontsize=20)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # correlation text
            if pearson_r is not None:
                p_sig = "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else " n.s.")
                textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.3f}){p_sig}'
                
                # Set the color according to the Pearson p value: significant in light green, not significant in light red
                if pearson_p < 0.05:
                    box_color = 'lightgreen'
                else:
                    box_color = 'mistyrose'  # Light red

                props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='none')
                # Put the text box below the xlabel, use negative y coordinates
                ax.text(0.5, -0.25, textstr, transform=ax.transAxes, fontsize=20,
                        ha='center', verticalalignment='top', bbox=props, zorder=4)

        # Add the legend at the top
        fig.legend(legend_handles, legend_labels_list, loc='upper center', 
                   ncol=len(legend_labels_list), fontsize=20, frameon=False,
                   bbox_to_anchor=(0.5, 1.12), handletextpad=0.1, columnspacing=1.0)
        
        plt.subplots_adjust(wspace=0.15)  # Reduce the horizontal space between subplots
        
        save_path = f'figs/avg_cosine_vs_query_variations_{dataset}_{split}.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print(f"Saved: {save_path}")

    return plot_data
if __name__ == '__main__':
    # Optional to run one of the following functions
    model_embedding_average_cosine_similarity_and_asr_20()
    # model_embedding_average_cosine_similarity_and_query_variations()