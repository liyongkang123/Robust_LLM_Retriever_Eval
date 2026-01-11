import torch
import numpy as np
from sklearn.decomposition import PCA
from isotropy_utils.isoscore_numpy import IsoScore as IsoScore_numpy

from itertools import chain
import numpy as np
import torch
from tqdm import tqdm
from beir.retrieval.search.dense.util import cos_sim, dot_score, pickle_load, save_embeddings
import os
import glob
from transformers import set_seed
import json

from eval_attack_data_statistics import document_attack_main,query_attack_main


def model_performance():
    data= {
        "nq_test": {
            "contriever": 49.8, 
            "bge_m3":60.6,  
            "qwen3":64.9, 
            "linq": 70.4,
            "gte": 66.8,
            "reasonir": 52.6,  
            "diver": 54.2,
            "bge_reasoner": 60.7,
            "qwen3_4B": 61.5,
            "qwen3_0.6B": 51.8,
        },
        "hotpotqa_test": {
            "contriever": 63.8, 
            "bge_m3":69.5,  
            "qwen3":76.8, 
            "linq": 76.4,
            "gte": 73.0,
            "reasonir": 62.8,  
            "diver": 65.8,
            "bge_reasoner": 44.4,
            "qwen3_4B": 73.9,
            "qwen3_0.6B": 65.2,
        },
        # "msmarco_dev": { # MRR@10
        #     "contriever": 34.1 , 
        #     "bge_m3": 32.2,  
        #     "qwen3": 36.8, 
        #     "linq": 38.2, 
        #     "gte": 39.14, 
        #     "reasonir": 26.3,  
        #     "diver": 25.7, 
        #     "bge_reasoner": 26.8 , 
        #     "qwen3_4B": 35.8,
        #     "qwen3_0.6B": 30.8,
        # },
        "msmarco_dev": { # NDCG@10
            "contriever": 40.7 , 
            "bge_m3": 38.4,  
            "qwen3": 43.7, 
            "linq": 44.9, 
            "gte": 46.0, 
            "reasonir": 32.2,  
            "diver": 31.4, 
            "bge_reasoner": 32.3 , 
            "qwen3_4B": 42.4,
            "qwen3_0.6B": 37.1,
        },
    }
    return data

 
def model_embedding_isoscore_and_asr_20():
 
    asr_source = 'each_seed_attack_mean'
 
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

    sample_size = 100000 # 100k samples, usually enough

    if os.path.exists('data/isoscore_data_100k.json'):
        with open('data/isoscore_data_100k.json', 'r') as f:
            isoscore_data = json.load(f)
    else:
        isoscore_data = {} # {"dataset_split": {model: [isoscore_mean, isoscore_std]}}
        for dataset in dataset_list:
            
            if dataset == 'msmarco':
                split = 'dev'
            else:
                split = 'test'
            key = f"{dataset}_{split}"  # Use string as key, JSON compatible
            isoscore_data[key] = {}
            
            for model in model_list:
                isoscore_list = []
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
                    points_numpy = np.transpose(corpus_embeddings_sampled_all)
                    sc_numpy = IsoScore_numpy(points_numpy)
                    print('seed:', seed, 'sc_numpy:', sc_numpy)
                    
                    # sc_torch = IsoScore_torch(torch.from_numpy(corpus_embeddings_sampled_all).float())
                    # print('sc_torch:', sc_torch)
                    isoscore_list.append(sc_numpy)
                isoscore_mean = np.mean(isoscore_list)
                isoscore_std = np.std(isoscore_list)
                isoscore_data[key][model] = [isoscore_mean, isoscore_std]  # Use list instead of tuple, JSON compatible
                print('model:', model, 'dataset:', dataset, 'split:', split, 'seed_list:', seed_list, 'isoscore_mean:', isoscore_mean, 'isoscore_std:', isoscore_std)
        
        # Save the calculated isoscore_data to the json file
        with open('data/isoscore_data_100k.json', 'w') as f:
            json.dump(isoscore_data, f, indent=2)
        print('Saved: data/isoscore_data_100k.json')


    # Draw the scatter plot, the horizontal axis is ISOscore, the vertical axis is ASR@20, use matplotlib to draw the figure, try to find the correlation between them, draw one figure for each dataset
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
        
        isoscores = []
        asr_20s = []
        labels = []
        performance =[]
        
        for model in model_list:
            if key not in isoscore_data or model not in isoscore_data[key]:
                continue
            if key not in asr_20_data or model not in asr_20_data[key]:
                continue
            if attack_num not in asr_20_data[key][model]:
                continue
                
            isoscore_mean, isoscore_std = isoscore_data[key][model]
            asr_20_mean, asr_20_std = asr_20_data[key][model][attack_num]
            
            isoscores.append(isoscore_mean)
            asr_20s.append(asr_20_mean)
            labels.append(model_display_names[model])
            performance.append(model_performance_data[key][model])
        # Calculate the correlation coefficient
        isoscores_arr = np.array(isoscores)
        asr_20s_arr = np.array(asr_20s)
        
        if len(isoscores) >= 3:  # At least 3 points are needed to calculate the correlation
            # Pearson correlation coefficient (linear correlation)
            pearson_r, pearson_p = pearsonr(isoscores_arr, asr_20s_arr)
            # Spearman rank correlation coefficient (monotonic correlation, more robust to outliers)
            spearman_r, spearman_p = spearmanr(isoscores_arr, asr_20s_arr)
            # Kendall rank correlation coefficient (another rank correlation)
            kendall_tau, kendall_p = kendalltau(isoscores_arr, asr_20s_arr)
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(isoscores_arr, asr_20s_arr)
            r_squared = r_value ** 2
            
            # Calculate the partial correlation coefficient: controlling performance, the correlation between IsoScore and ASR
            performance_arr = np.array(performance)
            # Method: calculate the partial correlation coefficient through the regression residual
            # 1. The residual of IsoScore about performance
            slope_iso_perf, intercept_iso_perf, _, _, _ = linregress(performance_arr, isoscores_arr)
            residual_iso = isoscores_arr - (slope_iso_perf * performance_arr + intercept_iso_perf)
            # 2. The residual of ASR about performance
            slope_asr_perf, intercept_asr_perf, _, _, _ = linregress(performance_arr, asr_20s_arr)
            residual_asr = asr_20s_arr - (slope_asr_perf * performance_arr + intercept_asr_perf)
            # 3. The correlation coefficient between the residuals is the partial correlation coefficient
            partial_r, partial_p = pearsonr(residual_iso, residual_asr)
            
            print(f"\n=== {dataset.upper()} ({split}) Correlation Analysis ===")
            print(f"Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.4f}")
            print(f"Spearman ρ = {spearman_r:.4f}, p-value = {spearman_p:.4f}")
            print(f"Kendall τ = {kendall_tau:.4f}, p-value = {kendall_p:.4f}")
            print(f"Linear Regression: R² = {r_squared:.4f}, slope = {slope:.4f}")
            print(f"Partial r (controlling performance) = {partial_r:.4f}, p-value = {partial_p:.4f}")
        else:
            pearson_r, spearman_r, kendall_tau, r_squared = None, None, None, None
            slope, intercept = None, None
        
        # Draw the scatter plot, the color changes according to performance (the higher the performance, the darker the color)
        scatter = ax.scatter(isoscores, asr_20s, s=100, alpha=0.9, zorder=5,
                            c=performance, cmap='Blues', edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NDCG@10 (%)', fontsize=16)
        
        # Draw the linear regression fitting line (commented out, because the data is not necessarily linear)
        # if slope is not None:
        #     x_line = np.linspace(min(isoscores) - 0.01, max(isoscores) + 0.01, 100)
        #     y_line = slope * x_line + intercept
        #     ax.plot(x_line, y_line, 'r--', alpha=0.7, linewidth=3, label='Linear fit', zorder=3)
        
        # Add the model name label to each point
        for i, label in enumerate(labels):
            if label == "ReasonIR":
                xytext = (-45, -15)  # ReasonIR label is in the lower left
            elif label == "Qwen3":
                xytext = (5, -3)  # Qwen3 label is in the正右方
            elif label == "BGE-M3":
                xytext = (-45, -16)  # BGE-M3 label is in the正下方
            elif label == "ReasonEmbed":
                xytext = (5,-1)  # ReasonEmbed label is in the正下方
            elif label == "DIVER":
                xytext = (-30, -17)  # DIVER label is in the正下方
            else:
                xytext = (5, 5)  # Other labels are in the upper right
            ax.annotate(label, (isoscores[i], asr_20s[i]), 
                       textcoords="offset points", xytext=xytext, fontsize=16)
        
        ax.set_xlabel('IsoScore', fontsize=20)
        ax.set_ylabel(f'ASR@20 on {dataset_display_names[dataset][split]} (%)', fontsize=20)
        # ax.set_title(f'IsoScore vs ASR@20 on {dataset_display_names[dataset][split]}', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add the correlation coefficient text box, mark the significance according to the significance
        if pearson_r is not None:
            # Determine the significance and mark it
            p_sig = "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else " n.s.")
            sp_sig = "**" if spearman_p < 0.01 else ("*" if spearman_p < 0.05 else " n.s.")
            partial_sig = "**" if partial_p < 0.01 else ("*" if partial_p < 0.05 else " n.s.")
            
            textstr = f'Pearson r = {pearson_r:.2f} (p={pearson_p:.2f}){p_sig}'
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
            text_y = 0.97 if dataset in ['hotpotqa', 'nq'] else 0.88
            ax.text(0.03, text_y, textstr, transform=ax.transAxes, fontsize=16,
                   verticalalignment='top', bbox=props, zorder=1)
        
        plt.tight_layout()
        plt.savefig(f'figs/isoscore_asr_20_{dataset}_{split}_attack{attack_num}_{asr_source}.pdf', bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved: figs/isoscore_asr_20_{dataset}_{split}_attack{attack_num}_{asr_source}.pdf")
    
    return
 
def model_embedding_isoscore_and_query_variations():
    # This function is to model the impact of query variations and IsoScore
    # Contains 5 query variations, each drawn one sub-figure, placed on the same figure (1 row 5 columns)
    
    # 1. Get data
    # Format: {f"{dataset}_{split}": {model: {attack_method: (mean, std)}}}
    query_variations_data = query_attack_main(draw_pic=False)
    
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
    
    # 2. Read or calculate IsoScore (logic same as model_embedding_isoscore_and_asr_20)
    embedding_save_root = '/scratch-shared/yli4/LLM_Robust/Clean'
    seed_list = [1999, 5, 27, 2016, 2026]
    sample_size = 100000 

    if os.path.exists('data/isoscore_data_100k.json'):
        with open('data/isoscore_data_100k.json', 'r') as f:
            isoscore_data = json.load(f)
    else:
        # If the file does not exist, here is a simple processing, print the prompt or copy the previous generation code.
        # To keep it simple, assume that the file has been generated by the previous function. If not, please run model_embedding_isoscore_and_asr_20 first.
        print("Warning: data/isoscore_data_100k.json not found. Please run model_embedding_isoscore_and_asr_20 first to generate it.")
        return 

    # 3. Draw the figure (1x5 Subplots per dataset)
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr, kendalltau, linregress
    
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
        # sharey=True to make all subplots share the Y axis, for easy comparison of the drop rate
        
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
            
            isoscores = []
            drop_rates = []
            model_keys = []  # Save the key of the model to get the marker
            
            for model in model_list:
                # Check data availability
                if key not in isoscore_data or model not in isoscore_data[key]:
                    continue
                if model not in query_variations_data[key]:
                    continue
                if method not in query_variations_data[key][model]:
                    continue
                
                iso_mean, _ = isoscore_data[key][model]
                drop_rate, _ = query_variations_data[key][model][method]
                
                # drop_rate original data is usually a small number (e.g. 0.05), convert to percentage
                isoscores.append(iso_mean)
                drop_rates.append(drop_rate * 100) 
                model_keys.append(model)
            
            # Calculate the correlation
            isoscores_arr = np.array(isoscores)
            drop_rates_arr = np.array(drop_rates)
            
            pearson_r, pearson_p, slope, intercept = None, None, None, None
            if len(isoscores) >= 3:
                pearson_r, pearson_p = pearsonr(isoscores_arr, drop_rates_arr)
                slope, intercept, r_value, p_value, std_err = linregress(isoscores_arr, drop_rates_arr)

            # Collect the plotting data of this method
            plot_data[key][method] = {
                'isoscores': isoscores,
                'drop_rates': drop_rates,
                'model_keys': model_keys,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
            }            
            # Scatter Plot - Use different marker shapes
            for i, model in enumerate(model_keys):
                scatter = ax.scatter(isoscores[i], drop_rates[i], s=200, alpha=0.9, zorder=5,
                                   marker=model_markers[model], color='steelblue', 
                                   edgecolors='black', linewidths=0.5)
                # Collect the legend information in the first subplot
                if idx == 0:
                    legend_handles.append(scatter)
                    legend_labels_list.append(model_display_names[model])
            
            # Regression Line (commented out)
            # if pearson_r is not None:
            #      x_line = np.linspace(min(isoscores) - 0.01, max(isoscores) + 0.01, 100)
            #      y_line = slope * x_line + intercept
            #      ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2, zorder=3)

            ax.set_title(attack_method_display_names[method], fontsize=20)
            ax.set_xlabel('IsoScore', fontsize=20, labelpad=-1.8)
            if idx == 0:
                ax.set_ylabel(f'Drop Rate on {dataset_display_names[dataset][split]} (%)', fontsize=20)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # correlation text
            if pearson_r is not None:
                p_sig = "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else " n.s.")
                textstr = f'Pearson r = {pearson_r:.2f} (p={pearson_p:.2f}){p_sig}'
                
                # Set the color according to the Pearson p value: significant in light green, not significant in light red
                if pearson_p < 0.05:
                    box_color = 'lightgreen'
                else:
                    box_color = 'mistyrose'  # Light red

                props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='none')
                # Put the text box below the xlabel, use negative y coordinates
                ax.text(0.5, -0.27, textstr, transform=ax.transAxes, fontsize=20,
                        ha='center', verticalalignment='top', bbox=props, zorder=4)

        # Add the legend at the top
        fig.legend(legend_handles, legend_labels_list, loc='upper center', 
                   ncol=len(legend_labels_list), fontsize=20, frameon=False,
                   bbox_to_anchor=(0.5, 1.12), handletextpad=0.1, columnspacing=1.0)
        
        # plt.suptitle(f'IsoScore vs Query Variation Drop Rate ({dataset_display_names[dataset][split]})', fontsize=20)
        plt.subplots_adjust(wspace=0.15)  # Reduce horizontal space between subplots
        # plt.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for suptitle and colorbar
        
        save_path = f'figs/isoscore_vs_query_variations_{dataset}_{split}.pdf' # Save as pdf
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {save_path}")

    return plot_data



def model_embedding_isoscore_and_avg_cosine_and_query_variations():
    """
    Merge the query variations figure of IsoScore and Avg Cosine Similarity into a 2x5 large figure
    First row: Avg Cosine vs Drop Rate (5 query variations)
    Second row: IsoScore vs Drop Rate (5 query variations)
    The color depth changes according to model_performance_data (NDCG@10)
    The scatter plot legend is placed in the upper left, and the color legend is placed in the upper right
    """
    from eval_attack_data_statistics import query_attack_main
    from scipy.stats import pearsonr, linregress
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Get query variations data
    query_variations_data = query_attack_main(draw_pic=False)
    
    # Get model performance data (NDCG@10)
    model_performance_data = model_performance()
    
    # Read IsoScore data
    if os.path.exists('data/isoscore_data_100k.json'):
        with open('data/isoscore_data_100k.json', 'r') as f:
            isoscore_data = json.load(f)
    else:
        print("Warning: data/isoscore_data_100k.json not found.")
        return
    
    # Read Avg Cosine data
    if os.path.exists('data/average_cosine_similarity_data_100k.json'):
        with open('data/average_cosine_similarity_data_100k.json', 'r') as f:
            avg_cosine_data = json.load(f)
    else:
        print("Warning: data/average_cosine_similarity_data_100k.json not found.")
        return
    
    model_list = ["bge_m3", "qwen3", "linq", "gte", "reasonir", "diver", "bge_reasoner"]
    
    model_display_names = {
        "bge_m3": "BGE-M3", "qwen3": "Qwen3", "linq": "Linq", "gte": "GTE",
        "reasonir": "ReasonIR", "diver": "DIVER", "bge_reasoner": "ReasonEmbed"
    }
    
    model_markers = {
        "bge_m3": "o", "qwen3": "s", "linq": "^", "gte": "D",
        "reasonir": "v", "diver": "p", "bge_reasoner": "*",
    }

    dataset_list = ["nq", "hotpotqa", "msmarco"]
    dataset_display_names = {
        "nq": {"test": "NQ"}, "hotpotqa": {"test": "HotpotQA"},
        "msmarco": {"dev": "MS MARCO"},
    }
    
    query_attack_method_list = ["mispelling", "ordering", "synonym", "paraphrase", "naturality"]
    attack_method_display_names = {
        "mispelling": "Misspelling", "ordering": "Reordering", "synonym": "Synonymizing",
        "paraphrase": "Paraphrasing", "naturality": "Naturalizing",
    }
    
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    for dataset in dataset_list:
        split = 'dev' if dataset == 'msmarco' else 'test'
        key = f"{dataset}_{split}"
        
        if key not in query_variations_data:
            print(f"Skipping {key}: No query variation data found.")
            continue
        
        # Collect all performance values of the dataset, for uniform colorbar range
        all_performances = []
        for model in model_list:
            if key in model_performance_data and model in model_performance_data[key]:
                all_performances.append(model_performance_data[key][model])
        
        if not all_performances:
            print(f"Skipping {key}: No performance data found.")
            continue
        
        vmin, vmax = min(all_performances), max(all_performances)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps['magma_r'] # plasma 还不错  viridis 里面有绿色不好   Magma  inferno 不错
        
        # Create a large figure with 2 rows 5 columns
        fig, axes = plt.subplots(2, 5, figsize=(25, 8), sharey=False)
        
        legend_handles = []
        legend_labels_list = []
        last_scatter = None  # Used for colorbar
        
        for idx, method in enumerate(query_attack_method_list):
            # === First row: Avg Cosine ===
            ax_cos = axes[0, idx]
            avg_cosines, drop_rates_cos, model_keys_cos, performances_cos = [], [], [], []
            
            for model in model_list:
                if key not in avg_cosine_data or model not in avg_cosine_data[key]:
                    continue
                if model not in query_variations_data[key] or method not in query_variations_data[key][model]:
                    continue
                if key not in model_performance_data or model not in model_performance_data[key]:
                    continue
                cos_mean, _ = avg_cosine_data[key][model]
                drop_rate, _ = query_variations_data[key][model][method]
                avg_cosines.append(cos_mean)
                drop_rates_cos.append(drop_rate * 100)
                model_keys_cos.append(model)
                performances_cos.append(model_performance_data[key][model])
            
            pearson_r_cos, pearson_p_cos = None, None
            if len(avg_cosines) >= 3:
                pearson_r_cos, pearson_p_cos = pearsonr(np.array(avg_cosines), np.array(drop_rates_cos))
            
            # Draw the scatter plot and collect the legend (using color mapping)
            for i, model in enumerate(model_keys_cos):
                color = cmap(norm(performances_cos[i]))
                scatter = ax_cos.scatter(avg_cosines[i], drop_rates_cos[i], s=250, alpha=0.9, zorder=5,
                              marker=model_markers[model], color=color,
                              edgecolors='black', linewidths=0.5)
                last_scatter = scatter
                if idx == 0:
                    # Create a scatter plot with the actual performance color for the legend
                    legend_scatter = ax_cos.scatter([], [], s=250, alpha=0.9,
                                  marker=model_markers[model], color=color,
                                  edgecolors='black', linewidths=0.5)
                    legend_handles.append(legend_scatter)
                    legend_labels_list.append(model_display_names[model])
            
            ax_cos.set_title(attack_method_display_names[method], fontsize=20, fontweight='bold')
            ax_cos.set_xlabel('Avg. Cosine Similarity', fontsize=20, labelpad=-1.8)
            if idx == 0:
                ax_cos.set_ylabel(f'Drop Rate on {dataset_display_names[dataset][split]} (%)', fontsize=20)
            ax_cos.grid(True, linestyle='--', alpha=0.5)
            
            if pearson_r_cos is not None:
                p_sig = "**" if pearson_p_cos < 0.01 else ("*" if pearson_p_cos < 0.05 else " n.s.")
                textstr = f'r = {pearson_r_cos:.2f} (p={pearson_p_cos:.2f}){p_sig}'
                # box_color = 'lightgreen' if pearson_p_cos < 0.05 else 'mistyrose'
                box_color = 'lightgreen' if pearson_p_cos < 0.05 else '#F2F2F2'
                props = dict(boxstyle='round,pad=0.15', facecolor=box_color, alpha=0.8, edgecolor='#B5B5B5')
                ax_cos.text(0.5, -0.28, textstr, transform=ax_cos.transAxes, fontsize=20,
                           ha='center', verticalalignment='top', bbox=props, zorder=4)
            
            # === Second row: IsoScore ===
            ax_iso = axes[1, idx]
            isoscores, drop_rates_iso, model_keys_iso, performances_iso = [], [], [], []
            
            for model in model_list:
                if key not in isoscore_data or model not in isoscore_data[key]:
                    continue
                if model not in query_variations_data[key] or method not in query_variations_data[key][model]:
                    continue
                if key not in model_performance_data or model not in model_performance_data[key]:
                    continue
                iso_mean, _ = isoscore_data[key][model]
                drop_rate, _ = query_variations_data[key][model][method]
                isoscores.append(iso_mean)
                drop_rates_iso.append(drop_rate * 100)
                model_keys_iso.append(model)
                performances_iso.append(model_performance_data[key][model])
            
            # Calculate the correlation
            pearson_r_iso, pearson_p_iso = None, None
            if len(isoscores) >= 3:
                pearson_r_iso, pearson_p_iso = pearsonr(np.array(isoscores), np.array(drop_rates_iso))
            
            # Draw the scatter plot (using color mapping)
            for i, model in enumerate(model_keys_iso):
                color = cmap(norm(performances_iso[i]))
                ax_iso.scatter(isoscores[i], drop_rates_iso[i], s=250, alpha=0.9, zorder=5,
                                        marker=model_markers[model], color=color,
                                        edgecolors='black', linewidths=0.5)
            
            ax_iso.set_xlabel('IsoScore', fontsize=20, labelpad=-1.8)
            if idx == 0:
                ax_iso.set_ylabel(f'Drop Rate on {dataset_display_names[dataset][split]} (%)', fontsize=20)
            ax_iso.grid(True, linestyle='--', alpha=0.5)
            
            # Correlation text
            if pearson_r_iso is not None:
                p_sig = "**" if pearson_p_iso < 0.01 else ("*" if pearson_p_iso < 0.05 else " n.s.")
                textstr = f'r = {pearson_r_iso:.2f} (p={pearson_p_iso:.2f}){p_sig}'
                # box_color = 'lightgreen' if pearson_p_iso < 0.05 else 'mistyrose'
                box_color = 'lightgreen' if pearson_p_iso < 0.05 else '#F2F2F2'
                props = dict(boxstyle='round,pad=0.15', facecolor=box_color, alpha=0.8, edgecolor='#B5B5B5')
                ax_iso.text(0.5, -0.27, textstr, transform=ax_iso.transAxes, fontsize=20,
                           ha='center', verticalalignment='top', bbox=props, zorder=4)
        
        # The scatter plot legend is placed in the upper left (adjust the position closer to the main figure)
        fig.legend(legend_handles, legend_labels_list, loc='upper left',
                   ncol=len(legend_labels_list), fontsize=20, frameon=True,
                   bbox_to_anchor=(0.13, 1.03), handletextpad=0.1, columnspacing=1.0)
        
        # The color legend (colorbar) is placed in the upper right (adjust the position closer to the main figure)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.65, 0.99, 0.15, 0.02])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=20)
        # Add the border (similar to frameon=True)
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        # Put the labels on the right side of the colorbar, aligned with the colorbar
        cbar_ax.text(1.05, 0.5, 'nDCG@10 (%)', fontsize=20, 
                     transform=cbar_ax.transAxes, va='center', ha='left')
        
        plt.subplots_adjust(wspace=0.12, hspace=0.45)
        
        save_path = f'figs/isoscore_and_avg_cosine_vs_query_variations_{dataset}_{split}.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved: {save_path}")




 
def model_embedding_isoscore_and_avg_cosine_and_asr_20_horizontal():
    """
    Merge the model_embedding_isoscore_and_asr_20 function and the model_embedding_average_cosine_similarity_and_asr_20 function
    Draw a 1*2 figure, left is Avg Cosine vs ASR@20, right is IsoScore vs ASR@20
    Use the same scatter style as model_embedding_isoscore_and_avg_cosine_and_query_variations
    (different marker + color depth represents different models), but without adding a legend (because it is placed together with other figures)
    """
    from eval_attack_data_statistics import document_attack_main
    from scipy.stats import pearsonr, linregress
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Get ASR@20 data
    asr_20_data = document_attack_main()
    model_performance_data = model_performance()
    
    # Read IsoScore data
    if os.path.exists('data/isoscore_data_100k.json'):
        with open('data/isoscore_data_100k.json', 'r') as f:
            isoscore_data = json.load(f)
    else:
        print("Warning: data/isoscore_data_100k.json not found.")
        return
    
    # Read Avg Cosine data
    if os.path.exists('data/average_cosine_similarity_data_100k.json'):
        with open('data/average_cosine_similarity_data_100k.json', 'r') as f:
            avg_cosine_data = json.load(f)
    else:
        print("Warning: data/average_cosine_similarity_data_100k.json not found.")
        return
    
    model_list = ["bge_m3", "qwen3", "linq", "gte", "reasonir", "diver", "bge_reasoner"]
    model_display_names = {
        "bge_m3": "BGE-M3", "qwen3": "Qwen3", "linq": "Linq", "gte": "GTE",
        "reasonir": "ReasonIR", "diver": "DIVER", "bge_reasoner": "ReasonEmbed"
    }
    
    # Same marker definition as model_embedding_isoscore_and_avg_cosine_and_query_variations
    model_markers = {
        "bge_m3": "o", "qwen3": "s", "linq": "^", "gte": "D",
        "reasonir": "v", "diver": "p", "bge_reasoner": "*",
    }
    
    dataset_list = ["nq", "hotpotqa", "msmarco"]
    dataset_display_names = {
        "nq": {"test": "NQ"}, "hotpotqa": {"test": "HotpotQA"},
        "msmarco": {"dev": "MS MARCO"},
    }
    
    attack_num = 50
    
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    for dataset in dataset_list:
        split = 'dev' if dataset == 'msmarco' else 'test'
        key = f"{dataset}_{split}"
        
        # Collect all performance values of the dataset, for uniform colorbar range
        all_performances = []
        for model in model_list:
            if key in model_performance_data and model in model_performance_data[key]:
                all_performances.append(model_performance_data[key][model])
        
        if not all_performances:
            print(f"Skipping {key}: No performance data found.")
            continue
        
        vmin, vmax = min(all_performances), max(all_performances)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps['magma_r']
        
        # Create a large figure with 1 row 2 columns (horizontal layout)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2 ))
        
        # === Left figure: Avg Cosine vs ASR@20 ===
        ax_cos = axes[0]
        avg_cosines, asr_20s_cos, model_keys_cos, performances_cos = [], [], [], []
        
        for model in model_list:
            if key not in avg_cosine_data or model not in avg_cosine_data[key]:
                continue
            if key not in asr_20_data or model not in asr_20_data[key]:
                continue
            if attack_num not in asr_20_data[key][model]:
                continue
            if key not in model_performance_data or model not in model_performance_data[key]:
                continue
            
            cos_mean, _ = avg_cosine_data[key][model]
            asr_20_mean, _ = asr_20_data[key][model][attack_num]
            avg_cosines.append(cos_mean)
            asr_20s_cos.append(asr_20_mean)
            model_keys_cos.append(model)
            performances_cos.append(model_performance_data[key][model])
        
        # Calculate the correlation
        pearson_r_cos, pearson_p_cos, partial_r_cos, partial_p_cos = None, None, None, None
        if len(avg_cosines) >= 3:
            pearson_r_cos, pearson_p_cos = pearsonr(np.array(avg_cosines), np.array(asr_20s_cos))
            # Partial correlation
            performance_arr = np.array(performances_cos)
            slope1, int1, _, _, _ = linregress(performance_arr, np.array(avg_cosines))
            residual1 = np.array(avg_cosines) - (slope1 * performance_arr + int1)
            slope2, int2, _, _, _ = linregress(performance_arr, np.array(asr_20s_cos))
            residual2 = np.array(asr_20s_cos) - (slope2 * performance_arr + int2)
            partial_r_cos, partial_p_cos = pearsonr(residual1, residual2)
        
        # Draw the scatter plot (using different markers and colors)
        for i, model in enumerate(model_keys_cos):
            color = cmap(norm(performances_cos[i]))
            ax_cos.scatter(avg_cosines[i], asr_20s_cos[i], s=250, alpha=0.9, zorder=5,
                          marker=model_markers[model], color=color,
                          edgecolors='black', linewidths=0.5)
        
        ax_cos.set_xlabel('Avg. Cosine Similarity', fontsize=20, labelpad=-1.8)
        ax_cos.set_ylabel(f'ASR@20 on {dataset_display_names[dataset][split]} (%)', fontsize=20)
        ax_cos.grid(True, linestyle='--', alpha=0.5)
        
        if pearson_r_cos is not None:
            p_sig = "**" if pearson_p_cos < 0.01 else ("*" if pearson_p_cos < 0.05 else " n.s.")
            # partial_sig = "**" if partial_p_cos < 0.01 else ("*" if partial_p_cos < 0.05 else " n.s.")
            textstr = f'r = {pearson_r_cos:.2f} (p={pearson_p_cos:.2f}){p_sig}'
            # textstr += f'\nPartial r = {partial_r_cos:.3f} (p={partial_p_cos:.3f}){partial_sig}'
            # box_color = 'lightgreen' if pearson_p_cos < 0.05 else 'mistyrose'
            box_color = 'lightgreen' if pearson_p_cos < 0.05 else '#F2F2F2'
            props = dict(boxstyle='round,pad=0.15', facecolor=box_color, alpha=0.8, edgecolor='#B5B5B5')
            ax_cos.text(0.5, -0.27, textstr, transform=ax_cos.transAxes, fontsize=20,
                       ha='center', verticalalignment='top', bbox=props, zorder=4)
        
        # === Right figure: IsoScore vs ASR@20 ===
        ax_iso = axes[1]
        isoscores, asr_20s_iso, model_keys_iso, performances_iso = [], [], [], []
        
        for model in model_list:
            if key not in isoscore_data or model not in isoscore_data[key]:
                continue
            if key not in asr_20_data or model not in asr_20_data[key]:
                continue
            if attack_num not in asr_20_data[key][model]:
                continue
            if key not in model_performance_data or model not in model_performance_data[key]:
                continue
            
            iso_mean, _ = isoscore_data[key][model]
            asr_20_mean, _ = asr_20_data[key][model][attack_num]
            isoscores.append(iso_mean)
            asr_20s_iso.append(asr_20_mean)
            model_keys_iso.append(model)
            performances_iso.append(model_performance_data[key][model])
        
        pearson_r_iso, pearson_p_iso, partial_r_iso, partial_p_iso = None, None, None, None
        if len(isoscores) >= 3:
            pearson_r_iso, pearson_p_iso = pearsonr(np.array(isoscores), np.array(asr_20s_iso))
            performance_arr = np.array(performances_iso)
            slope1, int1, _, _, _ = linregress(performance_arr, np.array(isoscores))
            residual1 = np.array(isoscores) - (slope1 * performance_arr + int1)
            slope2, int2, _, _, _ = linregress(performance_arr, np.array(asr_20s_iso))
            residual2 = np.array(asr_20s_iso) - (slope2 * performance_arr + int2)
            partial_r_iso, partial_p_iso = pearsonr(residual1, residual2)
        
        # Draw the scatter plot (using different markers and colors)
        for i, model in enumerate(model_keys_iso):
            color = cmap(norm(performances_iso[i]))
            ax_iso.scatter(isoscores[i], asr_20s_iso[i], s=250, alpha=0.9, zorder=5,
                          marker=model_markers[model], color=color,
                          edgecolors='black', linewidths=0.5)
        
        ax_iso.set_xlabel('IsoScore', fontsize=20, labelpad=-1.8)
        # The right subplot does not need a y-label, because it is the same as the left subplot
        # ax_iso.set_ylabel(f'ASR@20 on {dataset_display_names[dataset][split]} (%)', fontsize=20)
        ax_iso.grid(True, linestyle='--', alpha=0.5)
        
        if pearson_r_iso is not None:
            p_sig = "**" if pearson_p_iso < 0.01 else ("*" if pearson_p_iso < 0.05 else " n.s.")
            # partial_sig = "**" if partial_p_iso < 0.01 else ("*" if partial_p_iso < 0.05 else " n.s.")
            textstr = f'r = {pearson_r_iso:.2f} (p={pearson_p_iso:.2f}){p_sig}'
            # textstr += f'\nPartial r = {partial_r_iso:.3f} (p={partial_p_iso:.3f}){partial_sig}'
            # box_color = 'lightgreen' if pearson_p_iso < 0.05 else 'mistyrose'
            box_color = 'lightgreen' if pearson_p_iso < 0.05 else '#F2F2F2'
            props = dict(boxstyle='round,pad=0.15', facecolor=box_color, alpha=0.8, edgecolor='#B5B5B5')
            ax_iso.text(0.5, -0.27, textstr, transform=ax_iso.transAxes, fontsize=20,
                       ha='center', verticalalignment='top', bbox=props, zorder=4)
        
        # Do not add a legend (because it is placed together with other figures, the legend is displayed on the side)
        
        # Add the title in the middle of the two subplots at the top
        fig.suptitle('Corpus Poisoning', fontsize=20, fontweight='bold', y=0.97)
        
        plt.subplots_adjust(wspace=0.15)  # Reduce the spacing between subplots to make them closer
        
        save_path = f'figs/isoscore_and_avg_cosine_asr_20_{dataset}_{split}_horizontal.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    # model_embedding_isoscore_and_asr_20() # 1.draw the scatter plot of IsoScore vs ASR@20
    # model_embedding_isoscore_and_query_variations() # 2.draw the scatter plot of IsoScore vs Query Variations
    model_embedding_isoscore_and_avg_cosine_and_query_variations() # 3.draw the scatter plot of Avg Cosine and IsoScore vs Query Variations
    # model_embedding_isoscore_and_avg_cosine_and_asr_20_horizontal() # 4.draw the scatter plot of Avg Cosine and IsoScore vs ASR@20