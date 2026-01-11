'''
Calculate the spectral norm statistics (mean, max, min) of each layer (Linear Layer) in the LLM model.
'''
import os
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from draw_isoscore import model_performance
from eval_attack_data_statistics import document_attack_main

def analyze_spectral_norms(model, exclude_patterns=None):
    """
    Calculate the spectral norm distribution of all linear layers (Linear Layer) in the model.
    
    Args:
        model: The model to analyze
        exclude_patterns: The list of layer name patterns to exclude, e.g. ['pooler', 'lm_head']
                          These layers are usually not used in the embedding extraction
    """
    if exclude_patterns is None:
        # Default to exclude pooler and lm_head, because the embedding model usually does not use these layers
        exclude_patterns = ['pooler', 'lm_head']
    
    spectral_norms = []
    layer_names = []

    print(f"Calculating the spectral norm of each layer... (exclude patterns: {exclude_patterns})")
    
    for name, module in model.named_modules():
        # We mainly focus on Linear layers (including Q, K, V, FFN, Output projection)
        if isinstance(module, nn.Linear):
            # Check if this layer should be excluded
            should_exclude = any(pattern in name for pattern in exclude_patterns)
            if should_exclude:
                print(f"  Skipping layer: {name} (match exclude pattern)")
                continue
            
            # Get the weight matrix W
            # Convert to float32 to ensure numerical stability of SVD/norm calculation
            weight = module.weight.data.float()
            
            # If GPU is available, move to GPU for accelerated calculation
            if torch.cuda.is_available():
                weight = weight.cuda()
                
            # Calculate the spectral norm (2-norm of the matrix)
            # torch.linalg.matrix_norm(..., ord=2) 等价于最大奇异值
            try:
                s_norm = torch.linalg.matrix_norm(weight, ord=2).item()
            except Exception as e:
                print(f"Warning: Calculation failed for {name}, error: {e}")
                continue
            
            spectral_norms.append(s_norm)
            layer_names.append(name)

    # Convert to numpy for easier statistics
    spectral_norms = np.array(spectral_norms)
    
    # if len(spectral_norms) == 0:
    #     return {
    #         "Mean_Spectral_Norm": 0,
    #         "Max_Spectral_Norm": 0,
    #         "Min_Spectral_Norm": 0,
    #         "Std_Spectral_Norm": 0,
    #         "Layer_Count": 0,
    #         "Max_Layer_Name": "None"
    #     }

    max_idx = np.argmax(spectral_norms)
    
    # Record the spectral norm of the last Linear layer (possibly related to robustness)
    last_layer_norm = spectral_norms[-1] if len(spectral_norms) > 0 else 0
    last_layer_name = layer_names[-1] if len(layer_names) > 0 else "None"
    
    stats = {
        "Mean_Spectral_Norm": float(np.mean(spectral_norms)),
        "Max_Spectral_Norm": float(np.max(spectral_norms)),
        "Min_Spectral_Norm": float(np.min(spectral_norms)),
        "Std_Spectral_Norm": float(np.std(spectral_norms)),
        "Layer_Count": int(len(spectral_norms)),
        "Max_Layer_Name": layer_names[max_idx], # Record the layer where the maximum value appears
        "Last_Layer_Spectral_Norm": float(last_layer_norm), # Last layer spectral norm
        "Last_Layer_Name": last_layer_name, # Last layer name
        "Raw_Spectral_Norms": spectral_norms.tolist(), # Convert to list for JSON serialization
    }
    
    return stats

def all_model_spectral_norm():
    model_name_dict = {
        "contriever": "facebook/contriever-msmarco",
        "bge_m3": "BAAI/bge-m3",
        "qwen3" : "Qwen/Qwen3-Embedding-8B",
        "linq": 'Linq-AI-Research/Linq-Embed-Mistral' ,
        "gte" : "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "reasonir" : "reasonir/ReasonIR-8B",
        "diver" : 'AQ-MedAI/Diver-Retriever-4B',
        "bge_reasoner" : "hanhainebula/reason-embed-qwen3-8b-0928",
        "qwen3_4B" : "Qwen/Qwen3-Embedding-4B",
        "qwen3_0.6B" : "Qwen/Qwen3-Embedding-0.6B",
    }

    output_path = "data/spectral_norm_results.json"
    
    # 1. Load existing results, implement incremental update
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"Existing results loaded, containing models: {list(all_results.keys())}")
        except json.JSONDecodeError:
            print("Existing results file corrupted, will start over.")
            all_results = {}
    else:
        all_results = {}

    # 2. Traverse the model list
    for model_name, model_path in model_name_dict.items():
        if model_name in all_results:
            print(f"Model {model_name} already has results, skipping.")
            continue
            
        print(f"\n======== Processing model: {model_name} ========")
        try:
            model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=os.getenv('HF_HOME')
            )
        except Exception as e:
            print(f"Model {model_name} loading failed: {e}")
            continue

        stats = analyze_spectral_norms(model)
        
        print(f"Model name: {model_name}")
        print(f"Model average spectral norm: {stats['Mean_Spectral_Norm']:.4f}")
        print(f"Max spectral norm: {stats['Max_Spectral_Norm']:.4f} (Layer: {stats['Max_Layer_Name']})")
        print(f"Last layer spectral norm: {stats['Last_Layer_Spectral_Norm']:.4f} (Layer: {stats['Last_Layer_Name']})")

        # Save to result dictionary (stats is already JSON serializable)
        all_results[model_name] = stats

        # Real-time save, prevent loss due to interruption
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Release memory
        del model
        torch.cuda.empty_cache()

    print(f"\nAll results saved to: {output_path}")
    return all_results


def model_spectral_norm_and_query_variations():
    """
    This function is to model the relationship between the average spectral norm and the query variations drop rate
    Contains 5 query variations, each drawn one sub-figure, placed on the same figure (1 row 5 columns)
    The color depth changes according to model_performance_data (NDCG@10)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from scipy.stats import pearsonr, linregress
    from eval_attack_data_statistics import query_attack_main
    
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    # 1. Get the data
    query_variations_data = query_attack_main(draw_pic=False)
    model_performance_data = model_performance()
    
    # Read the spectral norm data
    spectral_norm_path = "data/spectral_norm_results.json"
    if os.path.exists(spectral_norm_path):
        with open(spectral_norm_path, "r", encoding='utf-8') as f:
            spectral_norm_data = json.load(f)
    else:
        print(f"Warning: {spectral_norm_path} not found. Please run all_model_spectral_norm first.")
        return
    
    model_list = [
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

    dataset_list = ["nq", "hotpotqa", "msmarco"]
    
    dataset_display_names = {
        "nq": {"test": "NQ"},
        "hotpotqa": {"test": "HotpotQA"},
        "msmarco": {"dev": "MS MARCO"},
    }

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
    
    # Define the marker shape for each model
    model_markers = {
        "bge_m3": "o",        # Circle
        "qwen3": "s",         # Square
        "linq": "^",          # Up triangle
        "gte": "D",           # Diamond
        "reasonir": "v",      # Down triangle
        "diver": "p",         # Pentagon
        "bge_reasoner": "*",  # Star
    }
    
    os.makedirs('figs', exist_ok=True)
    
    # For collecting all plotting data
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
        
        # Collect all performance values for this dataset, for uniform colorbar range
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
        
        plot_data[key] = {}
            
        # Create a large figure with 1 row 5 columns
        fig, axes = plt.subplots(1, 5, figsize=(25, 4), sharey=False)
        
        # For collecting legend handles
        legend_handles = []
        legend_labels_list = []
        
        for idx, method in enumerate(query_attack_method_list):
            ax = axes[idx]
            
            mean_norms = []
            drop_rates = []
            model_keys = []
            performances = []
            
            for model in model_list:
                # Check data availability
                if model not in spectral_norm_data:
                    continue
                if model not in query_variations_data[key]:
                    continue
                if method not in query_variations_data[key][model]:
                    continue
                if key not in model_performance_data or model not in model_performance_data[key]:
                    continue
                
                mean_norm = spectral_norm_data[model]['Mean_Spectral_Norm']
                drop_rate, _ = query_variations_data[key][model][method]
                
                mean_norms.append(mean_norm)
                drop_rates.append(drop_rate * 100)  # Convert to percentage
                model_keys.append(model)
                performances.append(model_performance_data[key][model])
            
            # Calculate the correlation
            mean_norms_arr = np.array(mean_norms)
            drop_rates_arr = np.array(drop_rates)
            
            pearson_r, pearson_p = None, None
            if len(mean_norms) >= 3:
                pearson_r, pearson_p = pearsonr(mean_norms_arr, drop_rates_arr)

            # Collect the plotting data of this method
            plot_data[key][method] = {
                'mean_norms': mean_norms,
                'drop_rates': drop_rates,
                'model_keys': model_keys,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
            }
            
            # Scatter Plot - Use different marker shapes and color mapping
            for i, model in enumerate(model_keys):
                color = cmap(norm(performances[i]))
                scatter = ax.scatter(mean_norms[i], drop_rates[i], s=250, alpha=0.9, zorder=5,
                                   marker=model_markers[model], color=color, 
                                   edgecolors='black', linewidths=0.5)
                # Collect the legend information in the first subplot
                if idx == 0:
                    # Create a scatter plot with the actual performance color for the legend
                    legend_scatter = ax.scatter([], [], s=250, alpha=0.9,
                                  marker=model_markers[model], color=color,
                                  edgecolors='black', linewidths=0.5)
                    legend_handles.append(legend_scatter)
                    legend_labels_list.append(model_display_names[model])

            ax.set_title(attack_method_display_names[method], fontsize=20, fontweight='bold')
            ax.set_xlabel('Mean Spectral Norm', fontsize=20, labelpad=-1.8)
            if idx == 0:
                ax.set_ylabel(f'Drop Rate on {dataset_display_names[dataset][split]} (%)', fontsize=20)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Correlation text
            if pearson_r is not None:
                p_sig = "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else " n.s.")
                textstr = f'r = {pearson_r:.2f} (p={pearson_p:.2f}){p_sig}'
                
                box_color = 'lightgreen' if pearson_p < 0.05 else '#F2F2F2'
                props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='#B5B5B5')
                ax.text(0.5, -0.27, textstr, transform=ax.transAxes, fontsize=20,
                        ha='center', verticalalignment='top', bbox=props, zorder=4)
                
                print(f"{dataset} - {method}: r = {pearson_r:.4f}, p = {pearson_p:.4f}")

        # Scatter plot legend placed at the top left
        fig.legend(legend_handles, legend_labels_list, loc='upper left', 
                   ncol=len(legend_labels_list), fontsize=20, frameon=True,
                   bbox_to_anchor=(0.13, 1.09), handletextpad=0.1, columnspacing=1.0)
        
        # Color legend (colorbar) placed at the top right
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.65, 0.97, 0.15, 0.025])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=18)
        # Add border
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        # Put the labels on the right side of the colorbar
        cbar_ax.text(1.05, 0.5, 'nDCG@10 (%)', fontsize=20, 
                     transform=cbar_ax.transAxes, va='center', ha='left')
        
        plt.subplots_adjust(wspace=0.15, top=0.80)
        
        save_path = f'figs/mean_spectral_norm_vs_query_variations_{dataset}_{split}.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved: {save_path}")

    return plot_data



def model_spectral_norm_and_asr20():
    """
    Draw the scatter plot of Mean Spectral Norm vs ASR@20 for each dataset
    Use the same marker shapes as model_spectral_norm_and_query_variations
    The color depth changes according to model_performance_data (NDCG@10)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from scipy.stats import pearsonr, linregress
    
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    # Get the data
    asr_20_data = document_attack_main()
    model_performance_data = model_performance()
    
    # Read the spectral norm data
    spectral_norm_path = "data/spectral_norm_results.json"
    if os.path.exists(spectral_norm_path):
        with open(spectral_norm_path, "r", encoding='utf-8') as f:
            spectral_norm_data = json.load(f)
    else:
        print(f"Warning: {spectral_norm_path} not found. Please run all_model_spectral_norm first.")
        return
    
    model_list = ["bge_m3", "qwen3", "linq", "gte", "reasonir", "diver", "bge_reasoner"]
    
    model_display_names = {
        "bge_m3": "BGE-M3", "qwen3": "Qwen3", "linq": "Linq", "gte": "GTE",
        "reasonir": "ReasonIR", "diver": "DIVER", "bge_reasoner": "ReasonEmbed"
    }
    
    # Same marker definition as model_spectral_norm_and_query_variations
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
    
    os.makedirs('figs', exist_ok=True)
    
    for dataset in dataset_list:
        split = 'dev' if dataset == 'msmarco' else 'test'
        key = f"{dataset}_{split}"
        
        # Collect all performance values for this dataset, for uniform colorbar range
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
        
        # Create a single figure
        fig, ax = plt.subplots(figsize=(5, 3.2))
        
        mean_norms = []
        asr_20s = []
        model_keys = []
        performances = []
        
        for model in model_list:
            if model not in spectral_norm_data:
                continue
            if key not in asr_20_data or model not in asr_20_data[key]:
                continue
            if attack_num not in asr_20_data[key][model]:
                continue
            if key not in model_performance_data or model not in model_performance_data[key]:
                continue
            
            mean_norm = spectral_norm_data[model]['Mean_Spectral_Norm']
            asr_20_mean, _ = asr_20_data[key][model][attack_num]
            
            mean_norms.append(mean_norm)
            asr_20s.append(asr_20_mean)
            model_keys.append(model)
            performances.append(model_performance_data[key][model])
        
        # Calculate the correlation
        pearson_r, pearson_p = None, None
        if len(mean_norms) >= 3:
            pearson_r, pearson_p = pearsonr(np.array(mean_norms), np.array(asr_20s))
        
        # Draw the scatter plot (using different markers and colors)
        for i, model in enumerate(model_keys):
            color = cmap(norm(performances[i]))
            ax.scatter(mean_norms[i], asr_20s[i], s=250, alpha=0.9, zorder=5,
                      marker=model_markers[model], color=color,
                      edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('Mean Spectral Norm', fontsize=20, labelpad=-1.8)
        ax.set_ylabel(f'ASR@20 on {dataset_display_names[dataset][split]} (%)', fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Correlation text
        if pearson_r is not None:
            p_sig = "**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else " n.s.")
            textstr = f'r = {pearson_r:.2f} (p={pearson_p:.2f}){p_sig}'
            box_color = 'lightgreen' if pearson_p < 0.05 else '#F2F2F2'
            props = dict(boxstyle='round,pad=0.15', facecolor=box_color, alpha=0.8, edgecolor='#B5B5B5')
            ax.text(0.5, -0.27, textstr, transform=ax.transAxes, fontsize=20,
                   ha='center', verticalalignment='top', bbox=props, zorder=4)
            
            print(f"{dataset}: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
        
        # Add the title at the top
        fig.suptitle('Corpus Poisoning', fontsize=20, fontweight='bold', y=0.97)
        
        plt.subplots_adjust(wspace=0.15)
        
        save_path = f'figs/mean_spectral_norm_vs_asr20_{dataset}_{split}.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved: {save_path}")



if __name__ == "__main__":
    all_model_spectral_norm() # 1.first run this function to get the spectral norm data
 
    # model_spectral_norm_and_query_variations() # 2.draw the scatter plot of Mean Spectral Norm vs Query Variations

    # model_spectral_norm_and_asr20() # 3.draw the scatter plot of Mean Spectral Norm vs ASR@20   
