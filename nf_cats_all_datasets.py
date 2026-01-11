'''
Perform  query classification on each query in the datasets of several benchmarks.
'''

import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import argparse
from tqdm import tqdm
from utils.utils import move_to_device

from utils.load_data import load_beir_data
from utils.logging import LoggingHandler
import pandas as pd
 

from transformers import AutoTokenizer

from utils.nfqa_model import RobertaNFQAClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nfqa_model = RobertaNFQAClassification.from_pretrained("Lurunchik/nf-cats") # This is the model used to classify the query type.
nfqa_model.to(device)

nfqa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

def get_nfqa_category_prediction(text):
    tokens = nfqa_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    tokens = move_to_device(tokens, device)
    output = nfqa_model(**tokens)
    index = output.logits.argmax()
    return nfqa_model.config.id2label[int(index)]

def main():

    scores_save_root = os.path.abspath(os.path.dirname(__file__))
    scores_save_path = os.path.join(
        scores_save_root,
        f'data/'
    )

    dataset_list =[
    "trec-covid",
    "nfcorpus",
    "hotpotqa",
    "fiqa",
    "arguana",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "nq",
    "fever",
    "climate-fever",
    "scifact",
    "webis-touche2020",
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
    "browsecomp_plus",
    'msmarco',
    ]

    # Create an empty DataFrame to store the results.
    # results_df = pd.DataFrame(columns=['query_id', 'dataset_name', 'query_type'])
    results_list = []

    model_name_list = ['bm25','contriever', 'bge_m3', 'qwen3', 'linq', 'gte', 'reasonir', 'diver','bge_reasoner']

    for dataset in dataset_list:
        split = 'test'
 
        data_output_dic = load_beir_data(dataset, split=split)  # # This function automatically handles various situations, and the queries for a dataset are in a single file, regardless of whether they are train, test, or dev.

        corpus, queries, qrels = data_output_dic['corpus'], data_output_dic['queries'], data_output_dic['qrels']
        queries_raw = data_output_dic['queries_raw'] if 'queries_raw' in data_output_dic else None

        for query_id, query_text in tqdm(queries.items(), desc=f"Processing {dataset}/{split}", total=len(queries)):
            new_query_id = dataset + '__'+ split + '__'+ query_id
            category = get_nfqa_category_prediction(query_text)
            # Add new row to DataFrame
            # Add to list
            dic_result= {
                'query_id': new_query_id,
                'dataset_name': dataset,
                'raw_query_id': query_id,
                'query_type': category,
                'query_text': query_text,
                'split': split,

            }

            results_list.append(dic_result )

        print(f"Finished classifying queries for dataset: {dataset}")
 
    # Create DataFrame in one go
    results_df = pd.DataFrame(results_list)
        # pass
    print(results_df.shape)
    print(results_df.head())

    # Save the results to a CSV file
    results_df.to_csv(os.path.join(scores_save_path, 'nf_cats_all_datasets.csv'), index=False)
    print(f"Results saved to {os.path.join(scores_save_path, 'nf_cats_all_datasets.csv')}")
    # I checked this file and all query_ids are unique; there are no duplicate rows.

def combine_scores():
    # This section adds the scores of each model for each query to a larger table.

    # This is the original data; the original data table needs to be modified and integrated to obtain a new table.
    data = pd.read_csv('data/nf_cats_all_datasets.csv')
    # Set the query_id as the index, while preserving the original query_id column
    data = data.set_index('query_id', drop=False) # The split here is always test

    scores_save_root = os.path.abspath(os.path.dirname(__file__))

    dataset_list =[
    "trec-covid",
    "nfcorpus",
    "hotpotqa",
    "fiqa",
    "arguana",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "nq",
    "fever",
    "climate-fever",
    "scifact",
    "webis-touche2020",
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
    "browsecomp_plus",
    'msmarco',
    ]

    model_name_list = ['bm25','contriever', 'bge_m3', 'qwen3', 'linq', 'gte', 'reasonir', 'diver','bge_reasoner']
    for model_name in model_name_list:
        #Create a new data column and assign it to Nan
        data[f'{model_name}_score_ndcg@10'] = np.nan

        for dataset in dataset_list:
            if 'browsecomp_plus' == dataset:
                splits = ['golds', 'evidence']
            elif 'msmarco' == dataset:
                splits = ['dev', 'trec_dl19', 'trec_dl20']
            else:
                splits = ['test']

            for split in splits:

                # Create a new data column

                scores_save_path = os.path.join(
                    scores_save_root,
                    f'output/scores/{model_name}/{dataset}_{split}/'
                )

                # Read the merged_scores.json file query-level performance scores
                merged_scores_file = os.path.join(scores_save_path, 'merged_scores.json')
                if not os.path.exists(merged_scores_file):
                    raise FileNotFoundError(f"File not found: {merged_scores_file}")

                # Read the JSON file
                try:
                    with open(merged_scores_file, 'r') as f:
                        merged_scores = json.load(f)  # This is the read operation
                    print(f'  ✓ Loaded {len(merged_scores)} items from merged_scores.json')
                except Exception as e:
                    print(f'  ✗ Error reading file: {str(e)}')
                    continue

                for raw_query_id in merged_scores.keys():
                    query_id = dataset + '__' + split + '__' + raw_query_id
                    search_query_id = dataset + '__test__' + raw_query_id # This is because all are test in the data
                    data.loc[search_query_id, f'{model_name}_score_ndcg@10'] = merged_scores[raw_query_id]['ndcg_cut_10']

    # Remove all rows where '{model_name}_score_ndcg@10' is still np.nan
    score_columns = [f'{model_name}_score_ndcg@10' for model_name in model_name_list]
    # Only keep rows where at least one model score is not NaN
    rows_before = len(data)
    data = data.dropna(subset=score_columns, how='all')
    rows_after = len(data)
    print(f"Dropped {rows_before - rows_after} rows (from {rows_before} to {rows_after}) where all score columns were NaN")
    
    # Save the data
    data.to_csv('data/nf_cats_all_datasets_with_scores.csv', index=False)
    print(f"Results saved to  'data/nf_cats_all_datasets_with_scores.csv')")


def draw_query_type_distribution():
    import matplotlib.pyplot as plt
    import os
    
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  # Math formulas also use a similar font to Times
    
    data = pd.read_csv('data/nf_cats_all_datasets.csv')
    
    # Count the number of each query_type
    query_type_counts = data['query_type'].value_counts()
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set the colors
    colors = plt.cm.Set3(range(len(query_type_counts)))
    
    # Draw a pie chart, display the percentage and the number
    wedges, texts, autotexts = ax.pie(
        query_type_counts.values, 
        labels=query_type_counts.index, 
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(query_type_counts.values))})',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 16}
    )
    
    # Set the title
    ax.set_title('Query Type Distribution', fontsize=18, fontweight='bold')
    
    # Ensure it is a circle
    ax.axis('equal')
    
    # Create the figs folder (if it does not exist)
    os.makedirs('figs', exist_ok=True)
    
    # Save as PDF
    plt.tight_layout()
    plt.savefig('figs/query_type_distribution.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    print("Saved: figs/query_type_distribution.pdf")
    print(f"\nQuery Type Count:")
    print(query_type_counts)
    
    # ========== Bar Chart ==========
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate the percentage
    total = query_type_counts.sum()
    percentages = (query_type_counts.values / total) * 100
    
    # Draw a bar chart (uniform color)
    bars = ax.bar(query_type_counts.index, percentages, color='#C47070', edgecolor='black', linewidth=0.5) #'#1399B2'
    
    # Add the percentage label on top of each bar
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=16)
    
    ax.set_xlabel('Query Type', fontsize=18, fontweight='bold', labelpad=-1)
    ax.set_ylabel('Percentage (%)', fontsize=18)
    # ax.set_title('Query Type Distribution', fontsize=14, fontweight='bold')
    
    # Rotate the x-axis labels to prevent overlap
    ax.tick_params(axis='x', labelsize=18)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    plt.yticks(fontsize=18)
    
    # Set the y-axis tick interval to 10
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(10))
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Set the y-axis upper limit, so that the highest bar is away from the top a bit to make the figure more beautiful
    ax.set_ylim(0, max(percentages) * 1.15)
    
    plt.tight_layout()
    plt.savefig('figs/query_type_distribution_bar.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    print("Saved: figs/query_type_distribution_bar.pdf")


if __name__ == "__main__":
    main() # This function is to classify the query type

    combine_scores() # This function adds the scores to the data after the main function
    # draw_query_type_distribution()
