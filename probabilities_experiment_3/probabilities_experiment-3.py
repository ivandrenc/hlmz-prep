import gc
import json
import os
import re
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn.functional as F
from circuitsvis.attention import attention_heads, attention_patterns
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global settings
torch.set_grad_enabled(False)  # to disable gradients -> faster computiations
torch.set_printoptions(sci_mode=False)
# Ensure GPU acceleration is enabled on Mac
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
model = None
tokenizer = None
META_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
GOOGLE_GEMMA_2_2B = "google/gemma-2-2b"
dataset = {}
CSV_PATH_DATASET = "dataset/examples-trigger.csv"
sns.set(style="whitegrid")



models = [META_LLAMA_3_2_3B]

# Experiment description: We run the model with the examples in no order and with the order switched, find the top 5
# induction heads by looking into the induction scores of both ran examples, average the scores across examples and
# select the top 5. Then with the identified heads we get the attention probabilities of the true, false token from every example
# and we average them as well. Then we can plot these results for the top 5 heads as a barplot to see which heads are more
# biased to output false information or correct information. Then we also compute the probabilities of the next token prediction
# by looking into the logits, for the correct token, false token, and the top probable to predict token. We then knock out
# the identified top induction heads one by one, and plotting the expected next token probability for the correct and false
# token. We do this top-5-heads times, using all the examples, and finally we run the process without knocking out the heads
# just to have some comparison data.


def initialize_model(model_name: str):
    tokenizer_name = model_name
    # Initialize model and tokenizer
    global model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    if not tokenizer_name:
        tokenizer_name = model_name
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def load_dataset(path_to_csv: str):
    # Check if the file at the given path exists
    if os.path.exists(path_to_csv):
        df = pd.read_csv(path_to_csv)
    else:
        print("File does not exist.", path_to_csv)
        exit(1)

    global dataset
    dataset = df

    # Create a new column "token_probability" for saving up the probabilites of the studied token for all prompts. Initially, 0.
    dataset["attention_probability_first_sentence_token"] = 0
    dataset["attention_probability_second_sentence_token"] = 0
    dataset["attention_probability_first_sentence_token_switched"] = 0
    dataset["attention_probability_second_sentence_token_switched"] = 0


def print_colored_separator(
    color="\033[94m", char="=", length=150, prints_enabled: bool = False
):
    if prints_enabled:
        reset = "\033[0m"  # Reset color
        print(f"{color}{char * length}{reset}")


# Returns the model's output after feeding it with a prompt concatenated prompt_repetitions times and the concatenated prompt tensor
def feed_forward(
    true_sentence: str,
    false_sentence: str,
    prints_enabled: bool = False,
):
    print_colored_separator(prints_enabled)
    # Before proceeding, check that the true_sentence and false_sentence contain the same amount of tokens after tokenizing them.
    # Important!: BOS token is usually not included for counting the tokens of a sentence, when indexing .shape[...]
    true_sentence_token_n = tokenizer(true_sentence, return_tensors="pt")["input_ids"][
        0
    ].shape[0]
    false_sentence_token_n = tokenizer(false_sentence, return_tensors="pt")[
        "input_ids"
    ][0].shape[0]
    if true_sentence_token_n != false_sentence_token_n:
        return None, None, None

    # Extract all the words except the last one, split by space.
    sentence_without_last_token = "".join(true_sentence.rsplit(" ", 1)[:-1])
    # Append the sentence without the last token to the prompt, starting with the true_sentence. This is one-shot learning.
    # Add space token to avoid that the point token "." gets tokenized together with the beginning of the next sentence.
    prompt = true_sentence + "\n" + false_sentence + "\n" + sentence_without_last_token
    token_sequence = tokenizer(prompt, return_tensors="pt")
    # print(f"prompt: {prompt}\ntoken_sequence: {token_sequence}\nNumber of tokens: {len(token_sequence['input_ids'][0])}")
    tokens = token_sequence["input_ids"][0]

    # Feed forward to the model
    global model
    out = model(
        tokens.unsqueeze(0).to(model.device), return_dict=True, output_attentions=True
    )
    # Return the output of the model, the tokenized prompt, number of tokens from the sentences (both sentences should have the same amount of tokens at this point)
    return out, tokens, true_sentence_token_n


def plot_induction_mask_with_plotly(induction_mask, induction_mask_text, prompt):
    # Create a Heatmap with the numeric mask (z) and attach the text
    heatmap = go.Heatmap(
        z=induction_mask,
        text=induction_mask_text,
        hoverinfo="text",  # Only show the text on hover
        colorscale="Blues",  # Choose any Plotly colorscale you like
        showscale=True,
    )

    fig = go.Figure(data=[heatmap])

    # Make the squares actually square by linking x/y scales
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(autorange="reversed"),  # Reverse y-axis so row 0 is at top
        title=f"Induction Mask for prompt: {prompt}\n",
    )

    fig.show()


def create_attention_mask(
    token_sequence: torch.Tensor,
    token_number_sentence: int,
    show_induction_mask: bool = False,
    prints_enabled: bool = False,
):
    print_colored_separator(prints_enabled)
    sequence_length = token_sequence.shape[0]
    induction_mask = torch.zeros(sequence_length, sequence_length).to(float)
    induction_mask_text = np.full((sequence_length, sequence_length), "", dtype=object)

    # Start at the beginning of the second sentence (+1 since BOS token was not counted).
    for i in range(token_number_sentence + 1, sequence_length):
        if token_sequence[i] not in token_sequence[:i]:
            continue
        for j in range(i):
            if token_sequence[i] == token_sequence[j]:
                induction_mask[i, j + 1] = 1
                # Encode to show raw strings (show e.g. new lines tokens)
                induction_mask_text[i, j + 1] = (
                    tokenizer.decode(token_sequence[i])
                    .encode("unicode_escape")
                    .decode("utf-8")
                    + "/"
                    + tokenizer.decode(token_sequence[j + 1])
                    .encode("unicode_escape")
                    .decode("utf-8")
                )

    if show_induction_mask:
        # print("Induction Mask:\n")
        # print(induction_mask)
        # print()
        # print("Induction Mask plot:\n")
        # plt.imshow(induction_mask)
        # plt.show()
        plot_induction_mask_with_plotly(
            induction_mask, induction_mask_text, prompt=tokenizer.decode(token_sequence)
        )
    return induction_mask


def compute_induction_head_scores(
    token_sequence: torch.Tensor, induction_mask: torch.Tensor, model_output
):
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    sequence_length = token_sequence.shape[0]

    tril = torch.tril_indices(
        sequence_length, sequence_length
    )  # gets the indices of elements on and below the diagonal
    induction_flat = induction_mask[tril[0], tril[1]].flatten()

    induction_scores = {}

    for layer in range(num_layers):
        for head in range(num_heads):
            pattern = model_output["attentions"][layer][0][head].cpu().to(float)
            pattern_flat = pattern[tril[0], tril[1]].flatten()
            score = (induction_flat @ pattern_flat) / pattern_flat.sum()
            induction_scores[f"L{layer}_H{head}"] = score.item()

    return induction_scores


def create_heatmap(induction_scores: torch.Tensor):
    print_colored_separator()
    _, ax = plt.subplots()
    print("Heatmap of induction scores across heads and layers: \n")
    sns.heatmap(induction_scores, cbar_kws={"label": "Induction Head Score"}, ax=ax)
    ax.set_ylabel("Layer #")
    ax.set_xlabel("Head #")
    plt.show()


def sort_filter_high_scoring_induction_heads(
    induction_scores: torch.Tensor,
    model_output: any,
    show_induction_heads: bool = False,
    prints_enabled: bool = False,
):
    print_colored_separator(prints_enabled)

    # Get flattened indices sorted by scores in descending order
    sorted_flat_indices = torch.argsort(induction_scores.flatten(), descending=True)

    # Convert flattened indices to 2D indices
    sorted_indices = torch.unravel_index(sorted_flat_indices, induction_scores.shape)

    # Stack the row and column indices for final output
    sorted_indices = torch.stack(sorted_indices, dim=1)

    if show_induction_heads:
        print(
            "Top 5 Induction Heads with the highest induction score - Descending order\n"
        )
        for layer, head in sorted_indices[:5]:
            induction_score = induction_scores[layer][head]
            print(f"Layer: {layer}\nHead: {head}\nInduction Score: {induction_score}")
            plt.imshow(model_output["attentions"][layer][0][head].cpu().float())
            plt.show()
            print()
    return sorted_indices


def token_probability_extraction(
    heads: dict,
    models_output: any,
    token_number_sentence: int,
    prints_enabled: bool = False,
):
    result_true_sentence = {}
    result_false_sentence = {}
    for idx in heads:
        print_colored_separator(prints_enabled)
        layer, head = re.findall(r"\d+", idx)
        probs = models_output["attentions"][int(layer)][0][int(head)]

        # Extract probability of the specified token
        sequence_length = probs.shape[0]
        # First index is y-axis, second is x-axis from the source destination diagram.
        # sequence_length - 1 because we want to index the last token of a sequence.
        # token_number_sentence - 1 because we skip the newline at the end of each sentence.
        probability_token_true_sentence = probs[
            sequence_length - 1, token_number_sentence - 1
        ].item()
        probability_token_false_sentence = probs[
            sequence_length - 1, 2 * token_number_sentence - 1
        ].item()

        # Results for token from true_sentence and false_sentence at current layer and head
        result_true_sentence[f"L{layer}_H{head}"] = probability_token_true_sentence
        result_false_sentence[f"L{layer}_H{head}"] = probability_token_false_sentence
    return json.dumps(result_true_sentence), json.dumps(result_false_sentence)


def logit_probability_extraction(models_output, token_sequence, token_number_sentence):
    # extract the logit probability for last token
    probabilities = F.softmax(models_output["logits"].squeeze(), dim=-1)

    first_sentence_last_token_idx = token_sequence[token_number_sentence - 1]
    second_sentence_last_token_idx = token_sequence[2 * token_number_sentence - 1]

    probability_logits = {}

    first_sentence_prob = probabilities[-1, first_sentence_last_token_idx]
    second_sentence_prob = probabilities[-1, second_sentence_last_token_idx]
    top_token_prob = torch.max(probabilities[-1])
    top_token_idx = torch.argmax(probabilities[-1])

    probability_logits[repr(tokenizer.decode(first_sentence_last_token_idx))] = (
        first_sentence_prob.item()
    )
    probability_logits[repr(tokenizer.decode(second_sentence_last_token_idx))] = (
        second_sentence_prob.item()
    )
    probability_logits["Predicted"] = {f"{repr(tokenizer.decode(top_token_idx))}": top_token_prob.item()}

    probability_logits = json.dumps(probability_logits)

    return probability_logits


def save_probability(
    token_probability: int,
    example_id: int,
    column_name_probability: str,
    prints_enabled: bool = False,
):
    if dataset.empty:
        raise Exception("Dataset is empty")

    print_colored_separator(prints_enabled)
    # Log the probability of the token into its corresponding row and column in the dataset.
    dataset.loc[dataset["example_id"] == example_id, f"{column_name_probability}"] = (
        token_probability
    )
    if prints_enabled:
        print(f"Saved probability for token from example_id: {example_id}\n")


def display_attention_visualizations(
    head_indices: torch.Tensor, token_sequence: torch.Tensor, models_output
):
    # Display attention diagrams
    tokens_vis = tokenizer.tokenize(tokenizer.decode(token_sequence.squeeze()))
    layer, head = head_indices[0]
    return attention_patterns(
        tokens_vis, models_output["attentions"][layer][0]
    ), attention_heads(models_output["attentions"][layer][0], tokens_vis)


def plot_attention_probabilities_tokens_heads_results(save_path: str):
    # Load the DataFrame (assuming df is already loaded)
    # Convert JSON strings to dictionaries
    dataset["true_probs"] = dataset[
        "attention_probability_first_sentence_token_top_induction_heads"
    ].apply(json.loads)
    dataset["false_probs"] = dataset[
        "attention_probability_second_sentence_token_top_induction_heads"
    ].apply(json.loads)

    # Convert probabilities to DataFrame
    true_df = pd.DataFrame(dataset["true_probs"].to_list())
    false_df = pd.DataFrame(dataset["false_probs"].to_list())

    # Add labels for True and False sentences
    true_df["Type"] = "True Sentence"
    false_df["Type"] = "False Sentence"

    # Concatenate both DataFrames
    long_df = pd.concat([true_df, false_df])

    # Convert to long format for seaborn
    long_df = long_df.melt(
        id_vars=["Type"], var_name="L_H_Key", value_name="Probability"
    )

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=long_df, x="L_H_Key", y="Probability", hue="Type")
    ax.get_figure().savefig(f"{save_path}-results-plot.png", dpi=300)


def get_average_across_heads(
    induction_scores, induction_scores_switched, top_k_heads: int
):
    h_sum = defaultdict(float)
    h_count = defaultdict(int)

    for data in induction_scores:
        for l_h in data.items():
            l_h_key, value = l_h
            h_sum[l_h_key] = h_sum[l_h_key] + value
            h_count[l_h_key] = h_count[l_h_key] + 1

    for data in induction_scores_switched:
        for l_h in data.items():
            l_h_key, value = l_h
            h_sum[l_h_key] = h_sum[l_h_key] + value
            h_count[l_h_key] = h_count[l_h_key] + 1

    h_average = {h: h_sum[h] / h_count[h] for h in h_sum}
    # Sort the values of the heads in descending order. Slice the top k heads.
    sorted_heads = dict(
        sorted(h_average.items(), key=itemgetter(1), reverse=True)[:top_k_heads]
    )
    return sorted_heads


def save_result_csv(model_name: str, results_path: str):
    global dataset
    model_name_folder = model_name.split("/")
    folder_path = results_path + "/" + model_name_folder[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)  # Creates parent directories if needed
    model_and_path = f"{model_name_folder[-1]}-results.csv"
    new_file_path = os.path.join(folder_path, model_and_path)
    dataset.to_csv(new_file_path, index=False)

    # Save plot for the current dataset
    model_and_path_image = os.path.join(folder_path, model_name_folder[-1])
    return model_and_path_image


def delete_model():
    global model
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # Clear MPS GPU memory


def calculate_induction_scores(first_sentence: str, second_sentence: str):
    models_output, token_sequence, token_number_sentence = feed_forward(
        true_sentence=first_sentence, false_sentence=second_sentence
    )
    if token_sequence is None:
        # first_sentence and second_sentence have different number of tokens after tokenizing them.
        return pd.Series([pd.NA, pd.NA])

    # Create attention mask
    induction_mask = create_attention_mask(
        token_sequence=token_sequence, token_number_sentence=token_number_sentence
    )

    # compute the induction heads
    induction_scores = compute_induction_head_scores(
        token_sequence=token_sequence,
        induction_mask=induction_mask,
        model_output=models_output,
    )

    # extract the token probability for the true and false sentence
    true_token_probability, false_token_probability = token_probability_extraction(
        heads=induction_scores,
        models_output=models_output,
        token_number_sentence=token_number_sentence,
    )

    # extract the probability of the next token to predict, for the correct token, false token and the top probable token
    probability_logits = logit_probability_extraction(
        models_output=models_output,
        token_sequence=token_sequence,
        token_number_sentence=token_number_sentence,
    )

    return pd.Series(
        [
            induction_scores,
            true_token_probability,
            false_token_probability,
            probability_logits,
        ]
    )


def attention_probs_for_heads(row, top_heads):
    probs_true = json.loads(row["attention_probability_first_sentence_token"])
    probs_false = json.loads(row["attention_probability_second_sentence_token"])
    probs_true_second = json.loads(
        row["attention_probability_first_sentence_token_switched"]
    )
    probs_false_second = json.loads(
        row["attention_probability_second_sentence_token_switched"]
    )

    data_true = {}
    data_false = {}
    for key, _ in top_heads.items():
        data_true[key] = (probs_true[key] + probs_true_second[key]) / 2
        data_false[key] = (probs_false[key] + probs_false_second[key]) / 2
    return pd.Series([json.dumps(data_true), json.dumps(data_false)])

def calculate_logit_probability(logits_example, logits_switched_example):
    logits_example = json.loads(logits_example)
    logits_switched_example = json.loads(logits_switched_example)
    # Store probabilities in a dictionary for averaging
    prob_sums = defaultdict(float)
    prob_counts = defaultdict(int)
    for token, prob in logits_example.items():
        if token != "Predicted":
            prob_sums[token] += prob
            prob_counts[token] += 1
    for token, prob in logits_switched_example.items():
        if token != "Predicted":
            prob_sums[token] += prob
            prob_counts[token] += 1

    averaged_probs = {
        token: prob_sums[token] / prob_counts[token]
        for token in prob_sums
    }
    # Combine Predicted tokens
    predicted = {}
    if "Predicted" in logits_example:
        predicted.update({
            k: v for k, v in logits_example["Predicted"].items()
        })
    if "Predicted" in logits_switched_example:
        for k, v in logits_switched_example["Predicted"].items():
            if k in predicted:
                predicted[k] += v  # Add the probabilities
                predicted[k] /= 2
            else:
                predicted[k] = v  # Set the initial value

    # Create final result
    result = averaged_probs.copy()
    if predicted:
        result["Predicted"] = predicted

    return json.dumps(result)


def plot_logit_probs():
    # Process data
    correct_probs = []
    false_probs = []
    predicted_probs = []

    for row in dataset["result_logit_probability"]:
        parsed = json.loads(row)
        # Get all keys except "Predicted"
        keys = [k for k in parsed.keys() if k != "Predicted"]

        # Correct token (always the first key)
        correct_probs.append(parsed[keys[0]])

        # False token (second key, if it exists)
        if len(keys) > 1:
            false_probs.append(parsed[keys[1]])
        else:
            false_probs.append(None)  # Append None if no false token

        # All predicted probabilities
        predicted_probs.extend(parsed["Predicted"].values())

    # Filter out None values from false_probs for the DataFrame
    valid_false_probs = [p for p in false_probs if p is not None]

    # Calculate lengths
    n_correct = len(correct_probs)
    n_false = len(valid_false_probs)
    n_predicted = len(predicted_probs)

    # Create arrays for the DataFrame
    probabilities = correct_probs + valid_false_probs + predicted_probs
    categories = (
        ['Correct'] * n_correct +
        ['False'] * n_false +
        ['Predicted'] * n_predicted
    )
    # Sentence indices should match the number of entries per category
    sentences = (
        list(range(n_correct)) +  # Correct
        [i for i, p in enumerate(false_probs) if p is not None] +  # False, only valid indices
        [i for i, row in enumerate(dataset["result_logit_probability"])
         for _ in json.loads(row)["Predicted"]]  # Predicted, repeat sentence index per pred token
    )

    # Verify lengths match
    assert len(probabilities) == len(categories) == len(sentences), \
        f"Lengths mismatch: {len(probabilities)}, {len(categories)}, {len(sentences)}"

    # Create DataFrame
    plot_data = pd.DataFrame({
        'Probability': probabilities,
        'Category': categories,
        'Sentence': sentences
    })

    # 1. Original Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Category', y='Probability', data=plot_data)
    plt.title('Probability Distributions: Correct vs False vs Predicted (Box Plot)')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    means = plot_data.groupby('Category')['Probability'].mean()
    for i, mean in enumerate(means):
        plt.axhline(y=mean, color='r', linestyle='--', alpha=0.5, xmin=i / 3, xmax=(i + 1) / 3)
    plt.show()

# Function to process probabilities and plot all subfolders in one plot
def plot_logit_probs_all(data_with_folders, results_path):
    # Dataframe to store all probabilities
    all_plot_data = []

    # Process each DataFrame
    for entry in data_with_folders:
        subfolder = list(entry.keys())[0]  # Get the Path object (e.g., Path("L1_H5"))
        df_for_subfolder = entry[subfolder]  # Get the DataFrame
        # Extract only the layer and head part or use "No Knockout" for main folder
        subfolder_name = str(subfolder)
        if "main" in subfolder_name:
            subfolder_name = "No Knockout"
        else:
            # Extract just the L#_H# part from paths like "L17_H9/meta-llama"
            match = re.match(r"(L\d+_H\d+)", subfolder_name)
            if match:
                subfolder_name = match.group(1)

        # Process data for each example in this subfolder's dataframe
        for index, data_row in df_for_subfolder.iterrows():
            parsed = json.loads(data_row["result_logit_probability"])
            # Get all keys except "Predicted"
            keys = [k for k in parsed.keys() if k != "Predicted"]

            # Correct token (first key)
            all_plot_data.append({
                "Subfolder": subfolder_name,
                "Category": "Correct Token",
                "Probability": parsed[keys[0]]
            })

            # False token (second key, if it exists)
            if len(keys) > 1:
                all_plot_data.append({
                    "Subfolder": subfolder_name,
                    "Category": "Incorrect Token",
                    "Probability": parsed[keys[1]]
                })
            else:
                all_plot_data.append({
                    "Subfolder": subfolder_name,
                    "Category": "Incorrect Token",
                    "Probability": 0  # Default to 0 if no false token
                })

    # Convert to DataFrame
    plot_df = pd.DataFrame(all_plot_data)

    # Configure matplotlib parameters
    size = 10
    plt.rc("font", size=size)
    plt.rc("axes", titlesize=size+3)
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.rc("legend", fontsize=size)
    
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define custom colors for each category
    category_colors = {
        'Correct Token': '#1f77b4',    # Blue
        'Incorrect Token': '#ff7f0e'   # Orange
    }
    
    # Create barplot with custom colors
    ax = sns.barplot(
        x="Subfolder",
        y="Probability",
        hue="Category",
        data=plot_df,
        palette=category_colors,
        errorbar=None
    )

    # Overlay individual data points with stripplot
    sns.stripplot(
        x="Subfolder", 
        y="Probability", 
        hue="Category",
        data=plot_df,
        dodge=True, 
        alpha=0.5, 
        zorder=1,
        ax=ax,
        palette=["k", "k"],
        legend=False
    )

    # Customize the plot
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Next Token Probability', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for consistency
    
    # Get the handles and labels from the plot to ensure correct color matching
    handles, _ = ax.get_legend_handles_labels()
    
    # Only keep the first two handles (from barplot) to avoid duplicates from stripplot
    handles = handles[:2]
    
    plt.legend(
        handles=handles,
        labels=["Correct Token", "Incorrect Token"],
        title="Category",
        loc="upper right",
        frameon=False,
        fontsize=12
    )
    
    # Remove the x-axis label by setting it to an empty string
    ax.set_xlabel('')
    
    plt.tight_layout()

    # Save the plot in the parent directory
    output_file = Path(results_path) / "ablation_induction_heads_experiment_3.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved plot: {output_file}")

    # Show the plot
    plt.show()

def plot_ablation_results(data_with_folders, results_path):
    # This function is a copy of plot_logit_probs_all to satisfy the request of a new plotting function.
    # Dataframe to store all probabilities
    all_plot_data = []

    # Process each DataFrame
    for entry in data_with_folders:
        subfolder = list(entry.keys())[0]  # Get the Path object (e.g., Path("L1_H5"))
        df_for_subfolder = entry[subfolder]  # Get the DataFrame
        # Extract only the layer and head part or use "No Knockout" for main folder
        subfolder_name = str(subfolder)
        if "main" in subfolder_name:
            subfolder_name = "No Knockout"
        elif "all_heads_knocked_out" in subfolder_name:
            subfolder_name = "All Heads Knocked Out"
        else:
            # Extract just the L#_H# part from paths like "L17_H9/meta-llama"
            match = re.match(r"(L\d+_H\d+)", subfolder_name)
            if match:
                subfolder_name = match.group(1)

        # Process data for each example in this subfolder's dataframe
        for index, data_row in df_for_subfolder.iterrows():
            parsed = json.loads(data_row["result_logit_probability"])
            # Get all keys except "Predicted"
            keys = [k for k in parsed.keys() if k != "Predicted"]

            # Correct token (first key)
            all_plot_data.append({
                "Subfolder": subfolder_name,
                "Category": "Correct Token",
                "Probability": parsed[keys[0]]
            })

            # False token (second key, if it exists)
            if len(keys) > 1:
                all_plot_data.append({
                    "Subfolder": subfolder_name,
                    "Category": "Incorrect Token",
                    "Probability": parsed[keys[1]]
                })
            else:
                all_plot_data.append({
                    "Subfolder": subfolder_name,
                    "Category": "Incorrect Token",
                    "Probability": 0  # Default to 0 if no false token
                })

    # Convert to DataFrame
    plot_df = pd.DataFrame(all_plot_data)

    # Configure matplotlib parameters
    size = 10
    plt.rc("font", size=size)
    plt.rc("axes", titlesize=size+3)
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.rc("legend", fontsize=size)
    
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define custom colors for each category
    category_colors = {
        'Correct Token': '#1f77b4',    # Blue
        'Incorrect Token': '#ff7f0e'   # Orange
    }
    
    # Create barplot with custom colors
    ax = sns.barplot(
        x="Subfolder",
        y="Probability",
        hue="Category",
        data=plot_df,
        palette=category_colors,
        errorbar=None
    )

    # Overlay individual data points with stripplot
    sns.stripplot(
        x="Subfolder", 
        y="Probability", 
        hue="Category",
        data=plot_df,
        dodge=True, 
        alpha=0.5, 
        zorder=1,
        ax=ax,
        palette=["k", "k"],
        legend=False
    )

    # Customize the plot
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Next Token Probability', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for consistency
    
    # Get the handles and labels from the plot to ensure correct color matching
    handles, _ = ax.get_legend_handles_labels()
    
    # Only keep the first two handles (from barplot) to avoid duplicates from stripplot
    handles = handles[:2]
    
    plt.legend(
        handles=handles,
        labels=["Correct Token", "Incorrect Token"],
        title="Category",
        loc="upper right",
        frameon=False,
        fontsize=12
    )
    
    # Remove the x-axis label by setting it to an empty string
    ax.set_xlabel('')
    
    plt.tight_layout()

    # Save the plot in the parent directory
    output_file = Path(results_path) / "ablation_induction_heads_experiment_3_with_all_knockout.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved plot: {output_file}")

    # Show the plot
    plt.show()

def knock_out_attention_head(layer_idx, head_idx):

    # note that this function modifies the model in place and does not return anything
    # the changes you make are not recoverable, so you may want to reload the model to undo them

    layer = model.model.layers[int(layer_idx)] # get the layer of interest
    attn = layer.self_attn # get the head of interest

    # q projection dimension is [hidden_size, hidden_size]
    hidden_size = attn.q_proj.weight.shape[0]
    num_heads = attn.config.num_attention_heads # how many heads does the layer have?
    head_dim = hidden_size // num_heads # how large is a given head?

    # Compute the slice corresponding to the head.
    start = int(head_idx) * head_dim
    end = (int(head_idx) + 1) * head_dim

    attn.o_proj.weight[start:end, :] = 0

def knock_out_all_attention_heads(heads: dict):
    # note that this function modifies the model in place and does not return anything
    # the changes you make are not recoverable, so you may want to reload the model to undo them
    for l_h in heads.keys():
        layer, head = extract_integers(l_h)
        knock_out_attention_head(layer_idx=layer, head_idx=head)

def extract_integers(input_string):
    # Split the string at '_'
    parts = input_string.split('_')

    # Extract integer after 'L' from first part
    l_value = int(parts[0].replace('L', ''))

    # Extract integer after 'H' from second part
    h_value = int(parts[1].replace('H', ''))

    return l_value, h_value

def run_experiment(dataset_csv_file_path: str, model_name: str, results_path: str, knockout_heads: bool = False, layer: int = None, head: int = None, all_heads_knockout_exp: bool = False):
    print(
        f"Using device: {torch.device('mps') if torch.backends.mps.is_available() else 'cpu'}"
    )
    if knockout_heads:
        knock_out_attention_head(layer_idx=layer, head_idx=head)

    load_dataset(path_to_csv=dataset_csv_file_path)

    global dataset
    dataset[
        [
            "induction_scores",
            "attention_probability_first_sentence_token",
            "attention_probability_second_sentence_token",
            "logit_probabilities",
        ]
    ] = dataset.apply(
        lambda row: calculate_induction_scores(
            first_sentence=row["true_sentence"],
            second_sentence=row["false_sentence"],
        ),
        axis=1,
    )

    print("Done calculating induction scores for first variant batch.")

    # Calculate probabilities now for the switched variant. First false sentence and then true sentence
    dataset[
        [
            "induction_scores_switched",
            "attention_probability_first_sentence_token_switched",
            "attention_probability_second_sentence_token_switched",
            "logit_probabilities_switched",
        ]
    ] = dataset.apply(
        lambda row: calculate_induction_scores(
            first_sentence=row["false_sentence"],
            second_sentence=row["true_sentence"],
        ),
        axis=1,
    )

    print("Done calculating induction scores for second variant batch.")

    # Identify rows that will be dropped
    rows_to_drop = dataset[
        dataset[
            [
                "attention_probability_first_sentence_token",
                "attention_probability_second_sentence_token",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    # Print the rows that will be dropped
    print(f"Found {len(rows_to_drop)} rows with mismatched token counts that will be dropped.")
    # print("Rows being dropped:\n", rows_to_drop)

    dataset.dropna(
        subset=[
            "attention_probability_first_sentence_token",
            "attention_probability_second_sentence_token",
        ],
        inplace=True,
    )
    print(f"Dataset size after dropping rows: {len(dataset)}")

    # From the computed induction scores for the normal variant and switched variant, we compute the average induction
    # score for each head, sort the heads by desc induction score and pick the top 5.
    induction_scores = dataset["induction_scores"]
    induction_scores_switched = dataset["induction_scores_switched"]
    top_heads = get_average_across_heads(
        induction_scores=induction_scores,
        induction_scores_switched=induction_scores_switched,
        top_k_heads=5,
    )
    # print(f"The top heads with the avg induction score for the first variant are: {top_heads}\n")

    # Calculate the attention probabilities of the token from the first sentence and token from the second sentence from the top heads over the
    # example and the switched variant. We average the attention probabilities.
    dataset[
        [
            "attention_probability_first_sentence_token_top_induction_heads",
            "attention_probability_second_sentence_token_top_induction_heads",
        ]
    ] = dataset.apply(lambda row: attention_probs_for_heads(row, top_heads), axis=1)

    # Calculate the logit probability of the first sentence token, first sentence token and the most probable token, average them as well.
    dataset["result_logit_probability"] = dataset.apply(
        lambda row: calculate_logit_probability(
            row["logit_probabilities"], row["logit_probabilities_switched"]
        ),
        axis=1,
    )

    print("Done calculating the results. Saving results...")

    # Create CSV result files saved in folders respective to the used LLM.
    if all_heads_knockout_exp:
        results_path = results_path + "/" + "all_heads_knocked_out" + "/"
    elif layer or head is not None:
        results_path = results_path + "/" + f"L{layer}_H{head}" + "/"
    else:
        results_path = results_path + "/" + "main" + "/"

    model_image_path = save_result_csv(model_name=model_name, results_path=results_path)

    # # Plot the results
    # plot_attention_probabilities_tokens_heads_results(save_path=model_image_path)
    #
    # # Plot the results of the logit probabilities
    # plot_logit_probs()



    # return the top found heads
    return top_heads

def aggregate_results(results_path):
    results_path = Path(results_path)
    data_with_folders = []
    desired_columns = ["result_logit_probability"]

    for csv_file in results_path.rglob("*.csv"):
        if csv_file.is_file():
            sub_folder = csv_file.parent.relative_to(results_path)
            df = pd.read_csv(csv_file)
            filtered_df = df[desired_columns]
            data_with_folders.append({sub_folder: filtered_df})

    return data_with_folders

def main():
    # ### Experiment Start
    print("Your current working directory:", os.getcwd())
    results_path = f"results-{models[0]}"

    # Calculate running the experiment without knocking out any attention heads -> return top found heads
    # Results are in columns "result_true_token, result_false_token"
    initialize_model(model_name=models[0])
    top_heads = run_experiment(
        dataset_csv_file_path=CSV_PATH_DATASET,
        model_name=models[0],
        results_path=results_path,
    )

    # print(f"The top heads for the experiment withouth knocking out heads are: {top_heads}.")

    for l_h in top_heads.keys():
        # get the layer and head from the string
        layer, head = extract_integers(l_h)

        # Run the experiment, knocking out head by head and save results.
        run_experiment(dataset_csv_file_path=CSV_PATH_DATASET, model_name=models[0], results_path=results_path, knockout_heads=True, layer=layer, head=head)

        # print(f"The top heads for the experiment after knocking out head {head} at layer {layer} are {knockout_top_heads}.")
        delete_model()
        initialize_model(model_name=models[0]) # reinit the model

    # Experiment with all heads knocked out
    print("Running experiment with all top heads knocked out.")
    knock_out_all_attention_heads(top_heads)
    run_experiment(
        dataset_csv_file_path=CSV_PATH_DATASET,
        model_name=models[0],
        results_path=results_path,
        all_heads_knockout_exp=True
    )
    delete_model()
    initialize_model(model_name=models[0]) # reinit the model

    # aggregate the results and plot them
    data_with_folders = aggregate_results(results_path)
    plot_ablation_results(data_with_folders=data_with_folders, results_path=Path(results_path))




    # Delete the model loaded in memory
    delete_model()


if __name__ == "__main__":
    main()
