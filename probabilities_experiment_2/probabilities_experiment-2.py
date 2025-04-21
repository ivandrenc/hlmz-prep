import gc
import json
import os
import re
from collections import defaultdict
from operator import itemgetter

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
mod = None
tokenizer = None
META_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
GOOGLE_GEMMA_2_2B = "google/gemma-2-2b"
dataset = {}
CSV_PATH_DATASET = "dataset/examples.csv"
num_examples = 1


models = [META_LLAMA_3_2_3B]

# Experiment description: We run the model with the examples in no order and with the order switched, find the top 5
# induction heads by looking into the induction scores of both ran examples, average the scores across examples and
# select the top 5. Then with the identified heads we get the attention probabilities of the true, false token from every example
# and we average them as well. Then we can plot these results for the top 5 heads as a barplot to see which heads are more
# biased to output false information or correct information. Then we also compute the probabilities of the next token prediction
# by looking into the logits, for the correct token, false token, and the top probable to predict token. We then plot these results as well.


def initialize_model(model_name: str, tokenizer_name: str = None):
    if not tokenizer_name:
        tokenizer_name = model_name
    # Initialize model and tokenizer
    global mod
    mod = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
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
        print("File does not exist.")
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
    global mod
    out = mod(
        tokens.unsqueeze(0).to(mod.device), return_dict=True, output_attentions=True
    )
    # Return the output of the model, the tokenized prompt, number of tokens from the sentences (both sentences should have the same amount of tokens at this point)
    return out, tokens, true_sentence_token_n


def plot_induction_mask_with_plotly(induction_mask, induction_mask_text, prompt):
    # specify how many examples to plot
    global num_examples
    if num_examples <= 0:
        return
    # Create a Heatmap with the numeric mask (z) and attach the text
    heatmap = go.Heatmap(
        z=induction_mask,
        text=induction_mask_text,
        hoverinfo="text",  # Only show the text on hover
        colorscale="Blues",  # Choose any Plotly colorscale you like
        showscale=True,
    )

    fig = go.Figure(data=[heatmap])

    # Truncate prompt if too long and wrap the title text
    truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
    
    # Make the squares actually square by linking x/y scales
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(autorange="reversed"),  # Reverse y-axis so row 0 is at top
        title={
            'text': f"Induction Mask for prompt: {truncated_prompt}",
            'font': {'size': 12},
            'xanchor': 'center',
            'yanchor': 'top',
            'x': 0.5
        },
        # Enable title wrapping
        autosize=True,
        margin=dict(t=80)  # Increase top margin to accommodate wrapped title
    )
    
    # Save the plot as a PDF
    fig.write_image("induction_mask_plot_1.pdf")
    
    # Display the plot
    fig.show()
    num_examples -= 1


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
        global num_examples
        if num_examples > 0:
            print("Induction Mask:\n")
            print(induction_mask)
            print()
            print("Induction Mask plot:\n")
            
            # Configure matplotlib parameters for consistent styling
            size = 10  # Base font size
            plt.rc("font", size=size)
            plt.rc("axes", titlesize=size+3)  # Increased title size
            plt.rc("xtick", labelsize=size)
            plt.rc("ytick", labelsize=size)
            plt.rc("legend", fontsize=size)
            
            import matplotlib as mpl
            mpl.rcParams["figure.dpi"] = 300
            mpl.rcParams["savefig.dpi"] = 300
            
            # Create a figure and plot the induction mask
            plt.figure(figsize=(10, 8))
            plt.imshow(induction_mask)
            
            # Decode tokens for axis labels
            token_labels = [tokenizer.decode(token) for token in token_sequence]
            
            # Set x and y axis ticks with the decoded tokens
            plt.xticks(range(sequence_length), token_labels, rotation=90)
            plt.yticks(range(sequence_length), token_labels)
            
            # Add axis labels with larger font size
            plt.ylabel("Source", fontsize=size+5)
            plt.xlabel("Destination", fontsize=size+5)
            
            # Save the plot as PDF
            plt.savefig("induction_mask_plot.pdf", format="pdf", bbox_inches="tight")
            
            # Display the plot
            plt.show()
            
            num_examples -= 1
    return induction_mask


def compute_induction_head_scores(
    token_sequence: torch.Tensor, induction_mask: torch.Tensor, model_output
):
    num_heads = mod.config.num_attention_heads
    num_layers = mod.config.num_hidden_layers
    sequence_length = token_sequence.shape[0]

    tril = torch.tril_indices(
        sequence_length, sequence_length
    )  # gets the indices of elements on and below the diagonal
    induction_flat = induction_mask[tril[0], tril[1]].flatten()

    induction_scores_heads = {}

    for layer in range(num_layers):
        for head in range(num_heads):
            pattern = model_output["attentions"][layer][0][head].cpu().to(float)
            pattern_flat = pattern[tril[0], tril[1]].flatten()
            score = (induction_flat @ pattern_flat) / pattern_flat.sum()
            induction_scores_heads[f"L{layer}_H{head}"] = score.item()

    return induction_scores_heads


def create_heatmap(induction_scores: dict):
    print_colored_separator()
    print("Heatmap of induction scores across heads and layers: \n")
    
    # Extract layer and head information from the dictionary keys
    layers_heads = [(int(k.split('_')[0][1:]), int(k.split('_')[1][1:])) for k in induction_scores.keys()]
    num_layers = max([l for l, _ in layers_heads]) + 1
    num_heads = max([h for _, h in layers_heads]) + 1
    
    # Create a 2D array to hold the scores
    scores_matrix = np.zeros((num_layers, num_heads))
    
    # Fill the matrix with scores from the dictionary
    for key, score in induction_scores.items():
        layer = int(key.split('_')[0][1:])
        head = int(key.split('_')[1][1:])
        scores_matrix[layer, head] = score
    
    # Create the heatmap
    plt.figure(figsize=(10, 8), dpi=300)
    _, ax = plt.subplots()
    sns.heatmap(scores_matrix, cbar_kws={"label": "Induction-Head Matching Score"}, ax=ax)
    ax.set_ylabel("Layer #")
    ax.set_xlabel("Head #")
    plt.tight_layout()
    plt.savefig("induction_scores_heatmap.pdf", format="pdf", bbox_inches="tight", dpi=300)
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
    true_df["Type"] = "Correct Token"
    false_df["Type"] = "Incorrect Token"

    # Concatenate both DataFrames
    long_df = pd.concat([true_df, false_df])

    # Convert to long format for seaborn
    long_df = long_df.melt(
        id_vars=["Type"], var_name="L_H_Key", value_name="Probability"
    )

    # Configure matplotlib parameters for consistent styling
    size = 10  # Base font size
    plt.rc("font", size=size)
    plt.rc("axes", titlesize=size+3)  # Increased title size
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.rc("legend", fontsize=size)
    
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=long_df, x="L_H_Key", y="Probability", hue="Type", errorbar=None)
    
    # Overlay individual data points with stripplot
    sns.stripplot(
        x="L_H_Key", 
        y="Probability", 
        hue="Type",
        data=long_df,
        dodge=True, 
        alpha=0.5, 
        zorder=1,
        ax=ax,
        palette=["k", "k"],
        legend=False
    )
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', labelsize=14)
    
    # Remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)
    
    # Set labels with larger font sizes
    plt.ylabel('Attention Probability', fontsize=16)
    plt.xlabel("Layer, Head", fontsize=15)
    
    # Set y-axis limits for probabilities
    plt.ylim(0, 1)
    
    # Get the handles and labels from the plot to ensure correct color matching
    handles, _ = ax.get_legend_handles_labels()
    
    # Only keep the first two handles (from barplot) to avoid duplicates from stripplot
    handles = handles[:2]
    
    plt.legend(
        handles=handles,
        labels=["Correct Token", "Incorrect Token"],
        loc="upper right",
        frameon=False,
        fontsize=14
    )
    
    plt.tight_layout()
    
    # Save as PDF for better quality
    ax.get_figure().savefig(f"{save_path}-results-plot.pdf", format="pdf", bbox_inches="tight", dpi=300)

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
    global mod
    del mod
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
        token_sequence=token_sequence, token_number_sentence=token_number_sentence, show_induction_mask=True
    )

    # compute the induction heads
    induction_scores_heads = compute_induction_head_scores(
        token_sequence=token_sequence,
        induction_mask=induction_mask,
        model_output=models_output,
    )

    # Create heatmap of induction scores
    create_heatmap(induction_scores=induction_scores_heads)

    # extract the token probability for the true and false sentence
    true_token_probability, false_token_probability = token_probability_extraction(
        heads=induction_scores_heads,
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
            induction_scores_heads,
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

    # Create DataFrames for each category
    correct_df = pd.DataFrame({
        'Probability': correct_probs,
        'Category': 'Correct Token'
    })
    
    false_df = pd.DataFrame({
        'Probability': valid_false_probs,
        'Category': 'Incorrect Token'
    })
    
    predicted_df = pd.DataFrame({
        'Probability': predicted_probs,
        'Category': 'Predicted Token'
    })
    
    # Concatenate all DataFrames
    plot_data = pd.concat([correct_df, false_df, predicted_df])

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
        'Correct Token': '#2ca02c',    # Green
        'Incorrect Token': '#d62728',      # Red
        'Predicted Token': '#1f77b4'   # Blue
    }
    
    # Create boxplot with individual points and custom colors
    ax = sns.boxplot(data=plot_data, x="Category", y="Probability", 
                    width=0.5, showfliers=True, 
                    palette=category_colors)
    
    # Add individual points with jitter - all points are black/grey with alpha
    sns.stripplot(data=plot_data, x="Category", y="Probability",
                 color='black', alpha=0.3, size=4, jitter=0.2,
                 edgecolor=None)  # Remove edge color to ensure all points are filled
    
    # Customize the plot
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Next Token Probability', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("logit_probabilities_boxplot.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

    



def run_experiment(dataset_csv_file_path: str, llm_models: list, results_path: str):
    for model in llm_models:
        print(
            f"Using device: {torch.device('mps') if torch.backends.mps.is_available() else 'cpu'}"
        )
        initialize_model(model_name=model, tokenizer_name=model)
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
        print("Rows being dropped:\n", rows_to_drop)

        dataset.dropna(
            subset=[
                "attention_probability_first_sentence_token",
                "attention_probability_second_sentence_token",
            ],
            inplace=True,
        )
        # From the computed induction scores for the normal variant and switched variant, we compute the average induction
        # score for each head, sort the heads by desc induction score and pick the top 5.
        induction_scores = dataset["induction_scores"]
        induction_scores_switched = dataset["induction_scores_switched"]
        top_heads = get_average_across_heads(
            induction_scores=induction_scores,
            induction_scores_switched=induction_scores_switched,
            top_k_heads=5,
        )

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
        model_image_path = save_result_csv(model_name=model, results_path=results_path)

        # Plot the results
        plot_attention_probabilities_tokens_heads_results(save_path=model_image_path)

        # Plot the results of the logit probabilities
        plot_logit_probs()

        # Delete the model loaded in memory
        delete_model()


def main():
    # ### Experiment Start
    print("Your current working directory:", os.getcwd())
    run_experiment(
        dataset_csv_file_path=CSV_PATH_DATASET,
        llm_models=models,
        results_path="results_attn",
    )


if __name__ == "__main__":
    main()
