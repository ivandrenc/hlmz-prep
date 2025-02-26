import re

import torch
import seaborn as sns
from circuitsvis.attention import attention_heads, attention_patterns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import gc
import json
from collections import defaultdict
from operator import itemgetter
import ast

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


models = [META_LLAMA_3_2_3B]


def initialize_model(model_name: str, tokenizer_name: str = None):
    if not tokenizer_name:
        tokenizer_name = model_name
    # Initialize model and tokenizer
    global model
    model = AutoModelForCausalLM.from_pretrained(
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
    dataset["token_probability_true_sentence"] = 0
    dataset["token_probability_false_sentence"] = 0


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
    prompt_repetitions: int = 1,
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


def save_plot_results(save_path: str):
    # Load the DataFrame (assuming df is already loaded)
    # Convert JSON strings to dictionaries
    dataset["true_probs"] = dataset["token_probability_true_sentence"].apply(json.loads)
    dataset["false_probs"] = dataset["token_probability_false_sentence"].apply(json.loads)

    # Convert probabilities to DataFrame
    true_df = pd.DataFrame(dataset["true_probs"].to_list())
    false_df = pd.DataFrame(dataset["false_probs"].to_list())

    # Add labels for True and False sentences
    true_df["Type"] = "True Sentence"
    false_df["Type"] = "False Sentence"

    # Concatenate both DataFrames
    long_df = pd.concat([true_df, false_df])

    # Convert to long format for seaborn
    long_df = long_df.melt(id_vars=["Type"], var_name="L_H_Key", value_name="Probability")

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=long_df, x="L_H_Key", y="Probability", hue="Type")
    ax.get_figure().savefig(f"{save_path}-results-plot.png", dpi=300)




def get_average_across_heads(head_set, top_k_heads: int):
    h_sum = defaultdict(float)
    h_count = defaultdict(int)

    for induction_scores in head_set:
        for l_h in induction_scores.items():
            l_h_key, value = l_h
            h_sum[l_h_key] = h_sum[l_h_key] + value
            h_count[l_h_key] = h_count[l_h_key] + 1

    h_average = {h: h_sum[h] / h_count[h] for h in h_sum}
    # Sort the values of the heads in descending order. Slice the top k heads.
    sorted_heads = dict(
        sorted(h_average.items(), key=itemgetter(1), reverse=True)[:top_k_heads]
    )
    return sorted_heads


def average_induction_scores(head_set, slice_top_k_heads: int):
    avg_induction_score_heads_sorted = get_average_across_heads(
        head_set=head_set, top_k_heads=slice_top_k_heads
    )
    return avg_induction_score_heads_sorted


def save_result_csv(model_name: str, dataset_csv_file_path: str):
    global dataset
    model_name_folder = model_name.split("/")
    folder_path = os.path.dirname(dataset_csv_file_path) + "/" + model_name_folder[0]
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
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


def calculate_induction_scores(true_sentence: str, false_sentence: str):
    models_output, token_sequence, token_number_sentence = feed_forward(
        true_sentence=true_sentence, false_sentence=false_sentence
    )
    if token_sequence is None:
        # true_sentence and false_sentence have different number of tokens after tokenizing them.
        return pd.Series([pd.NA, pd.NA])
    induction_mask = create_attention_mask(
        token_sequence=token_sequence, token_number_sentence=token_number_sentence
    )
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

    return pd.Series(
        [induction_scores, true_token_probability, false_token_probability]
    )


def attention_probs_for_heads(row, top_heads):
    probs_true = json.loads(row["token_probability_true_sentence"])
    probs_false = json.loads(row["token_probability_false_sentence"])
    data_true = {}
    data_false = {}
    for key, _ in top_heads.items():
        data_true[key] = probs_true[key]
        data_false[key] = probs_false[key]
    return pd.Series([json.dumps(data_true), json.dumps(data_false)])


def run_experiment(
    dataset_csv_file_path: str, llm_models: list, prompt_repetitions: int = 1
):
    for mod in llm_models:
        print(
            f"Using device: {torch.device('mps') if torch.backends.mps.is_available() else 'cpu'}"
        )
        initialize_model(model_name=mod, tokenizer_name=mod)
        load_dataset(path_to_csv=dataset_csv_file_path)

        global dataset
        dataset[
            [
                "induction_scores",
                "token_probability_true_sentence",
                "token_probability_false_sentence",
            ]
        ] = dataset.apply(
            lambda row: calculate_induction_scores(
                true_sentence=row["true_sentence"], false_sentence=row["false_sentence"]
            ),
            axis=1,
        )

        # Identify rows that will be dropped
        rows_to_drop = dataset[
            dataset[
                ["token_probability_true_sentence", "token_probability_false_sentence"]
            ]
            .isna()
            .any(axis=1)
        ]

        # Print the rows that will be dropped
        print("Rows being dropped:\n", rows_to_drop)

        dataset.dropna(
            subset=[
                "token_probability_true_sentence",
                "token_probability_false_sentence",
            ],
            inplace=True,
        )

        head_set = dataset["induction_scores"]
        top_heads = average_induction_scores(head_set=head_set, slice_top_k_heads=5)
        print(f"The top heads with the avg induction score are: {top_heads}\n")

        dataset[
            ["token_probability_true_sentence", "token_probability_false_sentence"]
        ] = dataset.apply(lambda row: attention_probs_for_heads(row, top_heads), axis=1)

        # Create CSV result files saved in folders respective to the used LLM.
        model_image_path = save_result_csv(
            model_name=mod, dataset_csv_file_path=dataset_csv_file_path
        )

        # Plot the results
        save_plot_results(save_path=model_image_path)

        # Delete the model loaded in memory
        delete_model()


def main():
    # ### Experiment Start
    print("Your current working directory:", os.getcwd())
    run_experiment(
        dataset_csv_file_path=CSV_PATH_DATASET, llm_models=models, prompt_repetitions=1
    )


if __name__ == "__main__":
    main()
