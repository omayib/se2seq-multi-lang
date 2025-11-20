import os
from pathlib import Path
from datetime import datetime

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.config import MAX_LENGTH, DROPOUT_RATE, HIDDEN_SIZE, ATTENTION_CHECKPOINT, RESULT_DIR, OUTPUT_DIR, device, \
    NUM_LAYER
from src.data.data_collection import DataCollection
from src.data.utils import index_tensor_from_sentence
from src.inference.predictor import Predictor
from src.layers.decoder_attention import AttentionDecoderRNN
from src.layers.encoder_rnn import EncoderRNN
def plot_attention(attention_weights: np.ndarray, input_sentence: list[str], output_sentence: list[str]):
    # Ensure attention_weights is a NumPy array for seaborn
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Create a figure and an axes object for the plot
    plt.figure(figsize=(len(input_sentence) * 0.8, len(output_sentence) * 0.8)) # Adjust figsize dynamically

    # Generate a heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=input_sentence,
        yticklabels=output_sentence,
        cmap='viridis',
        cbar=True,
        linewidths=.5,
        linecolor='lightgrey'
    )

    # Label axes and add title
    plt.xlabel('Input Sentence')
    plt.ylabel('Output Sentence')
    plt.title('Word Alignment (Attention Weights)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Ensure tight layout and display the plot
    plt.tight_layout()

    prefix = "result-wma"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"{prefix}_{timestamp}.png"
    plt.savefig(f"./../output/result/{plot_filename}")
    plt.show()
if __name__ == "__main__":

    model_save_dir = Path(OUTPUT_DIR)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_path_file = os.path.join(model_save_dir, ATTENTION_CHECKPOINT)
    print(f"model_path_file : {model_path_file} ")

    plot_dir_candidate = f"{OUTPUT_DIR}{RESULT_DIR}"
    plot_dir = Path(plot_dir_candidate)
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"plot path : {plot_dir} is dir {plot_dir.is_dir()}")

    print("started!")
    data_collection = DataCollection()
    combined_train_pairs = data_collection.get_train_pairs()
    combined_valid_pairs = data_collection.get_validation_pairs()
    combined_input_language = data_collection.get_combined_input_language()
    combined_output_language = data_collection.get_combined_output_language()
    pairs_id_jv = data_collection.get_pairs_id_jv()
    pairs_id_su = data_collection.get_pairs_id_su()

    multi_lang_checkpoint = torch.load(model_path_file)

    # Re-initialize models and load state dicts for evaluation
    encoder_multi_lang_eval = EncoderRNN(input_size=combined_input_language.num_words, hidden_size=HIDDEN_SIZE,
                                         dropout_rate=DROPOUT_RATE,num_layers=NUM_LAYER).to(device)
    decoder_attention_multi_lang_eval = AttentionDecoderRNN(hidden_size=HIDDEN_SIZE,
                                                            output_size=combined_output_language.num_words,
                                                            dropout_rate=DROPOUT_RATE, device=device,
                                                            max_length=MAX_LENGTH,num_layers=NUM_LAYER).to(device)
    encoder_cp =multi_lang_checkpoint["encoder_state_dict"]
    dencoder_cp = multi_lang_checkpoint["decoder_state_dict"]
    print(f"cp num_epochs: {multi_lang_checkpoint['num_epochs']}")
    print(f"cp learning_rate: {multi_lang_checkpoint['learning_rate']}")
    print(f"cp batch_size: {multi_lang_checkpoint['batch_size']}")
    print(f"cp dropout_rate: {multi_lang_checkpoint['dropout_rate']}")
    print(f"cp training_rate: {multi_lang_checkpoint['training_rate']}")
    print(f"cp epoch_accuracies: {multi_lang_checkpoint['epoch_accuracies']}")
    print(f"cp epoch_rouge_f1s: {multi_lang_checkpoint['epoch_rouge_f1s']}")
    print(f"cp epoch_bleu_scores: {multi_lang_checkpoint['epoch_bleu_scores']}")
    print(f"cp epoch_losses: {multi_lang_checkpoint['epoch_losses']}")
    encoder_multi_lang_eval.load_state_dict(encoder_cp)
    decoder_attention_multi_lang_eval.load_state_dict(dencoder_cp)

    encoder_multi_lang_eval.eval()
    decoder_attention_multi_lang_eval.eval()

    # Create a Predictor instance for the multi-language model
    multi_lang_predictor = Predictor(encoder=encoder_multi_lang_eval, decoder=decoder_attention_multi_lang_eval,
                                     input_language=combined_input_language, output_language=combined_output_language)
    sentence_ori_id = pairs_id_su[4][0]
    sentence_target_jv = pairs_id_su[4][1]
    print(f"bhs id : {sentence_ori_id}")
    print(f"terjemahan : {sentence_target_jv}")

    # sentence_to_translate = "<to_javanese> jika ada pertanyaan"
    translated_words = multi_lang_predictor.translate(sentence_ori_id)
    sentence_transleted = ' '.join(translated_words)
    print(f"Original Indonesian sentence: '{sentence_ori_id}'")
    print(f"Translated words: {translated_words}")
    print(f"Translated sentence: {' '.join(translated_words)}")

    input_tensor = index_tensor_from_sentence(combined_input_language, sentence_ori_id, device=device)

    print(f"input_tensor  {input_tensor}")
    with torch.no_grad():
        encoder_outputs_id, encoder_hidden_id = encoder_multi_lang_eval(input_tensor)
        _, _, attention_weights_id= decoder_attention_multi_lang_eval(encoder_outputs_id,encoder_hidden_id)
    print(f"attention_weights_id {attention_weights_id}")
    input_words_id = sentence_ori_id.split(" ")
    output_words_id = sentence_transleted.split(" ")

    attention_to_plot_id_jv = attention_weights_id.squeeze(0)[:len(output_words_id), :len(output_words_id)]
    plot_attention(attention_to_plot_id_jv, input_words_id, output_words_id)
