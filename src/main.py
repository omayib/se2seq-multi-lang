import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt

from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, HIDDEN_SIZE, device, DROPOUT_RATE, MAX_LENGTH, \
    TRAINING_RATE, ATTENTION_CHECKPOINT, OUTPUT_DIR, RESULT_DIR
from src.data.data_collection import DataCollection
from src.data.pair_data_loader import PairDataLoader
from src.inference.seq2seq import Seq2SeqTrainer
from src.layers.decoder_attention import AttentionDecoderRNN
from src.layers.encoder_rnn import EncoderRNN
from pathlib import Path
import seaborn as sns


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

    combined_train_dataloader = PairDataLoader(pairs=combined_train_pairs, input_language=combined_input_language,
                                               output_language=combined_output_language, batch_size=BATCH_SIZE).load
    combined_valid_dataloader = PairDataLoader(pairs=combined_valid_pairs, input_language=combined_input_language,
                                               output_language=combined_output_language,
                                               batch_size=BATCH_SIZE).validation_dataloader

    print("Combined training and validation DataLoaders created successfully.")

    encoder_multi_lang = EncoderRNN(input_size=combined_input_language.num_words, hidden_size=HIDDEN_SIZE,
                                    dropout_rate=DROPOUT_RATE).to(device)
    decoder_attention_multi_lang = AttentionDecoderRNN(hidden_size=HIDDEN_SIZE,
                                                       output_size=combined_output_language.num_words,
                                                       dropout_rate=DROPOUT_RATE, device=device,
                                                       max_length=MAX_LENGTH).to(device)

    trainer_multi_lang = Seq2SeqTrainer(
        train_dataloader=combined_train_dataloader,
        encoder=encoder_multi_lang,
        decoder=decoder_attention_multi_lang,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        input_language=combined_input_language,
        output_language=combined_output_language,
        valid_dataloader=combined_valid_dataloader,
        debug_log_tokens_every_n_epochs=10,
        debug_num_logged_tokens=5
    )

    multi_lang_losses, multi_lang_accuracies, multi_lang_rouge_f1s, attention_bleu_scores_multi = trainer_multi_lang.train()

    torch.save({
        "encoder_state_dict": encoder_multi_lang.state_dict(),
        "decoder_state_dict": decoder_attention_multi_lang.state_dict(),
        "num_epochs": trainer_multi_lang.num_epochs,
        "learning_rate": trainer_multi_lang.learning_rate,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "hidden_size": HIDDEN_SIZE,
        "training_rate": TRAINING_RATE,
        "epoch_accuracies": multi_lang_accuracies,
        "epoch_rouge_f1s": multi_lang_rouge_f1s,
        "epoch_bleu_scores": attention_bleu_scores_multi,
        "epoch_losses": multi_lang_losses
    }, model_path_file)

    print("Multi-language Attention model training complete and checkpoint saved.")
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # Convert tensor values to floats if necessary for plotting
    attention_losses_multi_float = [l.item() if isinstance(l, torch.Tensor) else l for l in multi_lang_losses]
    attention_accuracies_multi_float = [a.item() if isinstance(a, torch.Tensor) else a for a in multi_lang_accuracies]
    attention_rouge_f1s_multi_float = [r.item() if isinstance(r, torch.Tensor) else r for r in multi_lang_rouge_f1s]

    sns.lineplot(x=range(1, NUM_EPOCHS + 1), y=attention_losses_multi_float, ax=axes[0])
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss: Multilang Attention Model')

    sns.lineplot(x=range(1, NUM_EPOCHS + 1), y=attention_accuracies_multi_float, ax=axes[1])
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy: Multilang Attention Model')

    sns.lineplot(x=range(1, NUM_EPOCHS + 1), y=attention_rouge_f1s_multi_float, ax=axes[2])
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('ROUGE-1 F1')
    axes[2].set_title('ROUGE-1 F1: Multilang Attention Model')

    # Final adjustments
    plt.tight_layout()

    # --- The crucial saving step ---
    prefix = "result"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"{prefix}_{timestamp}.png"
    plt.savefig(f"./../output/result/{plot_filename}")

    # Show the plot (optional, but good for interactive sessions)
    plt.show()