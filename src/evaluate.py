import os
from pathlib import Path

import torch

from src.config import MAX_LENGTH, DROPOUT_RATE, HIDDEN_SIZE, ATTENTION_CHECKPOINT, RESULT_DIR, OUTPUT_DIR, device, \
    NUM_LAYER
from src.data.data_collection import DataCollection
from src.inference.predictor import Predictor
from src.layers.decoder_attention import AttentionDecoderRNN
from src.layers.encoder_rnn import EncoderRNN

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
    multi_lang_checkpoint = torch.load(model_path_file)

    # Re-initialize models and load state dicts for evaluation
    encoder_multi_lang_eval = EncoderRNN(input_size=combined_input_language.num_words, hidden_size=HIDDEN_SIZE,
                                         dropout_rate=DROPOUT_RATE,num_layers=NUM_LAYER).to(device)
    decoder_attention_multi_lang_eval = AttentionDecoderRNN(hidden_size=HIDDEN_SIZE,
                                                            output_size=combined_output_language.num_words,
                                                            dropout_rate=DROPOUT_RATE, device=device,
                                                            max_length=MAX_LENGTH,num_layers=NUM_LAYER).to(device)

    encoder_multi_lang_eval.load_state_dict(multi_lang_checkpoint["encoder_state_dict"])
    decoder_attention_multi_lang_eval.load_state_dict(multi_lang_checkpoint["decoder_state_dict"])

    encoder_multi_lang_eval.eval()
    decoder_attention_multi_lang_eval.eval()

    # Create a Predictor instance for the multi-language model
    multi_lang_predictor = Predictor(encoder=encoder_multi_lang_eval, decoder=decoder_attention_multi_lang_eval,
                                     input_language=combined_input_language, output_language=combined_output_language)

    sentence_to_translate = "<to_javanese> jika ada pertanyaan lebih lanjut yang ingin kamu ketahui atau mengalami kendala terkait dengan produk"
    translated_words = multi_lang_predictor.translate(sentence_to_translate)
    print(f"Original Indonesian sentence: '{sentence_to_translate}'")
    print(f"Translated Javanese words: {translated_words}")
    print(f"Translated Javanese sentence: {' '.join(translated_words)}")