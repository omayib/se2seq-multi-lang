import torch
from torch.utils.data import DataLoader
from torchmetrics.text import ROUGEScore, BLEUScore
from src.config import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from src.inference.predictor import Predictor
from src.data.language_data import LanguageData


class Evaluator:
    def __init__(self, valid_dataloader: DataLoader, predictor: Predictor, output_language: LanguageData):
        self.valid_dataloader = valid_dataloader
        self.predictor = predictor
        self.output_language = output_language  # Pengganti tokenizer
        self.device = predictor.device

        # Inisialisasi metrics sekali saja
        self.rouge_metric = ROUGEScore()
        self.bleu_metric = BLEUScore()

    def decode_ids_to_text(self, token_ids: torch.Tensor) -> str:
        """Mengubah Tensor ID menjadi kalimat string, mengabaikan special tokens."""
        words = []
        for tid in token_ids:
            idx = tid.item()
            # Skip token spesial agar tidak mengacaukan metrik
            if idx in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                continue
            # Ambil kata dari dictionary LanguageData
            word = self.output_language.index_to_word.get(idx, "")
            words.append(word)
        return " ".join(words)

    def evaluate(self) -> dict:
        """Menjalankan evaluasi penuh dalam satu loop efisien."""
        self.rouge_metric.reset()
        self.bleu_metric.reset()

        total_token_accuracy = 0
        total_samples = 0

        # Loop validasi
        for input_batch, target_batch in self.valid_dataloader:
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            # 1. Prediksi Batch
            predicted_ids_batch = self.predictor.predict_by_index(input_batch)

            decoded_preds = []
            decoded_targets = []

            # 2. Proses per kalimat dalam batch
            for i in range(len(target_batch)):
                pred_row = predicted_ids_batch[i]
                target_row = target_batch[i]

                # --- LOGIKA BARU: Token Accuracy ---

                # A. Ambil ID Asli (Hilangkan Padding)
                # Target
                tgt_valid_idx = target_row.nonzero().flatten()
                clean_target_ids = target_row[tgt_valid_idx]

                # Prediksi
                pred_valid_idx = pred_row.nonzero().flatten()
                clean_pred_ids = pred_row[pred_valid_idx]

                # B. Hitung berapa kata yang cocok (Posisi harus sama)
                # Kita bandingkan sampai panjang terpendek dari keduanya
                min_len = min(len(clean_pred_ids), len(clean_target_ids))

                if min_len > 0:
                    # Hitung jumlah token yang sama di posisi yang sama
                    matches = (clean_pred_ids[:min_len] == clean_target_ids[:min_len]).sum().item()
                    # Akurasi = jumlah cocok / panjang target
                    accuracy = matches / len(clean_target_ids)
                else:
                    accuracy = 0.0

                total_token_accuracy += accuracy

                # --- AKHIR LOGIKA BARU ---

                # C. Decode ke Teks (Untuk BLEU & ROUGE)
                pred_text = self.decode_ids_to_text(clean_pred_ids)
                target_text = self.decode_ids_to_text(clean_target_ids)

                decoded_preds.append(pred_text)
                decoded_targets.append([target_text])

            total_samples += len(target_batch)

            # 3. Update Metrics
            self.rouge_metric.update(decoded_preds, [t[0] for t in decoded_targets])
            self.bleu_metric.update(decoded_preds, decoded_targets)

        # 4. Hitung Hasil Akhir
        final_rouge = self.rouge_metric.compute()
        final_bleu = self.bleu_metric.compute()

        # Rata-rata token accuracy
        final_accuracy = total_token_accuracy / total_samples if total_samples > 0 else 0.0

        return {
            "accuracy": final_accuracy,
            "bleu": final_bleu.item(),
            "rouge1_f1": final_rouge['rouge1_fmeasure'].item()
        }