from pathlib import Path

import pandas as pd
from typing import Dict, Tuple

from src.config import DATA_RAW_DIR

RAW_DATA_DIR = Path(DATA_RAW_DIR)

DATA_FILENAMES={
    "train":"nusax_mt_train.csv",
    "valid":"nusax_mt_valid.csv",
    "test":"nusax_mt_test.csv"
}
DATA_URLS = {
    "train": "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/mt/train.csv",
    "valid": "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/mt/valid.csv",
    "test": "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/mt/test.csv",
}
def load_nusax_datasets()-> Dict[str, pd.DataFrame]:

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_dfs={}

    for split, filename in DATA_FILENAMES.items():
        local_path = RAW_DATA_DIR/filename

        if local_path.exists():
            raw_dfs[split] = pd.read_csv(local_path)
        else:
            url = DATA_URLS[split]
            print(f"Local file not found. Downloading {split} data from: {url}")
            try:
                df = pd.read_csv(url)
                df.to_csv(local_path,index=False)
                raw_dfs[split] = df
            except Exception as e:
                print(f"Error downloading or saving data dor {split}:{e}")
                raise

    df_train = raw_dfs["train"]
    df_valid = raw_dfs["valid"]
    df_test = raw_dfs["test"]

    selected_id_jv_columns = ["indonesian", "javanese"]
    df_train_jv = df_train[selected_id_jv_columns].copy()
    df_valid_jv = df_valid[selected_id_jv_columns].copy()
    df_test_jv = df_test[selected_id_jv_columns].copy()

    selected_id_su_columns = ["indonesian", "sundanese"]
    df_train_su = df_train[selected_id_su_columns].copy()
    df_valid_su = df_valid[selected_id_su_columns].copy()
    df_test_su = df_test[selected_id_su_columns].copy()

    print("Data loading and filtering complete.")

    return {
        "jv_train": df_train_jv,
        "jv_valid": df_valid_jv,
        "jv_test": df_test_jv,
        "su_train": df_train_su,
        "su_valid": df_valid_su,
        "su_test": df_test_su,
    }