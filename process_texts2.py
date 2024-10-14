import pandas as pd
from simalign import SentenceAligner

from config import INPUT_PATH, OUTPUT_PATH
from text_comparison_tools import create_diff_table

EXCEL_FILE = f"{INPUT_PATH}CR-2019-1-2-sentences.xlsx"
DIFF_TABLE = f"{OUTPUT_PATH}diff_table_1-2.xlsx"

if __name__ == "__main__":
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    df = pd.read_excel(EXCEL_FILE)
    src_sents = df["src_sentence"].values.tolist()
    trg_sents = df["trg_sentence"].values.tolist()

    diff_table = create_diff_table(src_sents, trg_sents, aligner)
    diff_table.to_excel(DIFF_TABLE, index=False)
