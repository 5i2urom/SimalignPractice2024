from docx import Document
from simalign import SentenceAligner

from config import INPUT_PATH, OUTPUT_PATH
from text_comparison_tools import create_diff_table, split_text_into_sentences

RU_DOC_FILE = f"{INPUT_PATH}CR-2019-1-1-RU.docx"
EN_DOC_FILE = f"{INPUT_PATH}CR-2019-1-1-EN.docx"
DIFF_TABLE = f"{OUTPUT_PATH}diff_table_1-1.xlsx"


def read_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        full_text = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(full_text)
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return ""


if __name__ == "__main__":
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    ru_text = read_docx(RU_DOC_FILE)
    en_text = read_docx(EN_DOC_FILE)

    ru_text_list = split_text_into_sentences(ru_text, "ru")
    en_text_list = split_text_into_sentences(en_text, "en")

    diff_table = create_diff_table(ru_text_list, en_text_list, aligner)
    diff_table.to_excel(DIFF_TABLE, index=False)
