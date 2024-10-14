from typing import List, Union

import pandas as pd
import stanza
from razdel import sentenize
from simalign import SentenceAligner

stanza.download("en")


def split_on_last_newline(src_list: List[str]) -> List[str]:
    """
    Разделяет строки в списке по последнему символу переноса '\n'
    """
    result = []

    for item in src_list:
        # Поиск индекса последнего переноса строки
        last_newline_index = item.rfind("\n")

        if last_newline_index != -1:
            result.append(item[:last_newline_index])
            result.append(item[last_newline_index + 1 :])
        else:
            result.append(item)

    return result


def split_text_into_sentences(text: str, lang: str) -> List[str]:
    """
    Разбивает текст на предложения.
    """
    if lang == "ru":
        sentences = [sent.text for sent in sentenize(text)]
        return split_on_last_newline(sentences)
    if lang == "en":
        nlp_trg = stanza.Pipeline(lang="en", processors="tokenize")
        trg_doc = nlp_trg(text)
        return [sent.text for sent in trg_doc.sentences]
    return text


def get_max_alignment_diff(
    src_sentence: Union[str, List[str]],
    trg_sentence: Union[str, List[str]],
    aligner: SentenceAligner,
    matching_method: str = "inter",
) -> int:
    """
    Возвращает максимальное значение расхождения в порядке слов в предложениях.
    По умолчанию, тип выравнивания - inter (Argmax).
    """
    alignments = aligner.get_word_aligns(src_sentence, trg_sentence)[matching_method]
    max_diff = max(abs(src_pos - trg_pos) for src_pos, trg_pos in alignments)
    return max_diff


def create_diff_table(
    src_text: List[str], trg_text: List[str], aligner: SentenceAligner
) -> pd.DataFrame:
    """
    Формирует для исходного и целевого текстов таблицу расхождений в порядке слов в предложениях.
    Столбцы таблицы: №, src_sentence, trg_sentence, diff.
    """
    flag_equal = len(src_text) == len(trg_text)
    if not flag_equal:
        print(
            f"Ошибка! В исходном тексте {len(src_text)} предложений, в целевом тексте {len(trg_text)} предложений!."
        )

    data = []
    for index, (src_sentence, trg_sentence) in enumerate(
        zip(src_text, trg_text), start=1
    ):
        diff = (
            get_max_alignment_diff(src_sentence, trg_sentence, aligner)
            if flag_equal
            else -1
        )
        data.append([index, src_sentence, trg_sentence, diff])

    df = pd.DataFrame(data, columns=["№", "src_sentence", "trg_sentence", "diff"])
    return df
