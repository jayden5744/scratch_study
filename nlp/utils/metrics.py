# n-gram 분석 https://blog.ilkyu.kr/entry/%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8%EB%A7%81-ngram
from typing import List, Tuple, Union


def word_ngram(sentence: str, num_gram: int) -> List[Tuple[str]]:
    """
    문장을 ngram 단위로 split 하는 함수
    Args:
        sentence(str): 분석할 문장
        num_gram(int): n-gram 단위
    Returns:
        List[str]: n-gram 단위로 split 된 sentence
    """
    # in the case a file is given, remove escape characters
    sentence = sentence.replace("\n", " ").replace("\r", " ")
    text = tuple(sentence.split(" "))
    ngrams = [
        text[x : x + num_gram] for x in range(0, len(text)) if x + num_gram <= len(text)
    ]
    return ngrams


def make_freq_list(ngrams: List[Tuple[str]]) -> List[List[Union[Tuple[str], int]]]:
    """
    n-gram 별로 빈도를 저장하는 리스트 생성
    Args:
        ngrams(str): n-gram 으로 분리된 sentence List
    Returns:
        List[List[Union[Tuple[str], int]]]: 각 n-gram 리스트별 빈도
    """
    unique_ngrams = list(set(ngrams))
    freq_list = [0 for _ in range(len(unique_ngrams))]
    for ngram in ngrams:
        idx = unique_ngrams.index(ngram)
        freq_list[idx] += 1
    result = [unique_ngrams, freq_list]
    return result


def precision(
    output: List[List[Union[Tuple[str], int]]],
    target: List[List[Union[Tuple[str], int]]],
) -> float:
    """
    output 문장과 target 문장 사이의 precision을 측정하는 함수
    Args:
        output(List[Union[List[Tuple[str], List[int]]]]): 각 ngram 별로 split한 output 문장 list와 빈도수 list
        target(List[Union[List[Tuple[str], List[int]]]]): 각 ngram 별로 split한 target 문장 list와 빈도수 list
    Returns:
        float: 두 문장의 precision
    """
    result = 0
    for i in range(len(output[0])):
        if output[0][i] in target[0]:
            idx = target[0].index(output[0][i])
            result += min(output[1][i], target[1][idx])
    try:
        return result / sum(output[1])
    except ZeroDivisionError:
        return 0


def calculate_bleu(predict_sentence: str, target_sentence: str) -> float:
    """
    두 문장의 Bleu score를 측정하는 함수
    Args:
        predict_sentence(str): 모델이 예측한 문장
        target_sentence(str): reference 문장
    Returns:
        float: bleu score
    """
    output = []
    target = []
    sentence = target_sentence.replace("\n", " ").replace("\r", " ").split(" ")
    if len(sentence) < 4:  # 문장 단어의 개수가 4개 미만일 때
        max_n = len(sentence) + 1
    else:
        max_n = 5

    for i in range(1, max_n):
        n_gram = word_ngram(predict_sentence, i)
        out_tmp = make_freq_list(n_gram)
        output.append(out_tmp)
        n_gram2 = word_ngram(target_sentence, i)
        tar_tmp = make_freq_list(n_gram2)
        target.append(tar_tmp)

    result = 0
    for i in range(len(output)):
        n_pre = precision(output[i], target[i])
        if i == 0:
            result = n_pre
        else:
            result *= n_pre
    result = pow(result, 1 / (max_n - 1))
    # Brevity Penalty
    bp = min(1, sum(output[0][1]) / sum(target[0][1]))
    print("bp:", bp)
    return bp * result
