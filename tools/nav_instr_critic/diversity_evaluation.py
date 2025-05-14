import os
import sys
import json
import gzip
import argparse
import functools
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Callable, Dict
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import math
import tempfile
import json
import jsonlines
import gzip
import lzma as xz
from multiprocessing import Pool
import multiprocessing
import random
import pandas as pd
import glob

from fast_bleu import BLEU, SelfBLEU
from rouge_score import rouge_scorer

# import bert_score
from lexical_diversity import lex_div
import time
from nltk.translate.bleu_score import sentence_bleu


class TickTimer:
    def __init__(self, precision=3, enable=True):
        self.cur_time = time.time()
        self.precision = precision
        self.enable = enable

    def __call__(self, info=""):
        if self.enable:
            print(
                "Time cost [{}] {}".format(
                    info, round(time.time() - self.cur_time, self.precision)
                )
            )
        self.cur_time = time.time()

    def reset(self):
        self.cur_time = time.time()


def get_pos(data: List[str]):
    """Turns a sequence into parts of speech.
    Args:
        data (List[str]): Data to tranform into part of speech tags.
    Returns:
        List[str]: Part-of-speech tags only
    """

    pos_tuples = [nltk.pos_tag(x.split()) for x in data]

    joined_pos = []

    for doc in pos_tuples:
        # joined_text.append(' '.join([x[0] for x in doc]))
        joined_pos.append(" ".join([x[1] for x in doc]))

    return joined_pos


def calculate_self_bleu(predictions):
    """Gets self-BLEU4 score.
        From https://arxiv.org/abs/1904.03971, https://dl.acm.org/doi/abs/10.1145/3209978.3210080
    Args:
        preditions: A list of strings, or a list of list of words
    Returns:
        score: The average self-BLEU score
    """
    assert len(predictions)
    assert isinstance(predictions[0], str) or (
        isinstance(predictions[0], list) and isinstance(predictions[0][0], str)
    )
    if isinstance(predictions[0], str):
        predictions = [word_tokenize(v) for v in predictions]
    scorer = SelfBLEU(predictions)
    score = np.mean(scorer.get_score()[4])
    return score


def _calc_homogenization_score(
    other, hyp, N, measure="bleu", smoothing_func=1, use_stemmer=False
):
    """Only used for parallelism"""
    res = 0
    if measure == "bleu":
        scorer = BLEU([hyp], smoothing_func=smoothing_func)
        res = np.mean(scorer.get_score(other)[4])
    elif measure == "rougel":
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
        res = np.mean([scorer.score(t, hyp)["rougeL"].fmeasure for t in other])
    return res


def calculate_homogenization_score_parallel(
    predictions, measure="bleu", smoothing_func=1, use_stemmer=False
):
    """Calculates the homogenization score for a set of documents (corpus-level).
        From https://arxiv.org/pdf/2309.05196.pdf
        Note: the implementation uses a smoothing function, which adds 0.1 to zero n-gram precision
    Args:
        preditions: A list of strings
        measure: Either 'rougel' or 'bleu'
        smoothing_func: BLEU smoothing, 0 or 1
        use_stemmer: ROUGE-L use stemmer, False or True
    Returns:
        score: The average paired BLEU score
    """
    score = 0
    pool_res = []
    pool = Pool(64)
    if measure == "bleu":
        predictions = [word_tokenize(v) for v in predictions]
        N = len(predictions)
        for i in range(N):
            hyp = predictions[i]
            other = predictions[0:i] + predictions[i + 1 :]
            pool_res.append(
                pool.apply_async(
                    _calc_homogenization_score,
                    args=(other, hyp, N, measure, smoothing_func, use_stemmer),
                )
            )
    elif measure == "rougel":
        N = len(predictions)
        for i in range(N):
            hyp = predictions[i]
            other = predictions[0:i] + predictions[i + 1 :]
            pool_res.append(
                pool.apply_async(
                    _calc_homogenization_score,
                    args=(other, hyp, N, measure, smoothing_func, use_stemmer),
                )
            )
    for i in tqdm(pool_res):
        res = i.get()
        score += res
    pool.close()
    pool.join()
    score = score / N
    return score


def calculate_homogenization_score(
    predictions, measure="bleu", smoothing_func=1, use_stemmer=False
):
    """Calculates the homogenization score for a set of documents (corpus-level).
        From https://arxiv.org/pdf/2309.05196.pdf
        Note: the implementation uses a smoothing function, which adds 0.1 to zero n-gram precision
    Args:
        preditions: A list of strings
        measure: Either 'rougel', 'bertscore', or 'bleu'
        smoothing_func: BLEU smoothing, 0 or 1
        use_stemmer: ROUGE-L use stemmer, False or True
    Returns:
        score: The average paired BLEU score
    """
    score = 0
    if measure == "bleu":
        predictions = [word_tokenize(v) for v in predictions]
        N = len(predictions)
        score = 0.0
        for i in tqdm(range(N)):
            scorer = BLEU([predictions[i]], smoothing_func=smoothing_func)
            score += np.mean(
                scorer.get_score(predictions[0:i] + predictions[i + 1 :])[4]
            )
        score = score / N
    elif measure == "rougel":
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
        N = len(predictions)
        score = 0.0
        for i in tqdm(range(N)):
            score += np.mean(
                [
                    scorer.score(t, predictions[i])["rougeL"].fmeasure
                    for t in predictions[0:i] + predictions[i + 1 :]
                ]
            )
        score = score / N
    # elif measure=="bertscore":
    #     N = len(predictions)
    #     N2 = N**2-N
    #     ref = [None] * N2
    #     hyp = [None] * N2
    #     for i in range(N):
    #         for j, v in enumerate(list(range(i))+list(range(i+1, N))):
    #             ref[i*(N-1)+j] = [predictions[v]]
    #             hyp[i*(N-1)+j] = predictions[i]
    #     # (precision, recall, f1score)
    #     score = bert_score.score(hyp, ref, model_type="distilbert-base-uncased", use_fast_tokenizer=True)[2].cpu().mean().item()
    return score


def _calc_self_repetition(ngram, predictions):
    res = np.zeros(len(predictions))
    for j in range(len(predictions)):
        res[j] = 1 if ngram in predictions[j] else 0
    return res


def calculate_self_repetition_parallel(predictions, n=4):
    """Calculates the self-repetition score.
        From https://aclanthology.org/2022.aacl-short.42.pdf
    Args:
        predictions: List of strings
        n: The order of N-gram
    Returns:
        score: The average of single-document self-repetition score
    """
    score = 0.0
    predictions = [" ".join(word_tokenize(v)) for v in predictions]
    ngrams = [
        [" ".join(ngram) for ngram in nltk.ngrams(v.split(" "), n=n)]
        for v in predictions
    ]
    ngrams_set = list(set([j for i in ngrams for j in i]))
    ngrams_map = {v: k for k, v in enumerate(ngrams_set)}
    N = len(predictions)
    ngrams_occ = np.zeros((len(ngrams_set), N))
    pool_res = []
    pool = Pool(64)
    for i, ngram in enumerate(ngrams_set):
        pool_res.append(
            pool.apply_async(_calc_self_repetition, args=(ngram, predictions))
        )
    for i, process in tqdm(enumerate(pool_res), total=len(pool_res)):
        ngrams_occ[i] = process.get()
    pool.close()
    pool.join()
    score = 0
    for i in tqdm(range(N)):
        ngram_indices = np.array([ngrams_map[v] for v in ngrams[i]])
        if len(ngram_indices):  # in case of the sentence is too short
            cnt = ngrams_occ[ngram_indices, :].sum() - len(ngrams[i])
            score += math.log(cnt + 1)
    score = score / N
    return score


def calculate_self_repetition(predictions, n=4):
    """Calculates the self-repetition score.
        From https://aclanthology.org/2022.aacl-short.42.pdf
    Args:
        predictions: List of strings
        n: The order of N-gram
    Returns:
        score: The average of single-document self-repetition score
    """
    score = 0.0
    predictions = [" ".join(word_tokenize(v)) for v in predictions]
    ngrams = [
        [" ".join(ngram) for ngram in nltk.ngrams(v.split(" "), n=n)]
        for v in predictions
    ]
    ngrams_set = list(set([j for i in ngrams for j in i]))
    ngrams_map = {v: k for k, v in enumerate(ngrams_set)}
    N = len(predictions)
    ngrams_occ = np.zeros((len(ngrams_set), N))
    for i, ngram in tqdm(enumerate(ngrams_set), total=len(ngrams_set)):
        for j in range(len(predictions)):
            ngrams_occ[i, j] = 1 if ngram in predictions[j] else 0
    score = 0
    for i in tqdm(range(N)):
        ngram_indices = np.array([ngrams_map[v] for v in ngrams[i]])
        if len(ngram_indices):  # in case of the sentence is too short
            cnt = ngrams_occ[ngram_indices, :].sum() - len(ngrams[i])
            score += math.log(cnt + 1)
    score = score / N
    return score


def calculate_ngram_diversity(
    predictions,
    num_n=4,
):
    """Calculates corpus-level ngram diversity based on unique ngrams
        (e.g., https://arxiv.org/pdf/2202.00666.pdf, https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00536/114593/Locally-Typical-Sampling).

    Args:
        predictions (List[str]): List of strings.
        num_n (int): Max ngrams to test up to. Defaults to 4.

    Returns:
        float: ngram diveristy score.
    """
    score = 0
    data = " ".join(predictions).split(" ")  # format to list of words

    for i in range(1, num_n + 1):
        ngrams = list(nltk.ngrams(data, i))
        # num unique ngrams / all ngrams for each size n
        score += len(set(ngrams)) / len(ngrams)

    return score


def calculate_compression_ratio(
    data: List[str],
    algorithm: str = "gzip",
    verbose: bool = False,
    path: Optional[str] = None,
    pos: Optional[bool] = False,
):
    """Calculates the compression ratio for a collection of text.
        From https://arxiv.org/abs/2403.00553
    Args:
        data (List[str]): Strings to compress.
        algorithm (str, optional): Either 'gzip' or 'xz'. Defaults to 'gzip'.
        verbose (bool, optional): Print out the original and compressed size separately. Defaults to False.
        path (str, optional): Path to store temporarily zipped files.
        pos (bool, optional): Whether to use POS tagging
    Returns:
        float: Compression ratio (original size / compressed size)
    """

    temp_dir = None
    if not path:
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name)
    else:
        path = Path(path)
    if pos:
        data = get_pos(data)

    with (path / "original.txt").open("w+") as f:
        f.write(" ".join(data))

    original_size = os.path.getsize(os.path.join(path, "original.txt"))

    if algorithm == "gzip":
        with gzip.GzipFile(str(path / "compressed.gz"), "w+") as f:
            f.write(gzip.compress(" ".join(data).encode("utf-8")))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))

    elif algorithm == "xz":
        with xz.open(str(path / "compressed.gz"), "wb") as f:
            f.write(" ".join(data).encode("utf-8"))

        compressed_size = (path / "compressed.gz").stat().st_size

    if verbose:
        print(f"Original Size: {original_size}\nCompressed Size: {compressed_size}")

    if temp_dir:
        temp_dir.cleanup()

    return original_size / compressed_size


def calculate_mattr(predictions):
    """Calculates the Moving Average Token-Type Ratio (MATTR).
        From https://www.tandfonline.com/doi/full/10.1080/09296171003643098

    Args:
        predictions: A list of strings
    Returns:
        score: The MATTR score
    """
    text = " ".join(predictions)
    tokens = lex_div.flemmatize(text)
    score = lex_div.mattr(tokens)
    return score


def calculate_hdd(predictions):
    """Calculates the Hypergeometric Distribution D (HD-D).
        From https://link.springer.com/article/10.3758/BRM.42.2.381

    Args:
        predictions: A list of strings
    Returns:
        score: The HD-D score
    """
    text = " ".join(predictions)
    tokens = lex_div.flemmatize(text)
    score = lex_div.hdd(tokens)
    return score


def calculate_all_metrics(
    get_instructions: Callable,
    dataset_name: str = "none",
    sample_ratio: float = 1.0,
    ngram: int = 4,
    time_info: bool = True,
    use_parallel: bool = False,
):
    """Calculates all diversity metrics
    Args:
        get_instructions: A callable function that returns a list of strings
        dataset_name: The dataset name used to generate result dict and file
        sample_ratio: A float number between [0,1] that selects a portion of samples to calculate
            Or a integer number that selects specific samples
        ngram: The N-gram order used in several metrics
        time_info: A boolean value indicating whether to show the time cost
        use_parallel: whether to use parallel self repetition and homogenization functions
    Returns:
        results: A single-element dict whose key is the dataset_name and the value is a dict containing all metrics
            {
                "dataset_name": {
                    "self_bleu": 1.0,
                    ...
                }
            }
            The results will be also stored in a json file results_{dataset_name}.json
    """
    random.seed(42)
    instructions = get_instructions()
    N = len(instructions)
    if sample_ratio < 1.0:
        sample_ratio = max(0.0, sample_ratio)
        N = max(1, int(N * sample_ratio))  # At least one sample is used
        instructions = random.sample(instructions, N)
    elif sample_ratio > 1.0:
        N = int(sample_ratio)
        instructions = random.sample(instructions, N)
    print("Calculate diversity metrics for {} with {} samples".format(dataset_name, N))

    lens = [len(word_tokenize(v)) for v in instructions]
    results = {dataset_name: {"length": np.mean(lens)}}
    tic = TickTimer(enable=time_info)
    results[dataset_name]["mattr"] = calculate_mattr(instructions)
    tic("mattr")
    results[dataset_name]["hdd"] = calculate_hdd(instructions)
    tic("hdd")
    results[dataset_name]["cr"] = calculate_compression_ratio(instructions)
    tic("cr")
    results[dataset_name]["cr_pos"] = calculate_compression_ratio(
        instructions, pos=True
    )
    tic("cr_pos")
    results[dataset_name]["ngd"] = calculate_ngram_diversity(instructions)
    tic("ngd")
    if use_parallel:
        results[dataset_name]["self_rep"] = calculate_self_repetition_parallel(
            instructions, n=ngram
        )
    else:
        results[dataset_name]["self_rep"] = calculate_self_repetition(
            instructions, n=ngram
        )
    tic("self_rep")
    results[dataset_name]["self_bleu"] = calculate_self_bleu(instructions)
    tic("self_bleu")
    # if use_parallel: # Too slow
    #     results[dataset_name]["hom_rougel"] = calculate_homogenization_score_parallel(
    #         instructions, measure="rougel"
    #     )
    # else:
    #     results[dataset_name]["hom_rougel"] = calculate_homogenization_score(
    #         instructions, measure="rougel"
    #     )
    # tic("hom_rougel")
    results[dataset_name]["hom_bleu"] = calculate_homogenization_score(
        instructions, measure="bleu"
    )
    tic("hom_bleu")

    # with open("results_{}.json".format(dataset_name), "w") as f:
    #     f.write(json.dumps(results))

    return results


def get_vlnce():
    dataset_folder = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed")
    split = "train"
    with gzip.open(dataset_folder / split / (split + ".json.gz"), "r") as f:
        data = json.loads(f.read())
    episodes = data["episodes"]
    instructions = [v["instruction"]["instruction_text"] for v in episodes]
    return instructions


def get_vlnce_diverse():
    dataset_folder = Path("data/datasets/R2R_VLNCE_v1-3_diverse")
    split = "train"
    with gzip.open(dataset_folder / split / (split + ".json.gz"), "r") as f:
        data = json.loads(f.read())
    episodes = data["episodes"]
    instructions = [v["instruction"]["instruction_text"] for v in episodes]
    return instructions


def get_envdrop():
    dataset_folder = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed")
    split = "envdrop"
    with gzip.open(dataset_folder / split / (split + ".json.gz"), "r") as f:
        data = json.loads(f.read())
    episodes = data["episodes"]
    instructions = [v["instruction"]["instruction_text"] for v in episodes]
    return instructions


def get_envdrop10819():
    dataset_folder = Path("data/datasets/R2R_VLNCE_v1-3_preprocessed")
    split = "envdrop10819"
    with gzip.open(dataset_folder / split / (split + ".json.gz"), "r") as f:
        data = json.loads(f.read())
    episodes = data["episodes"]
    instructions = [v["instruction"]["instruction_text"] for v in episodes]
    return instructions


def get_envdrop10819_diverse():
    dataset_folder = Path("data/datasets/R2R_VLNCE_v1-3_diverse")
    split = "envdrop10819"
    with gzip.open(dataset_folder / split / (split + ".json.gz"), "r") as f:
        data = json.loads(f.read())
    episodes = data["episodes"]
    instructions = [v["instruction"]["instruction_text"] for v in episodes]
    return instructions


def get_r2r():
    dataset_folder = Path("nav_data/datasets/R2R")
    split = "R2R_train"
    with open(dataset_folder / (split + ".json"), "r") as f:
        data = json.loads(f.read())
    instructions = []
    for v in data:
        instructions.extend(v["instructions"])
    return instructions


def get_r2r_diverse():
    dataset_folder = Path("nav_data/datasets/R2R")
    split = "R2R_train_diverse"
    with open(dataset_folder / (split + ".json"), "r") as f:
        data = json.loads(f.read())
    instructions = []
    for v in data:
        instructions.extend(v["instructions"])
    return instructions


def get_rxr():
    dataset_folder = Path("nav_data/datasets/RxR_marky")
    split = "rxr_train_guide"
    with gzip.open(dataset_folder / (split + ".jsonl.gz"), "r") as f:
        data = [json.loads(line) for line in f]
    instructions = []
    for v in data:
        if "en-" in v["language"]:
            instructions.append(v["instruction"])
    return instructions


def get_marky():
    dataset_folder = Path("nav_data/datasets/RxR_marky")
    split = "rxr_marky_train_guide"
    with gzip.open(dataset_folder / (split + ".jsonl.gz"), "r") as f:
        data = [json.loads(line) for line in f]
    instructions = []
    for v in data:
        if "en" in v["language"]:
            instructions.append(v["instruction"])
    return instructions


def get_youtube():
    dataset_folder = Path("nav_data/datasets/YoutubeVLN/data/")
    split = "task/aug+R2R_train"
    with open(dataset_folder / (split + ".json"), "r") as f:
        data = json.loads(f.read())
    instructions = []
    for v in data:
        instructions.extend(v["instructions"])
    return instructions


def get_modular_gpt():
    dataset_folder = Path("data/vlnce_traj_action_clean/train")
    instructions = []
    for inst_path in dataset_folder.glob("*/inst_modular_gpt/0.txt"):
        with open(inst_path, "r") as f:
            instructions.append(f.read())
    return instructions


def get_modular_llama():
    dataset_folder = Path("data/vlnce_traj_action_clean/train")
    instructions = []
    for inst_path in dataset_folder.glob("*/inst_modular_llama/0.txt"):
        with open(inst_path, "r") as f:
            instructions.append(f.read())
    return instructions


def get_generation(folder, splits, inst_alias):
    folder = Path(folder)
    instructions = []
    for split in splits:
        cur_folder = Path(folder) / split
        for inst_file in cur_folder.glob("*/{}/*.txt".format(inst_alias)):
            with open(inst_file, "r") as f:
                instructions.append(f.read())
    # print(len(instructions))
    return instructions


def merge_to_df():
    """Merges all result json files and converts it to a data frame
    Args: None
    Returns:
        A pandas.DataFrame
    """
    results = {}
    for file in glob.glob("results_*.json"):
        with open(file, "r") as f:
            results.update(json.load(f))
    df = pd.DataFrame(results).transpose()
    idx = [
        "length",
        "mattr",
        "hdd",
        "ngd",
        "cr",
        "cr_pos",
        "self_rep",
        "self_bleu",
        "hom_bleu",
        "hom_rougel",
    ]
    df = df[idx]
    df.columns = [
        "length",
        "mattr ↑",
        "hdd ↑",
        "ngd ↑",
        "cr ↓",
        "cr_pos ↓",
        "self_rep ↓",
        "self_bleu ↓",
        "hom_bleu ↓",
        "hom_rougel ↓",
    ]
    return df.sort_index(axis=0)


def run_evaluation_diversity(folder, splits, inst_alias):
    # https://arxiv.org/abs/2403.00553
    return calculate_all_metrics(
        functools.partial(get_generation, folder, splits, inst_alias),
        dataset_name="generation",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=Path,
        default="data/vlnce_traj_action_clean",
        required=True,
        help="Data folder",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default="val_seen",
        required=True,
        help="Evaluated splits",
    )
    parser.add_argument(
        "--inst_alias",
        type=str,
        default="inst_vo_gpt_gpt",
        required=True,
        help="The test instruction folder name",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default="data/speaker_evaluation",
        required=False,
        help="Whether to use bfloat16",
    )
    args = parser.parse_args()
    eval_splits_cat = "_".join(args.splits)
    res = run_evaluation_diversity(args.folder, args.splits, args.inst_alias)
    with open(
        Path(args.output_folder)
        / "{}_diversity_{}.json".format(args.inst_alias, eval_splits_cat),
        "w",
    ) as f:
        json.dump(res, f, indent=2)
    print("Diversity {}: ".format(args.inst_alias), res)
