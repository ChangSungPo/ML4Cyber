import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")
import os
import shutil

import matplotlib
import pandas as pd

import codecs  # this is used for file operations
import gc
import multiprocessing
import pickle
import pickle as pkl
import random as r
import re
from datetime import datetime as dt
from multiprocessing import Process  # this is used for multithreading

import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
from itertools import product
from pathlib import Path
from nltk.util import ngrams
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier

PREFIXES = [
    "HEADER:",
    ".text:",
    ".Pav:",
    ".idata:",
    ".data:",
    ".bss:",
    ".rdata:",
    ".edata:",
    ".rsrc:",
    ".tls:",
    ".reloc:",
    ".BSS:",
    ".CODE",
]
OPCODES = [
    "jmp",
    "mov",
    "retf",
    "push",
    "pop",
    "xor",
    "retn",
    "nop",
    "sub",
    "inc",
    "dec",
    "add",
    "imul",
    "xchg",
    "or",
    "shr",
    "cmp",
    "call",
    "shl",
    "ror",
    "rol",
    "jnb",
    "jz",
    "rtn",
    "lea",
    "movzx",
]
KEYWORDS = [".dll", "std::", ":dword"]
REGISTERS = ["edx", "esi", "eax", "ebx", "ecx", "edi", "ebp", "esp", "eip"]


def extract_asm_features(folder_name, output_filename, process_id):
    data_root_path = Path("..") / "data" / "raw_data" / folder_name
    # print(f"Working on : {data_root_path} -> output to : {output_filename}")
    files = os.listdir(data_root_path)

    with open(output_filename, "w+") as out_file:
        for f in tqdm(
            files,
            desc=f"Core {process_id} handling {folder_name:<6}",
            position=process_id,
            leave=True,
        ):
            pref_counts = np.zeros(len(PREFIXES), dtype=int)
            op_counts = np.zeros(len(OPCODES), dtype=int)
            kw_counts = np.zeros(len(KEYWORDS), dtype=int)
            reg_counts = np.zeros(len(REGISTERS), dtype=int)

            file_id = f.split(".")[0]
            out_file.write(file_id + ",")

            # read asm file by line
            file_path = os.path.join(data_root_path, f)
            with codecs.open(file_path, encoding="cp1252", errors="replace") as fli:
                for line in fli:
                    parts = line.rstrip().split()
                    if not parts:
                        continue

                    l_first = parts[0]
                    parts_rest = parts[1:]

                    # 1. Prefixes
                    for i, pref in enumerate(PREFIXES):
                        if pref in l_first:
                            pref_counts[i] += 1

                    # No content，skip
                    if not parts_rest:
                        continue

                    # 2. Opcodes
                    for i, op in enumerate(OPCODES):
                        if op in parts_rest:
                            op_counts[i] += 1

                    # 3. Registers (text or CODE)
                    if "text" in l_first or "CODE" in l_first:
                        for i, reg in enumerate(REGISTERS):
                            for part in parts_rest:
                                if reg in part:
                                    reg_counts[i] += 1

                    # 4. Keywords
                    for i, kw in enumerate(KEYWORDS):
                        for part in parts_rest:
                            if kw in part:
                                kw_counts[i] += 1

            all_counts = np.concatenate([pref_counts, op_counts, reg_counts, kw_counts])
            out_file.write(",".join(map(str, all_counts)) + "\n")

    print(f"Finish : {folder_name}")


# multi process for extracting asm features
def main():
    # the below code is used for multiprogramming
    # the number of process depends upon the number of cores present System
    # process is used to call multiprogramming
    os.makedirs("output", exist_ok=True)

    tasks = [
        ("first", "output/asm_fold_1.csv"),
        ("second", "output/asm_fold_2.csv"),
        ("third", "output/asm_fold_3.csv"),
        ("fourth", "output/asm_fold_4.csv"),
        ("fifth", "output/asm_fold_5.csv"),
    ]

    processes = []

    for i, (folder, out_file) in enumerate(tasks):
        p = multiprocessing.Process(
            target=extract_asm_features, args=(folder, out_file, i)
        )
        processes.append(p)
        p.start()

        # wait
    for p in processes:
        p.join()

    print("\n" * len(tasks))
    print("All Done!")


def merge_asm_features():
    asm_columns = ["ID"] + PREFIXES + OPCODES + REGISTERS + KEYWORDS

    output_path = Path("output")
    tasks = [
        output_path / "asm_fold_1.csv",
        output_path / "asm_fold_2.csv",
        output_path / "asm_fold_3.csv",
        output_path / "asm_fold_4.csv",
        output_path / "asm_fold_5.csv",
    ]

    print("Merging...")
    df_list = [pd.read_csv(f, names=asm_columns) for f in tasks]

    dfasm = pd.concat(df_list, ignore_index=True)
    print("Finish")

    data_label = Path("..") / "data" / "raw_data" / "trainLabels.csv"
    Y = pd.read_csv(data_label)

    Y.columns = ["ID", "Class"]

    result_asm = pd.merge(dfasm, Y, on="ID", how="left")

    print(result_asm.head())

    save_path = output_path / "final_asm_features_with_class.csv"
    result_asm.to_csv(save_path, index=False)
    print(f"\nSave to: {save_path}")

    return result_asm


# calculate asm file size
def calculate_file_size():
    base_path = Path("..") / "data" / "raw_data"
    data_label = base_path / "trainLabels.csv"

    Y = pd.read_csv(data_label)
    Y = Y.set_index("Id")

    subfolders = ["first", "second", "third", "fourth", "fifth"]

    fnames = []
    sizebytes = []
    class_bytes = []

    for folder in tqdm(subfolders, desc="Folder"):
        folder_path = base_path / folder

        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} not found.")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith(".asm")]

        for file in tqdm(files, desc=f"calculating {folder}", leave=False):
            if file.endswith(".asm"):
                # file size (MB)
                file_full_path = folder_path / file
                statinfo = os.stat(file_full_path)

                file_id = file.split(".")[0]

                if file_id in Y.index:
                    fnames.append(file_id)
                    sizebytes.append(statinfo.st_size / (1024.0 * 1024.0))
                    class_bytes.append(Y.loc[file_id, "Class"])

    asm_size_byte = pd.DataFrame(
        {"ID": fnames, "size": sizebytes, "Class": class_bytes}
    )

    print(asm_size_byte.head())
    asm_size_byte.to_csv("asm_file_sizes.csv", index=False, encoding="utf-8")
    return asm_size_byte


def calculate_sequence_of_opcodes():
    # Converting list to dictionary for faster runtime
    dict_asm_opcodes = dict(zip(OPCODES, [1 for i in range(len(OPCODES))]))

    raw_data_path = Path("..") / "data" / "raw_data"
    subfolders = ["first", "second", "third", "fourth", "fifth"]
    target_dir = Path("opcodes_asm_files")

    target_dir.mkdir(parents=True, exist_ok=True)

    for folder in tqdm(subfolders, desc="Totla Folder"):
        folder_path = raw_data_path / folder

        if not folder_path.exists():
            print(f"Warning, cannot find folder {folder_path}")
            continue

        asm_files = [f for f in os.listdir(folder_path) if f.endswith(".asm")]

        for this_asm_file in tqdm(asm_files, desc=f"Processing {folder}", leave=False):
            file_id = Path(this_asm_file).stem
            output_file = target_dir / f"{file_id}_opcode_asm_bi_grams.txt"

            opcode_sequence = []

            try:
                with open(
                    folder_path / this_asm_file,
                    "r",
                    encoding="cp1252",
                    errors="replace",
                ) as f:
                    for line in f:
                        words = line.strip().split()
                        for word in words:
                            if word in dict_asm_opcodes:
                                opcode_sequence.append(word)

                if opcode_sequence:
                    with open(output_file, "w") as out_f:
                        out_f.write(" ".join(opcode_sequence) + "\n")

            except Exception as e:
                print(f"Error:\n {this_asm_file} : {e}")


def calculate_bigram(bigram_tokens):
    sentence = ""
    vocabulary_list_for_byte_bigrams = []
    for i in tqdm(range(len(bigram_tokens))):
        for j in range(len(bigram_tokens)):
            bigram = bigram_tokens[i] + " " + bigram_tokens[j]
            sentence = sentence + bigram + ","
            vocabulary_list_for_byte_bigrams.append(bigram)
    return vocabulary_list_for_byte_bigrams


def calculate_opcodes_bigram(opcodes_asm__bigram_vocabulary):
    # 初始化 Vectorizer
    vectorizer_opcode = CountVectorizer(
        tokenizer=lambda x: x.split(),
        lowercase=False,
        ngram_range=(2, 2),
        vocabulary=opcodes_asm__bigram_vocabulary,
        token_pattern=None # 避免警告，因為已指定 tokenizer
    )

    file_list_opcode = os.listdir("opcodes_asm_files")
    
    # 取得特徵名稱
    feature_names = vectorizer_opcode.get_feature_names_out()
    header = ["ID"] + list(feature_names)

    output_path = "featurization/opcodes_asm_bigram_df.csv"
    
    # 使用不同的變數名稱 f_out 以避免衝突
    with open(output_path, mode="w", encoding="utf-8") as f_out:
        # 寫入標題
        f_out.write(",".join(map(str, header)) + "\n")

        for this_asm_filename in tqdm(file_list_opcode):
            this_file_id = this_asm_filename.split("_")[0]
            file_full_path = os.path.join("opcodes_asm_files", this_asm_filename)

            # 使用 with 開啟讀取檔案，確保自動關閉
            with open(file_full_path, "r", errors='ignore') as f_in:
                # 讀取並處理文字
                content = f_in.read().replace("\n", " ").lower()
                
            # transform 預期接收一個 list (corpus)
            sparse_matrix = vectorizer_opcode.transform([content])
            
            # 直接轉為 array，取第一列即可
            row_counts = sparse_matrix.toarray()[0]

            # 寫入資料列
            line = ",".join(map(str, [this_file_id] + list(row_counts)))
            f_out.write(line + "\n")

    # 最後讀取結果
    df_result = pd.read_csv(output_path)
    print(df_result.head())
    return df_result

def extract_top_500_asm_bigrams():
    opcodes_asm_bigram_df = pd.read_csv("featurization/opcodes_asm_bigram_df.csv")
    X_opcode_asm_bigram = opcodes_asm_bigram_df
    with open('featurization/class_labels.pkl', 'rb') as file:
        class_labels=pkl.load(file)
    y = class_labels
    # X_opcode_asm_bigram.head()

    #Get the best 500 features using SelectKBest. 


    kbest_object = SelectKBest(score_func=chi2, k=500)

    top_features=kbest_object.fit(X_opcode_asm_bigram.drop("ID", axis=1), y)

    # Save a dataframe with the feature scores along with the feature names.
    # And we will get the best fetures from this dataframe use to 
    top_features_scores=pd.DataFrame(top_features.scores_)

    # Now to get the original features names i.e. the names of all the columns we will need
    # `X_opcode_asm_bigram.columns`
    X_opcode_columns=pd.DataFrame(X_opcode_asm_bigram.columns)

    # Now concat all  original features names as a column with another column
    # which is "top_features_scores"
    top_asm_opcode_bigram_df=pd.concat([X_opcode_columns, top_features_scores],axis=1)

    # Give 2 Names for these 2 columns of data for this newly creaetd dataframe
    top_asm_opcode_bigram_df.columns=["ASM_Opcode_Bigram_Top_Feature_Name","ASM_Opcode_Bigram_Top_Feature_Score"]

    # Extract the largest 500 from this dataframw based on the values of "top_features_scores"
    top_asm_opcode_bigram_df=top_asm_opcode_bigram_df.nlargest(500,"ASM_Opcode_Bigram_Top_Feature_Score")

    print(top_asm_opcode_bigram_df.head())

if __name__ == "__main__":
    # main()
    # merge_asm_features()
    # calculate_file_size()
    # calculate_sequence_of_opcodes()
    # opcodes_asm__bigram_vocabulary = calculate_bigram(OPCODES)
    # calculate_opcodes_bigram(opcodes_asm__bigram_vocabulary)
    extract_top_500_asm_bigrams()
