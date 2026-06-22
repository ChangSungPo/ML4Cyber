from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import pickle as pkl
from tqdm import tqdm
import csv
from joblib import Parallel, delayed
from collections import Counter
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_regression


def calculate_file_size():

    data_root_path = Path("..")
    data_label = data_root_path / "data" / "raw_data" / "trainLabels.csv"

    byte_files_path = data_root_path / "data" / "raw_data" / "byteFiles"
    file_data = []
    byte_files = list(byte_files_path.glob("*.bytes"))

    Y = pd.read_csv(data_label)
    total = len(Y)

    for file_path in tqdm(byte_files, desc="Calculate file sizes"):
        # file_path.stem -> file.split(".")[0]
        file_id = file_path.stem

        size_mb = file_path.stat().st_size / (1024.0 * 1024.0)

        file_data.append({"Id": file_id, "size": size_mb})

    size_df = pd.DataFrame(file_data)
    data_size_byte = pd.merge(size_df, Y, on="Id", how="inner")
    print(data_size_byte.head())

    # save_path = data_root_path / "data" / "byte_file_sizes.parquet"
    # data_size_byte.to_parquet(save_path, index=False)

    plt.figure(figsize=(10, 6))
    ax_distribute = sns.countplot(x="Class", data=Y, palette="pastel")
    plt.title("Class Distribution")

    for container in ax_distribute.containers:
        ax_distribute.bar_label(
            container, fmt=lambda x: f"{(x/total)*100:.1f}%", padding=3
        )

    # put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
    ax_distribute.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=total))

    plt.figure(figsize=(10, 6))
    ax_file_size_box = sns.boxplot(
        x="Class", y="size", data=data_size_byte, palette="pastel"
    )
    plt.title("Boxplot of .bytes File Sizes")
    plt.show()

    return data_size_byte


def process_single_file(bytes_file_path):
    file_id = bytes_file_path.stem
    txt_file_path = bytes_file_path.with_suffix(".txt")

    byte_counter = Counter()

    try:
        with bytes_file_path.open("r") as fin, txt_file_path.open("w") as fout:
            for line in fin:
                parts = line.strip().split()
                if not parts:
                    continue

                hex_codes = parts[1:]

                fout.write(" ".join(hex_codes) + "\n")

                byte_counter.update(code.upper() for code in hex_codes)

        bytes_file_path.unlink()

        return file_id, byte_counter

    except Exception as e:
        return file_id, f"Error: {e}"


def process_all_files_parallel(
    output="byte_files_feature.parquet", file_type="*.bytes", max_workers=None
):
    data_root_path = Path("..")
    byte_files_output_path = data_root_path / "data" / "raw_data" / output
    byte_files_path = data_root_path / "data" / "raw_data" / "byteFiles"

    input_path = Path(byte_files_path)
    bytes_files = list(input_path.glob(file_type))
    total_files = len(bytes_files)

    feature_keys = [f"{i:02X}" for i in range(256)] + ["??"]
    columns = ["Id"] + feature_keys

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if file_type == "*.bytes":
            futures = {
                executor.submit(process_single_file, file): file for file in bytes_files
            }
        else:
            futures = {
                executor.submit(process_txt_only, file): file for file in bytes_files
            }

        for future in tqdm(
            as_completed(futures), total=total_files, desc="Processing", unit="File"
        ):
            file_id, result = future.result()

            if isinstance(result, str) and result.startswith("Error"):
                tqdm.write(f"File {file_id} fail: {result}")
            else:
                row = [file_id] + [result.get(key, 0) for key in feature_keys]
                all_results.append(row)

    df = pd.DataFrame(all_results, columns=columns)
    # byte_files_output_path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_parquet(byte_files_output_path, index=False)

    print(f"\nAll Done!")
    print(f"Total Files: {len(bytes_files)}")

    return df


def calculate_bigram(bigram_tokens):
    combinations = list(product(bigram_tokens, bigram_tokens))

    vocabulary_list = [
        f"{i} {j}" for i, j in tqdm(combinations, desc="Generating vocabulary list")
    ]

    return vocabulary_list


def extract_all_byte_bigrams():
    base_path = Path("..") / "data" / "raw_data"
    input_folder = base_path / "byteFiles"
    output_path = "byte_files_bigram_df.csv"

    bigram_tokens = "00,01,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,10,11,12,13,14,15,16,17,18,19,1a,1b,1c,1d,1e,1f,20,21,22,23,24,25,26,27,28,29,2a,\
        2b,2c,2d,2e,2f,30,31,32,33,34,35,36,37,38,39,3a,3b,3c,3d,3e,3f,40,41,42,43,44,45,46,47,48,49,4a,4b,4c,4d,4e,4f,50,51,52,53,54,55,56,57,58,\
        59,5a,5b,5c,5d,5e,5f,60,61,62,63,64,65,66,67,68,69,6a,6b,6c,6d,6e,6f,70,71,72,73,74,75,76,77,78,79,7a,7b,7c,7d,7e,7f,80,81,82,83,84,85,86,\
        87,88,89,8a,8b,8c,8d,8e,8f,90,91,92,93,94,95,96,97,98,99,9a,9b,9c,9d,9e,9f,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aa,ab,ac,ad,ae,af,b0,b1,b2,b3,b4,b5,\
        b6,b7,b8,b9,ba,bb,bc,bd,be,bf,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,ca,cb,cc,cd,ce,cf,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db,dc,dd,de,df,e0,e1,e2,e3,e4,\
        e5,e6,e7,e8,e9,ea,eb,ec,ed,ee,ef,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,fa,fb,fc,fd,fe,ff,??"

    bigram_tokens = bigram_tokens.split(",")

    vocabulary_list = calculate_bigram(bigram_tokens)

    # 2. vectorizer
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(),
        lowercase=False,
        ngram_range=(2, 2),
        vocabulary=vocabulary_list,
    )

    features = ["ID"] + vectorizer.get_feature_names_out().tolist()

    if not input_folder.exists():
        print(f"Error, Folder not exist {input_folder}")
        return

    all_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features)

        for file_name in tqdm(all_files, desc="Extracting Byte Bigrams"):
            file_id = file_name.split(".")[0]
            file_full_path = input_folder / file_name

            try:
                with open(file_full_path, "r") as bf:
                    content = bf.read().replace("\n", " ").lower()

                bigram_counts = vectorizer.transform([content])

                row_data = bigram_counts.toarray()[0]
                writer.writerow([file_id] + row_data.tolist())

            except Exception as e:
                print(f"\nError: {file_name}: {e}")

    print(f"\n--- All Done ---")
    print(f"Result save to : {output_path}")
    
def read_file_content(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read().replace("\n", " ").lower()
    except:
        return ""

def extract_all_byte_bigrams_fast():
    bigram_tokens = "00,01,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,10,11,12,13,14,15,16,17,18,19,1a,1b,1c,1d,1e,1f,20,21,22,23,24,25,26,27,28,29,2a,\
        2b,2c,2d,2e,2f,30,31,32,33,34,35,36,37,38,39,3a,3b,3c,3d,3e,3f,40,41,42,43,44,45,46,47,48,49,4a,4b,4c,4d,4e,4f,50,51,52,53,54,55,56,57,58,\
        59,5a,5b,5c,5d,5e,5f,60,61,62,63,64,65,66,67,68,69,6a,6b,6c,6d,6e,6f,70,71,72,73,74,75,76,77,78,79,7a,7b,7c,7d,7e,7f,80,81,82,83,84,85,86,\
        87,88,89,8a,8b,8c,8d,8e,8f,90,91,92,93,94,95,96,97,98,99,9a,9b,9c,9d,9e,9f,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aa,ab,ac,ad,ae,af,b0,b1,b2,b3,b4,b5,\
        b6,b7,b8,b9,ba,bb,bc,bd,be,bf,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,ca,cb,cc,cd,ce,cf,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db,dc,dd,de,df,e0,e1,e2,e3,e4,\
        e5,e6,e7,e8,e9,ea,eb,ec,ed,ee,ef,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,fa,fb,fc,fd,fe,ff,??"

    bigram_tokens = bigram_tokens.split(",")

    vocabulary_list = calculate_bigram(bigram_tokens)
    
    base_path = Path("..") / "data" / "raw_data"
    input_folder = base_path / "byteFiles"
    output_path = "byte_files_bigram_df.csv"
    
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(),
        lowercase=False,
        ngram_range=(2, 2),
        vocabulary=vocabulary_list,
    )

    all_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    features = ["ID"] + vectorizer.get_feature_names_out().tolist()

    batch_size = 100
    
    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features)

        for i in tqdm(range(0, len(all_files), batch_size), desc="Bath Processing"):
            batch_files = all_files[i : i + batch_size]
            file_ids = [f.split(".")[0] for f in batch_files]
            file_paths = [input_folder / f for f in batch_files]

            contents = Parallel(n_jobs=-1)(delayed(read_file_content)(fp) for fp in file_paths)

            bigram_sparse = vectorizer.transform(contents)

            rows = bigram_sparse.toarray()
            for fid, row_data in zip(file_ids, rows):
                writer.writerow([fid] + row_data.tolist())

    print("--- All Done ---")


def process_txt_only(txt_file_path):
    file_id = txt_file_path.stem
    byte_counter = Counter()

    try:
        with txt_file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                hex_codes = line.strip().split()
                if not hex_codes:
                    continue
                byte_counter.update(code.upper() for code in hex_codes)

        return file_id, byte_counter, "SUCCESS"
    except Exception as e:
        return file_id, None, f"ERROR: {str(e)}"


def merge_byte_file_feature_size(df_feature: pd.DataFrame, df_size: pd.DataFrame):
    byte_features_with_size = df_feature.merge(df_size, on="Id")
    byte_features_with_size.to_parquet("byte_file_feature_size.parquet")
    print(byte_features_with_size.head(5))


def extract_top_2000_byte_bigrams():
    base_path = Path("..") / "data" / "raw_data" / "trainLabels.csv"
    Y = pd.read_csv(base_path)
    X_byte_bigram_all_df = pd.read_csv("byte_files_bigram_df.csv")
    select_kbest_object = SelectKBest(score_func=chi2, k=2000)
    if not os.path.isdir("featurization"):
        os.makedirs("featurization")

    if not os.path.isdir("featurization/featurization_final"):
        os.mkdir("featurization/featurization_final")

    # Creating and writing to a file named "class_labels.pkl" to get class class_labels and ID from byte unigrams dataframe and save it for later use

    class_labels = Y

    with open("featurization/class_labels.pkl", "wb") as file:
        pkl.dump(class_labels, file)

    """
    https://www.datacamp.com/community/tutorials/pickle-python-tutorial

    To open the file for writing, simply use the open() function. The first argument should be the name of your file. The second argument is 'wb'. The w means that you'll be writing to the file, and b refers to binary mode. This means that the data will be written in the form of byte objects.
    """

    # Load the class class_labels for training with random forest feature selector

    with open("featurization/class_labels.pkl", "rb") as file:
        class_labels = pkl.load(file)

    # SelectKBest scores the features using a function, which is chi2 here
    # Then "removes all but the k highest scoring features"

    # Need to remove "ID" column, else will get below error
    # "SelectKBest fit: ValueError: could not convert string to float"

    most_imp_features_byte_bigram = select_kbest_object.fit(
        X_byte_bigram_all_df.drop("ID", axis=1), class_labels
    )

    # most_imp_features_byte_bigram.scores_ => gives an array of form
    # array([9.79531407e+05, 4.26642398e+04, 1.78812060e+04, ..., 4.33426736e+07])
    # So now creating a df from this array
    most_imp_byte_bigram_feature_score_df = pd.DataFrame(
        most_imp_features_byte_bigram.scores_
    )

    # Creating a df from all the column names from the original full X_byte_bigram_all_df df
    most_imp_byte_bigram_columns_df = pd.DataFrame(X_byte_bigram_all_df.columns)

    # Concat the feature scores along with the feature names in a byte_bigram_df_important_feature_score,
    # From this we will get all feature names later, to be matched against X_byte_bigram_all_df - to extract ONLY the best features from the bigrams df data
    byte_bigram_df_important_feature_score = pd.concat(
        [most_imp_byte_bigram_columns_df, most_imp_byte_bigram_feature_score_df], axis=1
    )

    byte_bigram_df_important_feature_score.columns = [
        "Byte Bigram Top 2000 Feature Names",
        "Byte Bigram Top 2000 Feature Score",
    ]

    # Find the top 2000 features along with their scores

    # byte_bigram_df_important_feature_score=byte_bigram_df_important_feature_score.nlargest(1000, "Byte Bigram Top 2000 Feature Score")

    # Return the first 2000 rows with the largest values in the specified column ( "Byte Bigram Top 2000 Feature Score" )
    # in descending order. The columns that are not specified are returned as well, but not used for ordering.
    # Let's look at the top 10 features along with their scores + Save the feature score DF
    byte_bigram_df_important_feature_score = (
        byte_bigram_df_important_feature_score.nlargest(
            10, "Byte Bigram Top 2000 Feature Score"
        )
    )

    print(byte_bigram_df_important_feature_score.head(2))

    # Getting the list of first 2000 feature names
    top_2000_most_imp_byte_bigram_feature_names = list(
        byte_bigram_df_important_feature_score["Byte Bigram Top 2000 Feature Names"]
    )

    # top_2000_byte_bigram_features = dd.concat([X_byte_bigram_all_df["ID"], X_byte_bigram_all_df[top_2000_most_imp_byte_bigram]], axis=1)
    top_2000_byte_bigram_features = pd.concat(
        [
            X_byte_bigram_all_df["ID"],
            X_byte_bigram_all_df[top_2000_most_imp_byte_bigram_feature_names],
        ],
        axis=1,
    )

    top_2000_byte_bigram_features.to_csv(
        "featurization/featurization_final/top_2000_imp_byte_bigram_df.csv", index=False
    )

    print(top_2000_byte_bigram_features.shape)
    top_2000_byte_bigram_features.head(2)


if __name__ == "__main__":
    # df_byte_file_size = calculate_file_size()
    # df_byte_file_feature = process_all_files_parallel()
    # merge_byte_file_feature_size(df_byte_file_feature, df_byte_file_size)
    # extract_all_byte_bigrams()
    # TODO: Extract 2000
    extract_top_2000_byte_bigrams()
