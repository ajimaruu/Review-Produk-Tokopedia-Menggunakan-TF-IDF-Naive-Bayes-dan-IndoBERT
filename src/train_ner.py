# src/train_ner.py

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


# ==========================
# 1. Load NER dataset
# ==========================

df = pd.read_csv(
    "data/ner_dataset.tsv",
    sep=r"\s+",
    names=["token", "label"]
)


print(df.head())


# ==========================
# 2. Buat label mapping
# ==========================

label_list = sorted(
    df["label"].unique()
)

label2id = {
    label:i
    for i, label in enumerate(label_list)
}

id2label = {
    i:label
    for label,i in label2id.items()
}


print(label2id)



# ==========================
# 3. Gabungkan token menjadi 1 kalimat
# ==========================

dataset = Dataset.from_dict({

    "tokens":[
        df["token"].tolist()
    ],

    "labels":[
        df["label"].tolist()
    ]

})


print(dataset[0])



# ==========================
# 4. Tokenizer IndoBERT
# ==========================

model_name = "indobenchmark/indobert-base-p1"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)



def tokenize(example):

    encoding = tokenizer(

        example["tokens"],

        is_split_into_words=True,

        padding="max_length",

        truncation=True,

        max_length=128

    )


    word_ids = encoding.word_ids()


    labels = []


    for word_id in word_ids:


        if word_id is None:

            labels.append(-100)


        else:

            labels.append(
                label2id[
                    example["labels"][word_id]
                ]
            )


    encoding["labels"] = labels


    return encoding



# ==========================
# 5. Tokenisasi dataset
# ==========================

tokenized_dataset = dataset.map(
    tokenize
)


print(tokenized_dataset[0])