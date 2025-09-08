# _*_ coding : utf-8 _*_
# @Author : 扶摇

import nltk

nltk.data.path.append(r"D:\Code\Deep-Learn\pythonProject\PLM\nltk_data")
import spacy

spacy.load("en_core_web_sm")  # 确保模型已安装

from tqdm import tqdm
import json
from collections import defaultdict, Counter
import numpy as np
from allennlp.predictors.predictor import Predictor


def gen_coref(predictor, doc_id, sample):
    sents = sample['sents']
    entities = sample['vertexSet']

    # Step 1: 构建 document 和各种索引映射
    document = ''
    word2char = []
    word2sent = []
    sent2word = []
    word_cnt = 0

    for sent_id, sent in enumerate(sents):
        sent2word.append([])
        for word_id, word in enumerate(sent):
            word2char.append([])
            word2sent.append([sent_id, word_id])
            word2char[-1].append(len(document))  # 起始字符位置
            document += word
            word2char[-1].append(len(document))  # 结束字符位置（不包含）
            document += ' '
            sent2word[-1].append(word_cnt)
            word_cnt += 1

    assert len(word2char) == len(word2sent) == sum(len(sent) for sent in sents) == word_cnt
    document = document[:-1]  # 去掉最后多余的空格

    # Step 2: 构建 char2word 映射（每个字符属于哪个词）
    CHAR_NUM = len(document)
    char2word = np.array([-1] * CHAR_NUM)
    for word_id, (start_idx, end_idx) in enumerate(word2char):
        if start_idx < CHAR_NUM:
            char2word[start_idx:end_idx] = word_id

    # Step 3: 使用 AllenNLP 进行共指分析
    result = predictor.predict(document=document)
    tokens = result["document"]
    clusters = result["clusters"]

    # Step 4: 构建 token 到字符位置的映射
    token2char = []
    start = 0
    for token in tokens:
        start = document.find(token, start)
        if start == -1:
            start = 0
            continue
        end = start + len(token)
        token2char.append((start, end))
        start = end

    # Step 5: 构建 char2cluster 映射（每个字符属于哪个共指簇）
    char2cluster = np.array([-1] * CHAR_NUM)
    for cluster_id, cluster in enumerate(clusters):
        for start_token, end_token in cluster:
            if start_token >= len(token2char) or end_token >= len(token2char):
                continue
            start_char = token2char[start_token][0]
            end_char = token2char[end_token][1]
            if end_char > CHAR_NUM:
                end_char = CHAR_NUM
            char2cluster[start_char:end_char] = cluster_id

    # Step 6: 构建 char2entity 映射，并统计 entity -> cluster 分布
    char2entity = np.array([-1] * CHAR_NUM)
    entity_clusters = defaultdict(Counter)

    for entity_id, entity in enumerate(entities):
        for mention_id, mention in enumerate(entity):
            sent_id, start_word, end_word = mention['sent_id'], mention['pos'][0], mention['pos'][1] - 1
            start_word_idx = sent2word[sent_id][start_word]
            end_word_idx = sent2word[sent_id][end_word]
            start_idx = word2char[start_word_idx][0]
            end_idx = word2char[end_word_idx][1]

            char2entity[start_idx:end_idx] = entity_id

            cluster_id = set(np.unique(char2cluster[start_idx:end_idx]))
            cluster_id.discard(-1)

            if len(cluster_id) > 1 and entity_id in entity_clusters:
                del entity_clusters[entity_id]
                break

            if cluster_id:
                entity_clusters[entity_id][cluster_id.pop()] += 1

    # Step 7: 将共指提及添加到对应实体中
    entities = sample['vertexSet']
    for entity_id, entity_cluster in entity_clusters.items():
        max_time = -1
        cluster_id = -1
        for k, v in entity_cluster.items():
            if v > max_time:
                cluster_id, max_time = k, v

        if cluster_id == -1:
            continue

        cluster = clusters[cluster_id]

        for start_token, end_token in cluster:
            if start_token >= len(token2char) or end_token >= len(token2char):
                continue
            start_char = token2char[start_token][0]
            end_char = token2char[end_token][1]

            if all(np.unique(char2entity[start_char:end_char]) == -1):
                word_ids = np.unique(char2word[start_char:end_char])
                word_ids = sorted(list(word_ids))
                if -1 in word_ids:
                    word_ids.pop(0)
                if not word_ids:
                    continue

                start_sent_id, start_word_id = word2sent[word_ids[0]]
                end_sent_id, end_word_id = word2sent[word_ids[-1]]

                assert start_sent_id == end_sent_id, f"{doc_id}/{entity_id}/{tokens[start_token:end_token + 1]}"

                entities[entity_id].append({
                    "sent_id": start_sent_id,
                    "pos": [start_word_id, end_word_id + 2],
                    "name": document[start_char:end_char],
                    "type": entities[entity_id][0]['type'],
                    "coref": True
                })

    return sample


def gen_dataset_coref(predictor, filename, split):
    dataset = json.load(open(filename))
    for doc_id, sample in tqdm(enumerate(dataset), desc=f"gen {split} data coref:", ncols=100, total=len(dataset)):
        gen_coref(predictor, doc_id, sample)

    json.dump(dataset, open(f"{split}_coref.json", "w"))


if __name__ == '__main__':
    # 加载 AllenNLP 共指解析模型（需提前下载模型文件）
    model_path = "../PLM/coref-spanbert-large-2021.03.10.tar.gz"
    allen_coref = Predictor.from_path(model_path)

    filepath1 = "../DocRED/train_annotated.json"
    gen_dataset_coref(allen_coref, filepath1, "train_annotated")

    filepath = "../DocRED/dev.json"
    gen_dataset_coref(allen_coref, filepath, "dev")

    filepath2 = "../re-docred/dev_revised.json"
    gen_dataset_coref(allen_coref, filepath2, "dev_revised")
    # filepath3 = "../re-docred/test_revised.json"
    # gen_dataset_coref(allen_coref, filepath3, "test_revised")
    filepath4 = "../re-docred/train_revised.json"
    gen_dataset_coref(allen_coref, filepath4, "train_revised")

