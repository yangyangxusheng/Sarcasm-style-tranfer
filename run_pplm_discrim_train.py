#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import argparse
import csv
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim

import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from transformers import DistilBertTokenizer, DistilBertModel


from pplm_classification_head import ClassificationHead

TRANSFORMERS_CACHE='./working_dir/GPT2-M-R' # 指定模型的下载路径, gpt2-large
torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10 # LayerNorm 层中的最小除数
# example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
example_sentence1 = "richard branson's global-warming donation nearly as much as cost of failed balloon trips."
example_sentence2 = "my husband got a ruben sandwich from Amazon."
example_sentence3 = "Ahh, the glorious american cuisine."
example_sentence4 = "You are sick of being told what's best for you."
example_sentence5 = "School leader said she hopes to get a $100 million-dollar-per-year."
example_sentence6 = "I have used my phone for a year now and and have you you know you are right. I have tried to."
max_length_seq = 100  # 默认100, 只使用数据集中长度小于100的数据
min_length_seq = 0  # 这是自己加的， 长度大于0的数据才能使用


class Discriminator(torch.nn.Module):  # 检测器模型使用的参数,定义预训练模型
    """Transformer encoder followed by a Classification Head transformer编码器"""

    def __init__(  # 定义类中可调用的对象，初始化，使用哪些函数和层
            self,
            class_size=None, # 有几个分类
            pretrained_model="gpt2-medium", # moren gpt2-medium , gpt2-medium
            classifier_head=None,  # 头的处理
            cached_mode=False, # 存函数和方法到内存中
            device='cpu' # 使用CPU
    ):
        super(Discriminator, self).__init__() # 继承父类
        if pretrained_model.startswith("gpt2"):  # 使用gpt作为与预训练模型时
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            # 定义模型解析器 ， tokenizer 得到的数据是：字典的id input_ids， 上下文的表示 0， 1 ， mask的值
            self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            # 编码器， 即调用模型
            self.embed_size = self.encoder.transformer.config.hidden_size
            # 词向量的维度，模型的配置单中，写出的词向量的维度
        elif pretrained_model.startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.encoder = BertModel.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.embed_size = self.encoder.config.hidden_size
        elif pretrained_model.startswith("roberta"): # gpt2-medium
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.encoder = RobertaModel.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.embed_size = self.encoder.config.hidden_size
        elif pretrained_model.startswith("bar"): # gpt2-medium
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.encoder = AutoModelForSequenceClassification.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.embed_size = self.encoder.config.hidden_size
        elif pretrained_model.startswith("text"): # gpt2-medium
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.encoder = AutoModelForSequenceClassification.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.embed_size = self.encoder.config.hidden_size
        elif pretrained_model.startswith("distilbert"): # distilbert-base-uncased-finetuned-sst-2-english
            # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            # self.encoder = AutoModelForSequenceClassification.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.encoder = DistilBertModel.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.embed_size = self.encoder.config.hidden_size
        elif pretrained_model.startswith("micro"):
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE)
            self.encoder = AutoModelForCausalLM.from_pretrained(pretrained_model, cache_dir=TRANSFORMERS_CACHE) # 编码器
            self.embed_size = self.encoder.transformer.config.hidden_size # 嵌入
        else:
            raise ValueError(
                "{} model not yet supported".format(pretrained_model)
            )
        if classifier_head:
            self.classifier_head = classifier_head
            # 将词向量的维度映射成分类的维度
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            # 如果没有分类的尺寸，报错
            self.classifier_head = ClassificationHead(
                # 全连接，线性处理，将词向量维度映射成分类的维数，分类的过程就是一个处理词向量到分类结果的过程
                class_size=class_size,  # 分类的个数
                embed_size=self.embed_size  # 词向量的维度
            )
        self.cached_mode = cached_mode  # 缓存的模式，在内存中
        self.device = device  # 使用的是cpu 还是 gpu

    def get_classifier(self):
        return self.classifier_head
        # 返回映射成的全连接的概率，在输入softmax前的部分

    def train_custom(self): # 自定义的训练
        for param in self.encoder.parameters(): # param 预训练模型中的参数
            param.requires_grad = False # 不需要梯度更新
        self.classifier_head.train()  # 告诉head模型，这是训练的过程，需要batchnorm 和 dropout
        # batchnorm 就是 将 训练数据的分布规范， 加速训练的过程

    def avg_representation(self, x): # 平均表示
        mask = x.ne(0).unsqueeze(2).repeat( 1, 1, self.embed_size).float().to(self.device).detach()
                # 得到 x的 mask 最后的结果是 x的行数 词向量列的 那么多的 1 [1 , 1, 1, 1]
                # .ne(0)， ne 的意思 是 not equal ， 不等于，即 x不等于0
                # .unsqueeze(2) , 扩张维度 ， x扩张两个维度
                # repeat 复制函数， 将原来的x复制成 ， 1个通道， 1行， 词向量的列数，
                # .detach() x的梯度不会更新
        if hasattr(self.encoder, 'transformer'):
            # hasattr() 函数用于判断对象是否包含对应的属性， 如果使用的模型是gpt，就会有transfomormer类
            # for gpt2
            hidden, _ = self.encoder.transformer(x)
            # x是gpt模型的输入张量 ， hidden是gpt的输出隐层
        else:
            # bert 没有 transformer模型
            # for bert
            hidden, _ = self.encoder(x)
            # x是bert模型的输入， hidden是bert模型的输出
        masked_hidden = hidden * mask
        # 掩码的偏置层 ， 就是 输出的隐层 * mask（很多1）
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        # .sum() 就是求和，把一列所有行的位置相加， 有几行就有几个和的值，保存在1维列表中，
        # 除以 所有的1的和 如768
        # 这个平均表达的结果，就是模型的输出的表达形式，如1行, 768列
        return avg_hidden # 求出预训练模型加入掩码后的输出

    def forward(self, x): # 前馈 如果有设备的话，将设备送入cpu或gpu 否则使用平均表示法计算出transformer隐层
        if self.cached_mode:
            avg_hidden = x.to(self.device)
            # 如果是存在缓存中，直接取出来
        else:
            avg_hidden = self.avg_representation(x.to(self.device))
            # 如果没有在缓存中，使用平均表示计算

        logits = self.classifier_head(avg_hidden)
        # 全连接层的预测输出， 将预训练模型加上掩码的输出， 输入到头的处理模型，将输出 映射成 分类的数
        probs = F.log_softmax(logits, dim=-1) # softmax 把输出的分类均一化

        return probs # 得到分类的概率

    def predict(self, input_sentence): # 显示预测的结果
        input_t = self.tokenizer.encode(input_sentence)
        # 将输入的句子分词，然后映射成字典的ID
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        # 将输入模型的input_ids 转成tensor, 变成能被模型处理的格式
        if self.cached_mode:
            input_t = self.avg_representation(input_t)
        # 加入掩码的格式

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        # 将结果映射为成一行，变成概率
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob
        # 返回所有数的指数概率


class Dataset(data.Dataset):  # 数据集类，用来读入数据集，返回数据集中数据数据的索引和长度
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""  # 读入源和目标序列
        self.X = X # 输入的comment是X
        self.y = y # 输入的target（label） 是y

    def __len__(self):
        return len(self.X) # 返回输入的comment的长度

    def __getitem__(self, index): # 返回X和y的索引，拼成data {'comment', 'label'}
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]  # 0，1的索引
        return data


def collate_fn(data): # 动态补零，整理函数
    def pad_sequences(sequences): # padding 补零
        lengths = [len(seq) for seq in sequences]
        # lengths 是每个sequences中每个句子的长度拼成的列表

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()
        # 补零把每个句子都补齐到最大长度

        for i, seq in enumerate(sequences):
            end = lengths[i]
            # end 是结束的i
            padded_sequences[i, :end] = seq[:end]
            # 补了0的句子 是 补的是seq的前end换成原始的句子

        return padded_sequences, lengths
        # 返回补零后的序列和所有序列的长度[列表]

    item_info = {}
    for key in data[0].keys(): # 在数据的第0列的键值对
        item_info[key] = [d[key] for d in data]
        # item的属性就是data的属性

    x_batch, _ = pad_sequences(item_info["X"])
    # x批， 是补零后的句子X，是预测的comment
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)
    # y批， 是label

    return x_batch, y_batch # 返x,y未分批


def cached_collate_fn(data): # 缓存的形式
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    # x批， 是补零后的句子X，是预测的comment
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)
    # y批， 是label
    return x_batch, y_batch # 返回x批和y批


def train_epoch(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'): # 训练过程： # 用gpu 输入 输出 计算loss 优化器 梯度更新
    samples_so_far = 0 # 已用样本数0
    discriminator.train_custom() # 不需要修改预训练模型梯度，随机取参数更新, 定义的是训练的过程
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()
        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()
        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), loss.item()
                )
            )


def evaluate_performance(data_loader, discriminator, device='cpu'):
    # 评估模型的性能
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 不需要回传梯度
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    # 模型的精度就是 正确的数据个数 / 全部二点个数

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * accuracy
        )
    )

    return test_loss, accuracy # 返回loss和精确度


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    # 预测输入的句子是不是讽刺的类型

    input_t = model.tokenizer.encode(input_sentence)
    # 对输入的句子分词和编码成字典ID
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    # 变成模型能接受的tensor形式
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    # 模型输出的概率，变成一列概率
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))
    # 输出其属于这个类的概率是多少


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):  # 缓存的datloader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)
    # 用来分批的data_loader , 数据集是输入的 ，分批大小， 不shuffle, 整理函数返回的是补零后的句子和他们的长度

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)): # 遍历有进度条
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach() # 分类器的输入x ，预训练模型的输出
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1)) # 把多维的张量，每一行做成一个单独的张量，拼成元组
            xs += avg_rep_list # xs 就是模型的输入的特征
            ys += y.cpu().numpy().tolist() # ys是模型的输入的label

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader  # 得到分批好的数据集


def get_idx2class(dataset_fp): # 处理分类
    classes = set()
    with open(dataset_fp) as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in tqdm(csv_reader, ascii=True):
            if row:
                classes.add(row[0]) # 将第一行的第一列作为分类

    return sorted(classes) # 得到好用的分类


def get_sarcasm_dataset(dataset_fp, tokenizer, device,
                        idx2class=['0', '1'], add_eos_token=False):
    # sarcasm数据集，dataset_fp是需要使用的数据集
    if not idx2class:
        idx2class = get_idx2class(dataset_fp)
    class2idx = {c: i for i, c in enumerate(idx2class)}

    x = []
    y = []
    with open('./datasets/train-sarcasm.tsv', encoding='utf-8') as f: # encoding='utf-8' 自己加的， 换数据集替换
        csv_reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tqdm(csv_reader, ascii=True)):
            if row:
                label = row[0]
                # label 是使用的标签
                text = row[1]
                # text 是 comment

                try:
                    seq = tokenizer.encode(text)
                    # 分词，编码 comment
                    if (len(seq) < max_length_seq and len(seq) > min_length_seq):
                        # 只保留长度合适的数据
                        if add_eos_token:
                            # 添加eos长度
                            seq = [50256] + seq
                        seq = torch.tensor(
                            # 序列转换成张量
                            seq,
                            device=device,
                            dtype=torch.long
                        )

                    else:
                        rightlength = '0 - 100' # 改过
                        print(
                            "Line {} is longer(shorter) than maximum(minimum) length {}".format( # 之前没有小括号里的内容
                                i, rightlength # 之前是 i , maxlength
                            ))
                        continue

                    x.append(seq)
                    # x 是comment
                    y.append(class2idx[label])
                    # y 是label

                except:
                    print("Error tokenizing line {}, skipping it".format(i))
                    pass

    return Dataset(x, y) # 把数据打包成data(x, y) 可以返回其属性索引


def train_discriminator( # 训练分类器
        dataset,# 数据集
        dataset_fp=None,
        pretrained_model="gpt2-medium", # 默认gpt2-medium
        epochs=10,
        learning_rate=0.0001,
        batch_size=64,
        log_interval=10,
        save_model=False,
        cached=False,
        no_cuda=False,
        output_fp='meta'
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    add_eos_token = pretrained_model.startswith("gpt") # 默认gpt

    if save_model: # 保存模型
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)

    classifier_head_meta_fp = os.path.join(
        output_fp, "{}_classifier_head_meta.json".format(dataset)
        # 从数据集中保存参数
    )
    classifier_head_fp_pattern = os.path.join(
        output_fp, "{}_classifier_head_epoch".format(dataset) + "_{}.pt"
    )
        # 把偶才能训练轮数的参数

    print("Preprocessing {} dataset...".format(dataset)) # 处理数据集
    start = time.time()

    if dataset == "SST": # 训练SST
        idx2class = ["positive", "negative", "very positive", "very negative",
                     "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
        )

        x = []
        y = []
        for i in trange(len(train_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(train_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            if add_eos_token:
                seq = [50256] + seq
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(class2idx[vars(train_data[i])["label"]])
        train_dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in trange(len(test_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(test_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            if add_eos_token:
                seq = [50256] + seq
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(class2idx[vars(test_data[i])["label"]])
        test_dataset = Dataset(test_x, test_y)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 2,
        }

    elif dataset == "clickbait":
        idx2class = ["non_clickbait", "clickbait"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        with open("datasets/clickbait/clickbait.txt") as f:
            data = []
            for i, line in enumerate(f):
                try:
                    data.append(eval(line))
                except:
                    print("Error evaluating line {}: {}".format(
                        i, line
                    ))
                    continue
        x = []
        y = []
        with open("datasets/clickbait/clickbait.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        if add_eos_token:
                            seq = [50256] + seq
                        seq = torch.tensor(
                            seq, device=device, dtype=torch.long
                        )
                    else:
                        print("Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                        continue
                    x.append(seq)
                    y.append(d["label"])
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 1,
        }

    elif dataset == "toxic":
        idx2class = ["non_toxic", "toxic"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        x = []
        y = []
        with open("datasets/toxic/toxic_train.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        if add_eos_token:
                            seq = [50256] + seq
                        seq = torch.tensor(
                            seq, device=device, dtype=torch.long
                        )
                    else:
                        print("Line {} is longer than maximum length {}".format(
                            i, max_length_seq
                        ))
                        continue
                    x.append(seq)
                    y.append(int(np.sum(d["label"]) > 0))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    else:
        if dataset == "sarcasm":
        # This assumes the input dataset is a TSV with the following structure:
        # class \t text

        # if dataset_fp is None:
        #     raise ValueError("When sarcasm dataset is selected, "
        #                      "dataset_fp needs to be specified aswell.")

            idx2class = ['0', '1']  # 默认get_idx2class('./datasets/train-sarcasm-10000.tsv')

            discriminator = Discriminator(
                class_size=len(idx2class),
                pretrained_model=pretrained_model,
                cached_mode=cached,
                device=device
            ).to(device)

            full_dataset = get_sarcasm_dataset(
                dataset_fp, discriminator.tokenizer, device,
                idx2class=idx2class, add_eos_token=add_eos_token
            )
            train_size = int(0.9 * len(full_dataset))
            # 训练集 0.9
            test_size = len(full_dataset) - train_size
            # 测试集 0.1
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, test_size]
            )
            # 训练集 和 测试集 随机划分

            discriminator_meta = {
                "class_size": len(idx2class),
                "embed_size": discriminator.embed_size,
                "pretrained_model": pretrained_model,
                "class_vocab": {c: i for i, c in enumerate(idx2class)},
                "default_class": 0,
            }

    end = time.time()
    print("Preprocessed {} data points".format(
        len(train_dataset) + len(test_dataset))
    )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,
            shuffle=True, device=device
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator, device=device
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)

    with open(classifier_head_meta_fp, "w") as meta_file:
        json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)



    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        test_loss, test_accuracy = evaluate_performance(
            data_loader=test_loader,
            discriminator=discriminator,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print("\nExample prediction")
        predict(example_sentence1, discriminator, idx2class,
                cached=cached, device=device)
        predict(example_sentence2, discriminator, idx2class,
                cached=cached, device=device)
        predict(example_sentence3, discriminator, idx2class,
                cached=cached, device=device)
        predict(example_sentence4, discriminator, idx2class,
                cached=cached, device=device)
        predict(example_sentence5, discriminator, idx2class,
                cached=cached, device=device)
        predict(example_sentence6, discriminator, idx2class,
                cached=cached, device=device)


        # if save_model:
        # torch.save(discriminator.state_dict(), "\{}_discriminator_{}.pt".format(args.dataset, epoch + 1))
        torch.save(discriminator.get_classifier().state_dict(),
                       classifier_head_fp_pattern.format(epoch + 1))

    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    print("Test performance per epoch")
    print("epoch\tloss\tacc")
    for e, (loss, acc) in enumerate(zip(test_losses, test_accuracies)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))

    return discriminator, discriminator_meta


def load_classifier_head(weights_path, meta_path, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params


def load_discriminator(weights_path, meta_path, device='cpu'):
    classifier_head, meta_param = load_classifier_head(
        weights_path, meta_path, device
    )
    discriminator =  Discriminator(
        pretrained_model=meta_param['pretrained_model'],
        classifier_head=classifier_head,
        cached_mode=False,
        device=device
    )
    return discriminator, meta_param


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="sarcasm",
                        choices=("SST", "clickbait", "toxic", "sarcasm"),
                        # choices=("sarcasm", "SST", "clickbait", "toxic", "sarcasm"),
                        help="dataset to train the discriminator on."
                             "In case of sarcasm, the dataset is expected"
                             "to be a TSBV file with structure: class \\t text") # 在这选择需要的分类器
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of sarcasm datadset")
    parser.add_argument("--pretrained_model", type=str, default="gpt2-medium", # gpt2-medium gpt2-medium gpt2-medium
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=15, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,# 默认0.0001
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=100, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save_model", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", default=False, action="store_true",
                        help="use to turn off cuda")
    parser.add_argument("--output_fp", default='./working_dir/GPT2-M-R',
                        help="path to save the output to")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
