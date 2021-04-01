import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random as rand

#
# Feverdata class and training arguments taken
# from thepythoncode.com/article/
# finetuning-bert-using-huggingface-transformers-python
#

model_name = "bert-base-uncased"
max_length = 512
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
labels_dec = {'SUPPORTS': 2, 'NOTENOUGHINFO' : 1, 'REFUTES': 0}


def create_NEI_train_set(data_path, fever_file, k=99):
    """
    remove a third of the evidence and raplace it with unrelated evidence
    and change label to NEI
    :param data_path: path to Eraser Fever data
    :param k number of shots for few shot training
    :return: None
    """
    nei_set = []
    sup_set = []
    ref_set = []
    ids = []
    with open(data_path + fever_file, 'r') as fevFiles: #replace with val.jsonl to evaluate finetuned model on NEI
        lines = fevFiles.readlines()
    rand.shuffle(lines)
    for line in lines:
        item = json.loads(line)
        if len(item['docids']) == 1 and len(item["evidences"]) == 1 and len(item["evidences"][0]) == 1:
            evid_dic = item["evidences"][0][0]

            # get doc with evidence
            try:
                with open(data_path + "docs/" + evid_dic['docid'], 'r', encoding="utf8") as docFile:
                    sentences = [sentence.strip() for sentence in docFile.readlines()]
            except FileNotFoundError:
                continue
            if len(sentences) < 3:
                continue
            index_real_evid = sentences.index(evid_dic["text"])
            id = item['annotation_id']
            # create train set with one third NEI claims
            emerg_break = 0
            if len(nei_set) < k / 3:
                r = rand.randint(1, len(sentences) - 1)
                while (r == index_real_evid) and emerg_break < 10:
                    emerg_break += 1
                    r = rand.randint(1, len(sentences) - 1)
                sentence = sentences[r]
                sentence = replace_quotes(sentence)
                evid_dic["text"] = sentence
                evid_dic["start_sentence"] = r
                item["classification"] = "NOTENOUGHINFO"
                item["evidences"][0][0] = evid_dic
                nei_set.append(item)
                ids.append(id)
                continue
            if item["classification"] == "SUPPORTS" and len(sup_set) < k / 3 and id not in ids:
                sup_set.append(item)
                ids.append(id)
                continue
            if item["classification"] == "REFUTES" and len(ref_set) < k / 3 and id not in ids:
                ref_set.append(item)
                ids.append(id)
    train_set = ref_set + sup_set + nei_set
    rand.shuffle(train_set)
    res_name = data_path + fever_file.split(".")[0] + str(k) + ".jsonl"
    with open(res_name, "w+", encoding="utf8") as res_f:
        for line in train_set:
            json.dump(line, res_f)
            res_f.write("\n")


def replace_quotes(sentence):
    """
    replace quotes in trainingdata to allow readability with json
    :param sentence: sentence from unrelated evidence
    :return:
    """
    left = True

    for i, ch in enumerate(sentence):
        if ch == "\"":
            if left:
                sentence[i] = "-LRB-"
            else:
                sentence[i] = "-RRB-"
    return sentence


def preprocessing(data_path):
    """
    preprocess training data
    :param data_path:
    :return:
    """
    X = []
    Y = []
    with open(data_path + "train99.jsonl", 'r') as fevFiles:
        for line in fevFiles.readlines():
            item = json.loads(line)
            evidence = ""
            for evids in item["evidences"]:
                for e in evids:
                    evidence += e["text"]
                    continue
            X.append((item['query'], evidence))
            Y.append(labels_dec[item['classification']])
    return X, Y


create_NEI_train_set("fever/", "train.jsonl")
create_NEI_train_set("fever/", "train.jsonl", 30000)
X, Y = preprocessing("fever/")
train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.5)
train_encodings = tokenizer.batch_encode_plus(train_x, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
valid_encodings = tokenizer.batch_encode_plus(valid_x, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

class FeverData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] =  torch.tensor([self.labels[idx]])
        print(item)
        return item

    def __len__(self):
        return len(self.labels)

# convert to torch data
train_dataset = FeverData(train_encodings, train_y)
valid_dataset = FeverData(valid_encodings, valid_y)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_dec)).to("cuda")


def compute_metrics(pred):
    """
    compute metrics on trainingsset
    :param pred:
    :return:
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)
