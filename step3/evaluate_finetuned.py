from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
import json
from sklearn.metrics import classification_report


target_names = ["REFUTES", "NOTENOUGHINFO", "SUPPORTS"]
model = BertForSequenceClassification.from_pretrained("/model_path", num_labels=len(target_names))
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)


def get_prediction(claim_evid_tup):
    """
    Get Fact Checking label
    :param claim_evid_tup: Array of tupel with claim and evidence
    :return:
    """
    # tokenize claim evidence tupel
    inputs = tokenizer.batch_encode_plus(claim_evid_tup, truncation=True, padding=True, max_length=512,
                                         return_tensors='pt')
    # infer predictions
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return target_names[probs.argmax()]


def eval_fact_checker(val_data):
    """
    get evaluation metrics for finetuned Fact Checker
    :param val_data: path to the validation Fever data (create version with NEI with finetune class first)
    :return: None
    """
    with open(val_data, 'r', encoding="utf16") as fevFiles:
        items = [json.loads(line) for line in fevFiles.readlines()]
    y_true = []
    y_pred = []
    for item in items:
        pred = get_prediction([(item['query'], item['evidences'][0][0]['text'])])
        print("Target value:", item['classification'], "Prediction:", pred)
        y_true.append(target_names.index(item['classification']))
        y_pred.append(target_names.index(pred))
    print(classification_report(y_true, y_pred, target_names=target_names))
