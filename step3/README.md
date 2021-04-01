# Priming LAMAS: 
## Evaluating Biases Encoded in BERT and their Impact on Fact Checking Models

![Design Overview](https://github.com/aujuhi/Priming-Lama/blob/master/designOverview.PNG)

### About
This directory contains the code to finetune the Fact Checker, manipulate the FEVER evidence  
and to conduct the FC experiment

### Finetune BERT
Finetune BERT using [finetune_fever.py](https://github.com/aujuhi/Priming-Lama/blob/master/step3/finetune_fever.py)  
(1) Trainingdata can be created with create_NEI_train_set 
(2) The finished model can be evaluated with [evaluate_finetuned.py](https://github.com/aujuhi/Priming-Lama/blob/master/step3/evaluate_finetuned.py)  

### Conduct Experiment
The FEVER data can be preparedfor the experiment with [prepro_manipulate.py](https://github.com/aujuhi/Priming-Lama/blob/master/step3/prepro_manipulate.py)  
and conducted with [predict_fact_checking.py](https://github.com/aujuhi/Priming-Lama/blob/master/step3/predict_fact_checking.py)  
