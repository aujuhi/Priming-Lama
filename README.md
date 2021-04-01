# Priming LAMAS: 
## Evaluating Biases Encoded in BERT and their Impact on Fact Checking Models


![Design Overview](https://github.com/aujuhi/Priming-Lama/blob/master/designOverview.PNG)


### About
Deep pre-trained Language Models contain social bias which likely persists even after finetuning a model for a specific task. We created an economic methodology to identify social biases. Closing the gap between the pre-trained and finetuned form, we evaluated the impact of the found biases on a finetuned Fact Checker. For that, we created a set of LAMA-probes to detect word-country or word-ethnic correlations in the Language Model BERT. We then manipulated the input of a Fact Checker with the found correlated pairs. We showed, that the Fact Checker's predictions changed in accordance to the polarity of the word-correlation.

### Prerequisites
*`Python 3.0` or further  
*I suggest the usage of `anaconda`  

for further requirements, please read the requirement file  

To filter for correlated words start with the sentence retrieval in step1/  


### Credit
The project uses a Fox-News comment corpus from [sjtuprog](https://github.com/sjtuprog)  
and a wikipedia corpus from Samuel Reese, Gemma Boleda, Montse Cuadros, Lluís Padró and German Rigau  
It is also based on the LAMA-probing setup from [facebookresearch](https://github.com/facebookresearch)
