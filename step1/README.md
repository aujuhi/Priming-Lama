# Priming LAMAS: 
## Evaluating Biases Encoded in BERT and their Impact on Fact Checking Models


Anke Unger


![Design Overview](https://github.com/aujuhi/Priming-Lama/blob/master/designOverview.PNG)


### About
This directory contains the code to generate the input data and analyse the results of Step 1

### Sentence retrieval and masking
The sentence retrieval is based on a [2006 dump of wikipedia](http://www.lrec-conf.org/proceedings/lrec2010/pdf/222_Paper.pdf).  
(1) The sentences are retrieved and pre-filtered in [retrieve_wiki_sents.py](https://github.com/aujuhi/Priming-Lama/blob/master/step1/retrieve_wiki_sents.py)  
(2) The created and saved sentences can then be masked in [mask_geowords.py](https://github.com/aujuhi/Priming-Lama/blob/master/step1/mask_geowords.py)  

NOTE: I added the lama_probe_automated.py file. The functions are part of the LAMA-probing
project and it therefore only works with the project from facebookresearch, just add 
this file to their cloned project


### Priming
Primes are retrieved from a [Fox-News comment corpus](https://github.com/sjtuprog/fox-news-comments)    
Fox News comments can be converted to primes in [priming_fox_news.py](https://github.com/aujuhi/Priming-Lama/blob/master/step1/priming_fox_news.py)  

### Analyse results
Predictions are created using the LAMA-set up  
To analyse ranks and effects of the created predictions use [analyse_results_step_1.py](https://github.com/aujuhi/Priming-Lama/blob/master/step1/analyse_results_step_1.py)    