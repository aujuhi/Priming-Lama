# Priming LAMAS: 
## Evaluating Biases Encoded in BERT and their Impact on Fact Checking Models



![Design Overview](https://github.com/aujuhi/Priming-Lama/blob/master/designOverview.PNG)


### About
This directory contains the code to generate the input data and analyse the results of Step 2

### Sentence retrieval and masking
Again, the sentence retrieval is based on a [2006 dump of wikipedia](http://www.lrec-conf.org/proceedings/lrec2010/pdf/222_Paper.pdf).  
(1) The sentences are retrieved and pre-filtered in [retrieve_wiki_sents.py](https://github.com/aujuhi/Priming-Lama/blob/master/step2/retrieve_wiki_sents.py)  
but the sentences for ethnicities can be reused for adjectives  
(2) And again the created and saved sentences can be masked in [mask_geowords.py](https://github.com/aujuhi/Priming-Lama/blob/master/step2/mask_geowords.py)  

### Antonyms
To create synonyms and antonyms of the adjectives from step 1 use [find_antonyms.py](https://github.com/aujuhi/Priming-Lama/blob/master/step2/find_antonyms.py)

### Analyse results
Predictions are created using the LAMA-set up  
To analyse ranks and effects of the created predictions use [analyse_results_step_2.py](https://github.com/aujuhi/Priming-Lama/blob/master/step2/analyse_results_step_2.py)    
(1) start with sort_results_closed to sort the json predictions of BERT  
(2) calculate the rank differences with rank_diff_closed  
(3) create_wordlists_columnwise creates the positive and negative wordlists of effective premises  
I additionally calculated z-values/outliers in excel but this part can be removed

### Synonym-Antonym pairs  
To find out if two words have a similar likelihood use [likelihood_pairs.py](https://github.com/aujuhi/Priming-Lama/blob/master/step2/likelihood_pairs.py)    
This file is part of the lama-probing project by facebook-research. If you want to  
use the functions, clone their repository and add this file.
