## NLBSE'23 Tool Competition: Code Comment Classification

This repository contains the data and results for the baseline classifiers the [NLBSE’23 tool competition](https://nlbse2023.github.io/tools/) on code comment classification.

Participants of the competition must use the provided data to train/test their classifiers, which should outperform the baselines.

Details on how to participate in the competition are found [here](https://colab.research.google.com/drive/1cW8iUPY9rTjZdXnGYtJ4ARBSISyKieWt#scrollTo=7ITz0v7mv4jV).

## Contents of this package
---
- [NLBSE'23 Tool Competition: Code Comment Classification](#nlbse23-tool-competition-code-comment-classification)
- [Contents of this package](#contents-of-this-package)
- [Folder structure](#folder-structure)
- [Data for classification](#data-for-classification)
- [Dataset Preparation](#dataset-preparation)
- [Software Projects](#software-projects)
- [Baseline Model Features](#baseline-model-features)

## Folder structure
- ### Java
    - `classifiers`:  We have trained Random Forest classifiers (also provided in the folder) on the selected sentence categories. 
    - `input`: The CSV files of the sentences for each category (within a training and testing split). **These are are the main files used for classification**. See the format of these files below.
    - `results`: The results contain a CSV file with the classification results of each classifier for each category.
    - `weka-arff`: ready-made input files for WEKA, with TF_IDF and NLP features extracted from the sentences (more information below). 
    - `project_classes`: CSV files with the list of classes for each software project and corresponding code comments.
- ### Pharo
  Same structure as Java.
- ### Python 
  Same structure as Java.

## Data for classification

We provide a CSV file for each programming language (in the `input` folder) where each row represent a sentence (aka an instance) and each sentence contains six columns as follow:
- `comment_sentence_id` is the unique sentence ID;
- `class` is the class name referring to the source code file where the sentence comes from;
- `comment_sentence` is the actual sentence string, which is a part of a (multi-line) class comment;
- `partition` is the dataset split in training and testing, 0 identifies training instances and 1 identifies testing instances, respectively;
- `instance_type` specifies if an instance actually belongs to the given category or not: 0 for negative and 1 for positive instances;
- `category` is the ground-truth or oracle category.


## Dataset Preparation

- **Preprocessing**. Before splitting, the manually-tagged class comments were preprocessed as follows:
    - We changed the sentences to lowercase, reduced multiple line endings to one, and removed special characters except for  `a-z0-9,.@#&^%!? \n`  since different languages can have different meanings for the symbols. For example, `$,:{}!!` are markup symbols in Pharo, while in Java it is `‘/* */ <p>`, and `#,`  in Python. For simplicity reasons, we removed all such special character meanings.
    - We replaced periods in numbers, "e.g.", "i.e.", etc, so that comment sentences do not get split incorrectly. 
    - We removed extra spaces before and after comments or lines. 

- **Splitting sentences**.
    - Since the classification is sentence-based, we split the comments into sentences. 
    - As we use NEON tool to extract NLP features, we use the same splitting method to split the sentence. It splits the sentences based on selected characters `(\\n|:)`. This is another reason to remove some of the special characters to avoid unnecessary splitting. 
    - Note: the sentences may not be complete sentences. Sometimes the annotators classified a relevant phrase a sentence into a category. 
- **Partition selection**.  
    - After splitting comments into  sentences, we split the sentence dataset in an 80/20 training-testing split. 
    - The partitions are determined based on an algorithm in which we first determine the stratum of each class comment. The original paper gives more details on strata distribution. 
    - Then, we follow a round-robin approach to fill training and testing partitions from the strata. We select a stratum, select the category with a minimum number of instances in it to achieve the best balancing, and assign it to the train or test partition based on the required proportions. 

- **Feature preparation**. We use two kinds of features: TEXT and NLP. 
    - For NLP features, we prepare a term-by-document matrix M, where each row represents a comment sentence (i.e., a sentence belongs to our language dataset composing CCTM) and each column represents the extracted feature. 
    - To extract the NLP features, we use NEON, a tool proposed in the previous work of Andrea Di Sorbo. The tool detects NLP patterns from natural language-based sentences. We add the identified NLP patterns as feature columns in M, where each of them models the presence (modeled by 1) or absence (modeled by 0) of an NLP pattern in the comment sentences. In sum, each i\_th row represents a comment sentence, and j_th represents an NLP feature.
    - For the TEXT features, we apply typical preprocessing steps such as stop word removal, stemmer, and convert it a vector based on the TF-IDF approach. The first attribute is a sentence. In case of TEXT features, the rows are the comment sentences and the column represents a term contained in it. Each cell of the matrix represents the weight (or importance) of the j\_th term contained in the i_th comment sentence. The terms in M are weighted using the TF–IDF score. 
    - We prepare such Matrix M for each category of each language. The last column of the Matrix represents the category. 

- **Classification**. We used Weka to classify the comment sentences into each category using the Random Forest model (the baseline).

- *Evaluation*. We evaluated our baseline models (i.e., for each category) using standard evaluation metrics, precision, recall, and F1-score. 

## Software Projects
We extracted the class comments from selected projects.

- ### Java 
     Details of six java projects. 
    - Eclipse:  The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Eclipse](https://github.com/eclipse).
    
    - Guava: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Guava](https://github.com/google/guava).
    
    - Guice: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Guice](https://github.com/google/guice).
    
    - Hadoop:  The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Apache Hadoop](https://github.com/apache/hadoop)
    
    - Spark.csv: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Apache Spark](https://github.com/apache/spark)
    
    - Vaadin: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Vaadin](https://github.com/vaadin/framework)
   
- ### Pharo
     Contains the details of seven Pharo projects.     
    - GToolkit: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo.  
     
    - Moose: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. 
     
    - PetitParser: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo.
    
    - Pillar: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo.
    
    - PolyMath: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo.
    
    - Roassal2: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo.
    
    - Seaside: The version of the project referred to extracted class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo.

- ### Python
     Details of the extracted class comments of seven Python projects. 
    - Django: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Django](https://github.com/django)
    
    - IPython: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub[IPython](https://github.com/ipython/ipython)
    
    - Mailpile: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Mailpile](https://github.com/mailpile/Mailpile)
        
    - Pandas: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [pandas](https://github.com/pandas-dev/pandas)
        
    - Pipenv: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Pipenv](https://github.com/pypa/pipenv)
        
    - Pytorch: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [PyTorch](https://github.com/pytorch/pytorch)
        
    - Requests: The version of the project referred to extract class comments is available as [Raw Dataset](https://doi.org/10.5281/zenodo.4311839) on Zenodo. More detail about the project is available on GitHub [Requests](https://github.com/psf/requests/)

## Baseline Model Features
`0-0-<category>-<Feature-Set>.arff`       - ARFF format of the input file for a classifier for a "category" with the set of "feature". The feature set are TEXT (tfidf), NLP (heuristic). For example:   
 - [0-0-summary-tfidf-heuristic.arff](/Java/weka-arff/data/0-0-summary-tfidf-heuristic.arff) input training file for a classifier for the summary category with the TEXT (tfidf) features and the NLP (heuristic) features.
- [1-0-summary-tfidf-heuristic.arff](/Java/weka-arff/data/1-0-summary-tfidf-heuristic.arff)  - input testing file for a classifier for the summary category with the TEXT (tfidf) features and the NLP (heuristic) features.

<!---
## Baseline Results

**Note**: the baseline results in these files are currently outdated. We will update the baseline results soon.

`0-0-summary-tfidf-heuristic-randomforest-outputs.csv’ - The CSV output file stores the results for a particular category  “summary” with a feature set of “tfidf-heauristic” on a classifier “randomforest”. 
The results contain a confusion matrix of true positive (tp), false positive (fp), true negative (tn), and false negative (fn). We also have weighted precision (w\_pr), weighted recall (w\_re), and weighted f-measure (w\_f_measure). 
-->