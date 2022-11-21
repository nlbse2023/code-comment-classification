## NLBSE'23 Tool Competition: Code Comment Classification

This repository contains the data and results for the baseline classifiers the [NLBSE’23 tool competition](https://nlbse2023.github.io/tools/) on code comment classification.

Participants of the competition must use the provided data to train/test their classifiers, which should outperform the baselines.

## Contents of this package
---
- [NLBSE'23 Tool Competition: Code Comment Classification](#NLBSE-code-comment-classification)
  - [## Contents of this package](#-content-of-the-replication-package)
  - [Folder strcuture](#folder-strucutre)
    - [db.sqlite](#db.sqlite)
    - [Java](#java)
    - [Pharo/](#pharo)
    - [Python/](python)
  - [About the Pipeline](#about-the-pipeline)
  - [Projects](#projects)
  - [Model Features](#model-features)
  - [Results](#results)

## Folder strcuture:
- ### db.sqlite: contains data in SQLite format.
- ### Java/
    - classifiers:  We have trained RandomForest classifiers (also provided in the folder) on the selected categories. 
    - input: The input files (training and testing split) of the categories of our interest is provided in the input folder. The folder contains, raw,  preprocessed, and partitioned files for all categories as well.
    - results: The results contain the CSV file output of each classifier for each selected category.
    - weka-arff: readymade input files for WEKA. 
- ### Pharo/- same structure as Java.
- ### Python/ - same structure as Java.

## About the Pipeline: 
- Raw dataset:  ground truth (manually labeled class comments) for each language, e.g.,  `java_0_raw.csv` contains the classified for JAVA. 

- Preprocessing: All the manually-labeled class comments in raw files in our dataset are used as ground truth to classify the unseen class comments. 
    - As a common preprocessing step, we change the sentences to lowercase, reduce multiple line endings to one, and remove special characters except for  `a-z0-9,.@#&^%!? \n`  since different languages can have different meanings for the symbols. For example, `$,:{}!!` are markup symbols in Pharo, while in Java it is `‘/* */ <p>`, and `#,`  in python. For simplicity reasons, we remove all such special character meanings.
    - We replace periods in floats, e.g., and, i.e., etc, so that sentences do not get split intermediately. 
    - We remove extra spaces before and after comments or lines. 

- Splitting sentences: 
    - Since the classification is sentence-based, we split the comments into sentences. 
    - As we use NEON tool to extract NLP features, we use the same splitting method to split the sentence. It splits the sentences based on selected characters `(\\n|:)`. This is another reason to remove some of the special characters to avoid unnecessary splitting. 
    - Note: An important note is that not all sentences classified into various categories are complete sentences. Sometimes the authors classify a relevant part of big sentence into a category. 
- Partition selection:  
    - After splitting the sentences, we split the dataset in an 80-20 training and testing split. 
    - The partitions are determined based on an algorithm in which we first determine the stratum of each class comment. The paper gives more details on strata distribution. 
    - Then we follow a round-robin approach to fill partitions from strata. We select a stratum, select the category with a minimum number of instances in it to achieve the best balancing, and assign it to the train or test partition based on the required proportions. 

- Feature preparation: We use two kinds of features: TEXT and NLP. 
    - For NLP features, we prepare a term-by-document matrix M, where each row represents a comment sentence (i.e., a sentence belongs to our language dataset composing CCTM) and each column represents the extracted feature. 
    - To extract the NLP features, we use NEON, a tool proposed in the previous work of Andrea Di Sorbo. The tool detects NLP patterns from natural language-based sentences. We add the identified NLP patterns as feature columns in M, where each of them models the presence (modeled by 1) or absence (modeled by 0) of an NLP pattern in the comment sentences. In sum, each i\_th row represents a comment sentence, and j_th represents an NLP feature.
    - For the TEXT features or attributes, we apply typical preprocessing steps such as stop word removal, stemmer, and convert it a vector. The first attribute is a sentence. In case of TEXT features, the rows are the comment sentences and the column represents a term contained in it. Each cell of the matrix represents the weight (or importance) of the j\_th term contained in the i_th comment sentence.The terms in M are weighted using the TF–IDF score. 
    - We prepare such Matrix M for each category of each language. The last column of the Matrix represents the category. 

- Classification : We use Weka to classify the comment sentences into each category. We experiment (relying on the Weka tool) with the machine-learning technique: the Random Forest model.

- Evaluation: We evaluate our models using standard evaluation metrics, precision, recall, and f-measure. 

## Projects:
Contains the extracted class comments from the selected projects of each language used in this study. 

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

## Model Features:
`0-0-<category>-<Feature-Set>.arff`       - ARFF format of the input file for a classifier for a "category" with the set of "feature". The feature set are TEXT (tfidf), NLP (heuristic). For example:   
 - [0-0-summary-tfidf-heuristic.arff](/Java/weka-arff/data/0-0-summary-tfidf-heuristic.arff) input training file for a classifier for the summary category with the TEXT (tfidf) features and the NLP (heuristic) features.
- [1-0-summary-tfidf-heuristic.arff](/Java/weka-arff/data/1-0-summary-tfidf-heuristic.arff)  - input testing file for a classifier for the summary category with the TEXT (tfidf) features and the NLP (heuristic) features.

## Results:
`0-0-summary-tfidf-heuristic-randomforest-outputs.csv’ - The CSV output file stores the results for a particular category  “summary” with a feature set of “tfidf-heauristic” on a classifier “randomforest”. 
The results contain a confusion matrix of true positive (tp), false positive (fp), true negative (tn), and false negative (fn). We also have weighted precision (w\_pr), weighted recall (w\_re), and weighted f-measure (w\_f_measure). 
