# :potable_water:  Drinking Water Potability Report :potable_water:

[Jérôme Auguste](https://github.com/jerome-auguste) - [Marco Boucas](https://github.com/marcoboucas) - [Ariane Dalens](https://github.com/ArianeDlns) 

Project for the Machine Learning course @CentraleSupélec

Based on the [Kaggle Challenge Drinking_Water_Potability](https://www.kaggle.com/artimule/drinking-water-probability)

### Context
Access to safe drinking water is essential to health, a basic human right, and a component of effective policy for health protection. This is important as a health and development issue at a national, regional, and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.

### Dataset 
The dataset is available at this link [drinking_water_potability.csv](https://www.kaggle.com/artimule/drinking-water-probability/download). The file file contains water quality metrics for 3276 different water bodies. The easiest way to run the following notebooks using this dataset is to unzip the csv file in the root folder of the repo.

|                     |     Type     | count |   mean   |   min  |    max   |
|:-------------------:|:------------:|:-----:|:--------:|:------:|:--------:|
| ph                  | float        | 2785  | 7,08     | 0      | 14,00    |
| Hardness            | float        | 3276  | 196,37   | 47,43  | 323,12   |
| Solids              | float        | 3276  | 22014,09 | 320,94 | 61227,20 |
| Chloramines         | float        | 3276  | 7,12     | 0,35   | 13,13    |
| Sulfate             | float        | 2495  | 333,78   | 129,00 | 481,03   |
| Conductivity        | float        | 3276  | 426,21   | 181,48 | 753,34   |
| Organic_carbon      | float        | 3276  | 14,28    | 2,20   | 28,30    |
| Trihalomethanes     | float        | 3114  | 66,40    | 0,74   | 124,00   |
| Turbidity           | float        | 3276  | 3,97     | 1,45   | 6,74     |
| Potability (target) | categorical  | 3276  | 39%      | 0      | 1        |

<!--- [Overleaf report](https://fr.overleaf.com/read/jznbtvznsrfb) --->

## :package: Structure of the project

First, we sought to visualise our data, to study correlations, missing data and outliers. 
TThis led to an exploration and visualisation of our data in the notebook [data_exploration_and_visualization.ipynb](https://github.com/ArianeDlns/ML-Drinking_Water_Potability/blob/main/Data_exploration_and_visualization.ipynb). 

After processing and preprocessing the data we compared and finetuned different models in the notebook [pipeline.ipynb](https://github.com/ArianeDlns/ML-Drinking_Water_Potability/blob/main/Pipeline.ipynb) which led us to our final model

## :trophy: Winning Model for this dataset

After our research on which model is the most relevant for this problem, we selected 3 different well-performing algorithms (K-NN, SVM and Random Forest). Each one was then fine-tuned and work collaboratively in a voting classifier.

## References

[1] Lundberg, Scott M and Lee, Su-In, [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf), _Advances in Neural Information Processing Systems 30_ (2017)  

[2] Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani, [An introduction to stastical learning](https://centralesupelec.edunao.com/pluginfile.php/171459/course/section/30032/2013_Book_AnIntroductionToStatisticalLea.pdf), _Springer_  

[3] Friedman, Jerome H., The elements of statistical learning: Data mining, inference, and prediction, _Springer_ 