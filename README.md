# :potable_water:  Drinking Water Potability Report :potable_water:
Project for the Machine Learning course @CentraleSupélec

 [Jérôme Auguste](https://github.com/jerome-auguste) - [Marco Boucas](https://github.com/marcoboucas) - [Ariane Dalens](https://github.com/ArianeDlns) 

Based on the [Kaggle Challenge Drinking_Water_Potability](https://www.kaggle.com/artimule/drinking-water-probability)

### Context
Access to safe drinking water is essential to health, a basic human right, and a component of effective policy for health protection. This is important as a health and development issue at a national, regional, and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.

### Content
The dataset is available at this link [drinking_water_potability.csv](https://www.kaggle.com/artimule/drinking-water-probability/download). The file file contains water quality metrics for 3276 different water bodies. The easiest way to run the following notebooks using this dataset is to unzip the csv file in the root folder of the repo.

[Overleaf report](https://fr.overleaf.com/read/jznbtvznsrfb)

## :package: Structure

The first data exploration and visualization can be found in the first notebook: [data_exploration_and_visualization.ipynb](https://github.com/ArianeDlns/ML-Drinking_Water_Potability/blob/main/Data_exploration_and_visualization.ipynb)

The modeling part can be found in the second notebook: [pipeline.ipynb](https://github.com/ArianeDlns/ML-Drinking_Water_Potability/blob/main/Pipeline.ipynb)

## :trophy: Winning Algorithm

After our research on which model is the most relevant for this problem, we selected 3 different well-performing algorithms (K-NN, SVM and Random Forest). Each one was then fine-tuned and work collaboratively in a voting classifier.

## References

[1] Lundberg, Scott M and Lee, Su-In, [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf), _Advances in Neural Information Processing Systems 30_ (2017)  

[2] Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani, [An introduction to stastical learning](https://centralesupelec.edunao.com/pluginfile.php/171459/course/section/30032/2013_Book_AnIntroductionToStatisticalLea.pdf), _Springer_  

[3] Friedman, Jerome H., The elements of statistical learning: Data mining, inference, and prediction, _Springer_ 