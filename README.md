# yelp-is-all-you-need
## Big Data Mining and Management Projects
Leveraged a subset of the Yelp dataset to devise Deep Learning Algorithms for Sentiment Classification, Link Prediction and to build a recommender system.

### [Project 1: Sentiment Classification](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project1/35/RMBI4310_COMP4332%20Project%201%20Report.pdf)
This project aims to solve the multiclass sentiment classification problem on a subset of the Yelp dataset. The data consists of reviews as well as attributes such as ‘funny’, ‘cool’ and useful and the stars for each review that range from 1-5 which serve as the labels. Two different approaches for models are compared - 1) Heavyweight feature engineering-based ensemble model and 2) Contextualized word representation (BERT) based model.

- Ensemble | [model](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project1/35/Ensemble%20Model%2C%20TF-IDF%20%2B%20Glove%20%2B%20CNN%20%2B%20RNN.ipynb) | [motivation](https://www.kaggle.com/ajithvajrala/word-embeddings-with-tfidf-ensemble)
- BERT | [model](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project1/35/4332_P1_BERT_Finalised_Model_%2B_Test_set_predictions.ipynb) | [motivation](https://arxiv.org/abs/1810.04805)

### [Project 2: Link Prediction](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project2/35/RMBI4310_COMP4332%20Project%202%20Report.pdf)
This project aims to solve the link prediction problem on a subset of the Yelp dataset. The data consists of user_id and friends which correspond to a directed graph, whose edges serve as the labels. Two different random walk based embedding algorithms are compared - 1) DeepWalk and 2) node2vec.

- DeepWalk | [model](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project2/35/Project_2_Social_Network_Mining_Colab_revised.ipynb) | [motivation](https://arxiv.org/abs/1403.6652)
- Node2Vec | [model](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project2/35/Project_2_Social_Network_Mining_Colab_revised.ipynb) | [motivation](https://arxiv.org/abs/1607.00653)

### [Project 3: Recommender System](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project3/35/COMP4332_RMBI4310%20Project%203%20Report.pdf)
This project aims to solve the rating prediction problem on a subset of the Yelp dataset. The data consists of users and businesses, with a rating that corresponds to the rating a user has given the respective business. In addition, various individual user attributes as well as business attributes are available to supplement these ratings. The Wide and Deep Model (WDL) has been used since it performed better than the Neural Collaborative Filtering Model (NCF) during our analysis, with a validation RMSE (Root Mean Squared Error) of 0.9996 from the former compared to RMSE of 1.0533 from the latter model.

- NCF | [model](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project3/35/Project_3_Rating_Prediction_NCF.ipynb) | [motivation](https://arxiv.org/abs/1708.05031)
- WDL | [model](https://github.com/nikhilnanda21/yelp-is-all-you-need/blob/master/Project3/35/Project_3_Rating_Prediction_Wide_and_Deep.ipynb) | [motivation](https://arxiv.org/abs/1606.07792)

## Usage
All notebooks can be downloaded and run on [Google Colab](https://colab.research.google.com/)

## Acknowledgements
Grateful to the Big Data Mining and Management faculty
