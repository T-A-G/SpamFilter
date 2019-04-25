# Spam-Filter

Simple spam filter using a Multinomial Naive Bayes. Implemented using Python 3.6 using [Scikit Learn](https://scikit-learn.org/stable/) with [Numpy](https://www.numpy.org/), [Pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/)

# Data  

Uses small dataset of 10,000 emails. The data contains the following fields 

| Field  | Type | Details |
| ------ | ---- | ------- |
| text   | ```String```  |The text of the email|
| has_link  | ```Bool```  |Whether or not the text contains one or more links |
| has_image  | ```Bool```  |Whether or not the text contains one or more images |
| label  | ```String```  |Classification of text ("ham" for valid emails; "spam" for spam emails)|
