This project uses Naive Bayse Classification algorithm to predict sentiment. This is a very basic ML project for beginners. 
In this I have used only 2 classifications namely GOOD(represented by 1) and BAD(represented by 0). 
Any user can enter a text(sentence) and this algorithm will classify it into either class 1 or class 0.
Firstly I have used NLTK stopwords ( So that all the unnecessary words like is,that,this etc, which don't contribute to any kind of sentiment are removed and the remaining sentence only contains keywords which contribute to some sentiment).
Then we calculate the conditional probabilities of each keyword firstly assuming the sentence to belong to class 0 and then class 1. then we multiply all the conditional probabilities.Then the product is multiplied by the probability that the sentence belongs to that class respectively.
If the product is greater in case of class 0 then the sentence is classified as class 0 oterwise, it is classified as class 1.  
