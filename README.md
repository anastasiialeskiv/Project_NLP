![How-to-Get-Amazon-Reviews-the-Right-Way](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/dfed82bc-e1e3-4ba4-b823-85c6f7c55f9c)

Anastasiia Leskiv

# Business Understanding

Everyday in our life we are buying different products, we have a huge selection under one category. It's always hard to decide what brand or type of the product to choose. Here where reviews help us to make that dessision.Customers who have already purchased a product share their experiences by providing ratings and detailed reviews.I analyzed the reviews provided by customers. I used machine learning and deep learning models to create a predictive model. Those models will determine whether customers are satisfied or dissatisfied with the product based on their reviews. My model would help manufacturers to make their products better by incorporating customer feedback and suggestions.

# Data Understanding

This dataset containing Amazon Product Data includes product categories and various metadata. The product with the most comments in the electronics category has user ratings and comments.

###### Tools:

Numpy & Pandas for data processing

Scattertext for finding distinguishing terms

Steamlit for Classification app deployment

Matplotlib, WordCloud for visualization

Scikit-learn for machine learning

NLTK for natural language processing

###### Features:

Unnamed - Index

reviewerName - User Name

overall - Product Rating

eviewText - Evaluation Summary

reviewTime - Evaluation Time {RAW}

day_diff - Number of days since assessment

helpful_yes - The number of times the evaluation was found useful

helpful_no - Number of people who didn't support the comment and didn't find it helpful

total_vote - Number of votes given to the evaluation

score_pos_neg_diff - score poz-neg

score_average_rating

wilson_lower_bound

# Exploratory Data Analysis

<img width="937" alt="Screenshot 2023-10-25 at 5 54 21 PM" src="https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/e8b6c37b-e783-4db2-ac4b-58f962a0dbdf">

In this visualization I want to show how many reviews in each category from 1 to 5 stars review

5 stars review  -  3921

4 stars review  -  526

3 stars review  -  142

2 stars review  -  80

1 stars review  -  244


![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/0d0487e1-3f3d-4e2f-86d2-fedfdede56a3)

Each square shows the correlation between the variables on each axis. Correlation ranges from -1 to +1. Values closer to zero means there is no linear trend between the two variables. The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other and the closer to 1 the stronger this relationship is. A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases. The diagonals are all 1/dark because those squares are correlating each variable to itself (so it's a perfect correlation). For the rest the larger the number and darker the color the higher the correlation between the two variables. In this particular case we can see that Total vote and Helpful yes and no have extremely high correlation.

![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/c97a6d12-0541-4dc9-8000-d82d3e7fc234)

Here in this visualization we can see the most used words in this dataset

![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/63b49be6-2212-4a01-b387-e7b04381f0e0)

Here in this visualization we can see the most common words in positive reviews

![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/a895dc3b-3061-400f-81c0-942cf2cd4f8f)

Here in this visualization we can see the most common words in negative reviews

![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/03877cf5-1f82-43ab-a1fe-1ce7912416d3)

Here in this visualization we can see the most common words in good (3stars) reviews



I also decided to check what the common words for each section are: “Positive”,”Good”, and “Negative”.

<img width="545" alt="Screenshot 2023-10-25 at 5 59 26 PM" src="https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/7e5e8509-0860-4dd7-a44e-371413f3bdc1">


It's hard to see what the review is about in only one word. I'll try to use 2 word

<img width="584" alt="Screenshot 2023-10-25 at 6 00 16 PM" src="https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/5412f130-e5a4-436a-91ad-a324e2bced11">

Here I can get a better idea for example "great price" was used 77 and that is what makes this product good versus "stopped working" here we can see that 10 reviews were negative about this product because of that and we can come to the conclusion how to make this product better.But let's also check 3 words

<img width="750" alt="Screenshot 2023-10-25 at 6 01 08 PM" src="https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/6b7bc36a-869c-4abf-8e14-6b1f9a6964b4">

Now I can get ever better idea of what product is rewiev about or why the product is good of bad.


# Modeling

![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/4525c60f-d0fe-432a-ac87-c8b7e7a15329)

See the diagonal elements (893, 906, 854), we observe that they represent correctly predicted records, while the rest correspond to instances that were incorrectly classified by the algorithm. This highlights the importance of evaluating not only accuracy but also other metrics like precision, recall, and the F1 score to gain a comprehensive understanding of the model's performance.

# Evaluation

![image](https://github.com/anastasiialeskiv/Movies_Exploratory_Data_Analysis/assets/124845922/5e93357a-af5b-414b-97ae-12b9ee4e1124)

ROC-AUC curve
The Receiver Operating Characteristic (ROC) curve is indeed a critical tool, especially when dealing with classification tasks involving multiple classes. It allows us to visualize and assess the model's performance across different classes and provides valuable insights into class-specific classification. By plotting ROC curves for each class, we can determine how well the model discriminates between the different sentiment categories.

The micro and macro averages on the ROC curve are also essential. The micro-average considers all instances individually, regardless of their class labels. It's useful when all classes are of equal importance. The macro-average, on the other hand, calculates the average performance across all classes. This gives you a sense of overall model performance, which is crucial when different classes have varying degrees of importance.

These evaluations and visualizations are fundamental for making informed decisions about the model's performance and tuning its behavior to meet specific objectives.

It's great to see that class 2 and class 1 have been classified effectively, as indicated by their high area under the curve (AUC) values. This suggests that the model performs well in distinguishing these classes.

The ROC curve provides a range of threshold values that can be chosen to balance true positive rate (TPR) and false positive rate (FPR) based on the specific requirements of the task.

Micro and Macro Averages:

Micro-average is performing well, indicating an aggregate measure that considers all classes and is beneficial when class imbalance is suspected.

Macro-average computes the metric independently for each class and then takes the average, treating all classes equally. In this case, it appears to have a lower score, suggesting that the model's performance may vary across different classes.

These insights provide a clear understanding of how well the model is performing for different classes and the overall classification task. They help in making informed decisions about the choice of thresholds and understanding the impact of class imbalance on the evaluation metrics.

# Conclusion 

N-gram Consideration:

Incorporating n-grams in sentiment analysis is a valuable approach, as it allows the model to capture the meaning of phrases rather than relying solely on individual words.

Stopwords Handling:

Manually checking and customizing the stopwords list based on the specific requirements of the sentiment analysis task is crucial. In this case, avoiding certain stopwords improved the results.

Feedback from Good Reviews:

Recognizing that many good reviews were actually criticism or feedback from buyers is a valuable insight. This feedback can be shared with sellers to help them improve their products. Most of the reviews were on electronics such as phones, samsung galaxy, GoPro, cards ets.

Balancing the Dataset:

Achieving a balanced dataset through techniques like SMOTE proved to be beneficial, leading to improved accuracy, precision, recall, and F1 score. Balancing is particularly important for maintaining a fair evaluation of the model's performance.

Emphasis on F1 Score:

Concentrating on the F1 score in sentiment analysis is highlighted, and achieving an average of 99% demonstrates the effectiveness of the model in both precision and recall. These insights reflect a thoughtful and thorough approach to sentiment analysis, addressing various challenges and considerations in the process.

## Limitation

Sentiment analysis using sentiment word dictionaries has low reliability when the number of positive and negative words is small. For example, if there are 0 positive words and 1 negative word, it is classified as negative. Therefore, if the number of sentiment words is 5 or less, we could exclude the observations.

## Recommendations

My recommendation is to pay attention to good reviews(3 stars reviews) . These are the most helpful and will help manufacturers to make their products better by incorporating customer feedback and suggestions.

## Next Steps


My next step would be to explore 3 stars reviews more to have specific recommendations to my client

├── .gitignore

├── .zip.zip

├── README.md

├── amazon_reviews.csv

└── notebook.ipynb
