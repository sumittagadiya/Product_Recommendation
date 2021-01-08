# Product_Recommendation
This is product recommendation project like amazon or flipkart

# 1.Description
* In this project i had csv file of cloths product. From csv i made sqllite database and made flask app in which top 20 product in descending order of price will be display and when user clicks on any product then top 10 similar products will appear at bottom of the page.


* There are total 7 columns in database with name asin(unique_id),brand,color,medium_image_url,product_type_name,title,formatted_price.
# 2.My approach:
  1. First of all i had data in csv file format i.
  2. I have done little preprocessing on csv file which is in **core_operations.ipynb**
  3. After that i created database in sqllite and made table to store data from csv file
  4. Load data from csv to **sqllite database**
  5. made routes in **app.py** file to get top 20 products based on descending order of price
  6. made route for most similar products when user clicks on any product it will redirect to the page with detail of the selected product.
  7. To get top 10 products first of all i preprocessed title column of csv file and removed stopwords and punctuations.
  8. After preprocessing i have used **TFIDF + Glove 840B 300d** to vectorize data.
  9. All necessary files dumped in pickle_files folder.
  10. Made function named **find_similar** to get top k most similar products based on text description.
  11. Cosine similarity has been used to get top k products.
  12. All the similarity related code is in **ml_model.py** file.
  13. Above mentioned steps i have followed to made this app work.
  14. Deployed flask app on **heroku**.
  
# 3. Deployment on Heroku
* You can find deplyoed app on Heroku [here](https://product--recommendation.herokuapp.com/)
  
## NOTE
* All the operations has been mentioned in **core_operations.ipynb**

