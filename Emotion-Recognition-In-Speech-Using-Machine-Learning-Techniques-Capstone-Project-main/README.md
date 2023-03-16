# Capstone Project: Emotion Recognition in Speech Using Machine Learning Using Machine Learning Techniques

This repository contains code for our Capstone Project titled 'Emotion Recogntion In Speech Using Machine Learning Techniques. It includes a folder of pre-trained models, and also some test clips to test out the prediction of the code. The test clips have their features extracted into a csv which is then uploaded to the demo ipynb file. The pre-trained models are also uploaded to the ipynb instance and loaded using joblib, after which one can make predictions, and view the classification report and confusion matrices to compare all the models. 

## Emotion labels considered: 
Anger, Happiness, Fear, Neutrality, Sadness, Sarcasm

## Implementation

The steps to implement them are as follows:

1. Download this repository
2. Unzip 
3. At the unzipped folder, open a terminal window
4. Create the CSV of the test clip features (code given below)
5. Upload the IPYNB to Google Colab
6. Run the code and upload the models and Test csv in the necessary cells
7. Run the rest of the cells

Code to create CSV:

In the terminal at the unzipped repository folder, run:

python test_clip_csv_generation.py


This should create a CSV called Test_Clips_1.csv in the same folder. This has to be uploaded to the demo ipynb on colab (See step 6)

