# Todd Bartoszkiewicz
# CSC510: Foundations of Artificial Intelligence
# Module 6: Critical Thinking
#
# Naive Bayes classifiers are quick and easy to code in Python and are very efficient.
#
# Naive Bayes classifiers are based on Bayes' Theorem and assume independence among predictors (hence the "Naive"
# terminology). Not only are Naive Bayes classifiers handy and straightforward in a pinch, but they also outperform
# many other methods without the need for advanced feature engineering of the data.
#
# Check out the following for further information on Naive Bayes classification:
# https://www.ibm.com/topics/naive-bayes
#
# Using scikit-learn, write a Naive Bayes classifier in Python. It can be single or multiple features. Submit the
# classifier in the form of an executable Python script alongside basic instructions for testing.
#
# Your Naive Bayes classification script should allow you to do the following:
#
# Calculate the posterior probability by converting the dataset into a frequency table.
# Create a "Likelihood" table by finding relevant probabilities.
# Calculate the posterior probability for each class.
# Correct Zero Probability errors using Laplacian correction.
# Your classifier may use a Gaussian, Multinomial, or Bernoulli model, depending on your chosen function. Your
# classifier must properly display its probability prediction based on its input data.
#
# Check out scikit-learn and its documentation at the following website:
#
# scikit-learn:
# https://scikit-learn.org/stable/
#
# For this Critical Thinking assignment, we'll train a classifier on a small dataset of 'spam' vs 'ham'.
# 'Spam' is unsolicited and often unwanted e-mail messages that include promotional messages, phishing scams, and
# malicious links. 'Ham' is legitimate emails from trusted sources.
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


"""
Trains a Multinomial Naive Bayes classifier.

1. Create frequency table with bag-of-words (CountVectorizer)
2. Calculate likelihood using Multinomial Naive Bayes
3. Use alpha=1.0 in Multinomial Naive Bayes for Laplacian correction to prevent 0 probability for unseen events.
"""
if __name__ == "__main__":
    print("Training Multinomial Naive Bayes classifier")

    # Training Data using e-mails from my spam folder
    data = {
        'text': [
            'Debt Relief: Feeling the Inflation Pinch? Let Us Help',
            'CreditCardBonus Offer: What Is A High Paying Rewards Credit Card These Days',
            'Payment-Declined: [username] , Your Account Has been Blocked! Your Photos and Videos will be Removed.',
            'noreply-MDOS-Services@mail.mighigan.gov: Reminder your appointment is 10:30 AM on Friday, September 12',
            'ArbiterSports: urgent need Thursday',
            'Wells Fargo: We received your deposit',
            'Type2Defense: 1/2 teaspoon KILLS high blood sugar (diabetes) permanently',
            'CVS Caremark: Your prescription is ready',
            'JGWentworth: RE:Your_Credit-card Bill',
            'Mr.Bernard Arnault: you have been gifted $15 MILLION USD in 2025 Donation Funds',
            'HealthCare.com Obamacare: See which Obamacare plans are a good fit for you',
            'Empire Today Partner: Upgrade Your Floors for Up to 50% Off!',
            'Hertz Gold Plus Rewards: Todd, Get 2 FREE days this season',
            'USPS Informed Delivery: USPS Expected Delivery by Monday, September 22, 2025',
            'Freecash.com: Win cash prize free money'
        ],
        'label': [
            'Spam', 'Spam', 'Spam', 'Ham', 'Ham', 'Ham', 'Spam', 'Ham', 'Spam', 'Spam', 'Spam', 'Spam', 'Ham', 'Ham',
            'Spam'
        ]
    }

    df = pd.DataFrame(data)
    x_text = df['text']
    y_label = df['label']

    # Create the frequency table using the CountVectorizer to convert the text to a matrix of token counts
    # MultinomialNB with default alpha=1.0 for Laplace smoothing, calculates probabilities.
    count_vectorizer = CountVectorizer()
    X = count_vectorizer.fit_transform(x_text)
    feature_names = count_vectorizer.get_feature_names_out()

    model = MultinomialNB(alpha=1.0)
    # model.fit(x_text, y_label)

    pipeline = make_pipeline(
        count_vectorizer,
        model
    )

    # Train the model
    pipeline.fit(x_text, y_label)

    print("\n--- Frequency Table ---")
    counts = model.feature_count_.astype(int)
    classes = model.classes_
    for class_idx, class_name in enumerate(classes):
        row = dict(zip(feature_names, counts[class_idx]))
        print(f"{class_name}: {row}")

    print("\n--- Likelihood Table with Laplance smoothing ---")
    likelihoods = np.exp(model.feature_log_prob_)
    for class_idx, class_name in enumerate(classes):
        row = dict(zip(feature_names, likelihoods[class_idx].round(4)))
        print(f"{class_name}: {row}")

    print("Training complete")

    # Get input from user
    input_text = input("Enter text to classify:").strip()

    print(f"\n--- Analysis for: '{input_text}' ---")

    # Calculate Posterior Probabilities
    probabilities = pipeline.predict_proba([input_text])[0]
    predicted_class = pipeline.predict([input_text])[0]

    # Display results
    print("\nPosterior Probabilities:")
    for i, class_label in enumerate(pipeline.classes_):
        print(f"  {class_label}: {probabilities[i]:.4f}")

    print(f"\nPrediction: The message is classified as: {predicted_class}")
