# import libraries
import sys
import pandas as pd
import sqlite3
import re
from io import StringIO
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import multioutput
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')


def classification_report_df(report,col):
    '''
    This method takes the classifcation report and converts it into dataframe 
    by iterating over it line wise and the split and extract the required features like class, precision,recall,f1_score

    Args:
    report: classificaton report.
    col: column for which report exists

    Returns:
    report dataframe col wise
    '''

    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['col'] = col
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

def get_classification_report(y_test,y_pred):
    '''
    Reports the f1 score, precision and recall for each output category of the dataset. 
    by iterating through the columns and calling sklearn's classification_report on each.

    Args:
    y_test: the train  data.
    y_pred: the train prediction data

    Returns:
    classification report for each column
    '''
    report = pd.DataFrame()

    for col in y_test.columns:
        #returning dictionary from classification report
        class_report = classification_report(y_true = y_test.loc [:,col], y_pred = y_pred.loc [:,col])
        class_report_df = classification_report_df(class_report,col)
        av_eval_df = pd.DataFrame (class_report_df)
        mean_df=av_eval_df.drop('class',axis=1).groupby('col', as_index=False).mean().drop('col',axis=1)

        #appending result to report df
        report = report.append(mean_df)   
    #renaming indexes for convinience
    report.index = y_test.columns
    return report

def load_data(database_filepath):
    '''
    Loads data from sqllite database and extract the test and train and categories

    Args:
    database_filepath: the database from where data can be retrieved

    Returns:
    train data,test data , category_names
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql ('SELECT * FROM message_categories', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names=y.columns
    return X,y,category_names

def tokenize(text):
    '''
    Tokenize data using word tokenizer , lemmatizer and coverted to lower case

    Args:
    text: string to be tokenized

    Returns:
    cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Build GridSearch Model using Pipeline with steps
        1. CountVectorizer
        2. TfIdf Transformer
        3. MultiOutputClassifier

    Parameters chosen for best performing model

    Returns:
    ml model
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('mclf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = { 'vect__max_df': (0.75, 1.0),
                'mclf__estimator__n_estimators': [10, 20],
                'mclf__estimator__min_samples_split': [2, 5]
              }

    model = GridSearchCV (pipeline, param_grid= parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This method evaluates the model by generating mean value of f1_score,precisoin,recall for each column

    Args:
    model: ml model 
    X_test: input test data
    Y_test: output test data
    category_names: category names

    Returns:
    None
    '''
    y_pred = model.predict(X_test)
    #converting to a dataframe
    y_pred= pd.DataFrame(y_pred, columns = Y_test.columns)
    report_tuned = get_classification_report (Y_test, y_pred)
    print(report_tuned.mean())


def save_model(model, model_filepath):
    '''
    Saves model in the working dir

    Args:
    model: ml model 
    model_filepath: file path  where to store model

    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
