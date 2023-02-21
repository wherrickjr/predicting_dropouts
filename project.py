import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing




def acquire_grads():
   df = pd.read_csv('drop_or_grad.csv')
   return df




def prep_grads(drop):
    drop['Marital status'] = drop['Marital status'].replace(1, 'single')
    drop['Marital status'] = drop['Marital status'].replace(2, 'married')
    drop['Marital status'] = drop['Marital status'].replace(3, 'widower')
    drop['Marital status'] = drop['Marital status'].replace(4, 'divorced')
    drop['Marital status'] = drop['Marital status'].replace(5, 'married')
    drop['Marital status'] = drop['Marital status'].replace(6, 'divorced')
    drop['Nacionality'] = drop['Nacionality'].replace(1, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(2, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(3, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(4, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(5, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(6, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(7, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(8, 'african')
    drop['Nacionality'] = drop['Nacionality'].replace(9, 'african')
    drop['Nacionality'] = drop['Nacionality'].replace(10, 'african')
    drop['Nacionality'] = drop['Nacionality'].replace(11, 'african')
    drop['Nacionality'] = drop['Nacionality'].replace(12, 'african')
    drop['Nacionality'] = drop['Nacionality'].replace(13, 'middle-eastern')
    drop['Nacionality'] = drop['Nacionality'].replace(14, 'latino')
    drop['Nacionality'] = drop['Nacionality'].replace(15, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(16, 'european')
    drop['Nacionality'] = drop['Nacionality'].replace(17, 'latino')
    drop['Nacionality'] = drop['Nacionality'].replace(18, 'asian')
    drop['Nacionality'] = drop['Nacionality'].replace(19, 'asian')
    drop['Nacionality'] = drop['Nacionality'].replace(20, 'latino')
    drop['Nacionality'] = drop['Nacionality'].replace(21, 'latino')
    drop['Previous qualification'] = drop['Previous qualification'].replace(1, 'high school only')
    drop['Previous qualification'] = drop['Previous qualification'].replace(2, 'bachelor\'s degree')
    drop['Previous qualification'] = drop['Previous qualification'].replace(3, 'bachelor\'s degree')
    drop['Previous qualification'] = drop['Previous qualification'].replace(4, 'post grad degree')
    drop['Previous qualification'] = drop['Previous qualification'].replace(5, 'post grad degree')
    drop['Previous qualification'] = drop['Previous qualification'].replace(6, 'some college')
    drop['Previous qualification'] = drop['Previous qualification'].replace(7, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(8, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(9, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(10, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(11, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(12, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(13, 'no high school')
    drop['Previous qualification'] = drop['Previous qualification'].replace(14, 'some college')
    drop['Previous qualification'] = drop['Previous qualification'].replace(15, 'bachelor\'s degree')
    drop['Previous qualification'] = drop['Previous qualification'].replace(16, 'some college')
    drop['Previous qualification'] = drop['Previous qualification'].replace(17, 'post grad degree')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(1,'high school only')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(2,'bachelor\'s degree')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(3,'bachelor\'s degree')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(4,'post grad degree')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(5,'post grad degree')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(6,'some college')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(7,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(8,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(9,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(10,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(11,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(12,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(13,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(14,'high school only')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(15,'high school only')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(16,'some college')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(17,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(18,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(19,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(20,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(21,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(22,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(23,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(24,'unknown')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(25,'illiterate')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(26,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(27,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(28,'no high school')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(29,'some college')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(30,'some college')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(31,'some college')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(32,'some college')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(33,'post grad degree')
    drop['Mother\'s qualification'] = drop['Mother\'s qualification'].replace(34,'post grad degree')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(1,'high school only')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(2,'bachelor\'s degree')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(3,'bachelor\'s degree')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(4,'post grad degree')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(5,'post grad degree')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(6,'some college')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(7,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(8,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(9,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(10,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(11,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(12,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(13,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(14,'high school only')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(15,'high school only')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(16,'some college')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(17,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(18,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(19,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(20,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(21,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(22,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(23,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(24,'unknown')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(25,'illiterate')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(26,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(27,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(28,'no high school')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(29,'some college')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(30,'some college')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(31,'some college')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(32,'some college')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(33,'post grad degree')
    drop['Father\'s qualification'] = drop['Father\'s qualification'].replace(34,'post grad degree')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(1,'education')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(2,'law')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(3,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(4,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(5,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(6,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(7,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(8,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(9, 'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(10,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(11,'military')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(12,'none')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(13,'military')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(14,'military')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(15,'military')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(16,'military')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(17,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(18,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(19,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(20,'health')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(21,'education')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(22,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(23,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(24,'unknown')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(25,'health')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(26,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(27,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(28,'STEM')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(29,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(30,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(31,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(32,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(33,'law')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(34,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(35,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(36,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(37,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(38,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(39,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(40,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(41,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(42,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(43,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(44,'trade')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(45,'service')
    drop['Mother\'s occupation'] = drop['Mother\'s occupation'].replace(46,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(1,'education')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(2,'law')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(3,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(4,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(5,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(6,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(7,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(8,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(9,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(10,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(11,'military')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(12,'none')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(13,'military')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(14,'military')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(15,'military')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(16,'military')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(17,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(18,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(19,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(20,'health')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(21,'education')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(22,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(23,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(24,'unknown')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(26,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(27,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(28,'STEM')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(29,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(30,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(31,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(32,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(33,'law')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(34,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(35,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(36,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(37,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(38,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(39,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(40,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(41,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(42,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(43,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(44,'trade')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(45,'service')
    drop['Father\'s occupation'] = drop['Father\'s occupation'].replace(46,'service')

    drop = drop.drop(drop.columns[[19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ,30]], axis = 1)
    drop['Course'] = drop['Course'].replace(1, 'agriculture')
    drop['Course'] = drop['Course'].replace(2, 'Animation')
    drop['Course'] = drop['Course'].replace(3, 'social service')
    drop['Course'] = drop['Course'].replace(4, 'agriculture')
    drop['Course'] = drop['Course'].replace(5, 'communication')
    drop['Course'] = drop['Course'].replace(6, 'nursing')
    drop['Course'] = drop['Course'].replace(7, 'engineering')
    drop['Course'] = drop['Course'].replace(8, 'agriculture')
    drop['Course'] = drop['Course'].replace(9, 'management')
    drop['Course'] = drop['Course'].replace(10, 'social service')
    drop['Course'] = drop['Course'].replace(11, 'tourism')
    drop['Course'] = drop['Course'].replace(12, 'nursing')
    drop['Course'] = drop['Course'].replace(13, 'oral hygiene')
    drop['Course'] = drop['Course'].replace(14, 'marketing')
    drop['Course'] = drop['Course'].replace(15, 'communication')
    drop['Course'] = drop['Course'].replace(16, 'education')
    drop['Course'] = drop['Course'].replace(17, 'management')
    drop = drop.drop(drop.columns[[1,2,11,13,18,19,20,21]], axis = 1)
    drop = drop[drop['Target'] != 'Enrolled']
    drop['Marital status'] = drop['Marital status'].replace('facto union', 'married')
    drop['Marital status'] = drop['Marital status'].replace('legally separated', 'divorced')

    return drop


def count_plots(df, x, y):
    ''' this function takes in a df and two columns and returns
    a seaborn countplot of percentages without a title
    '''
    df1 = df.groupby(x)[y].value_counts(normalize=True)
    df1 = df1.mul(100)
    df1 = df1.rename('percent').reset_index()

    sns.set(font_scale = 1)
    g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)

    g.ax.set_ylim(0,100)


    for p in g.ax.patches:
        txt = str(p.get_height().round(2)) + '%'
        txt_x = p.get_x() 
        txt_y = p.get_height()
    
    return g


def dummies(df):
    dummy_df = pd.get_dummies(df[['Marital status', 'Course', 'Previous qualification', \
                               'Nacionality', 'Mother\'s qualification', 'Father\'s qualification', \
                               'Mother\'s occupation', 'Father\'s occupation']], dummy_na=False, drop_first=[True])
    df2 = pd.concat([df, dummy_df], axis = 1)
    just_dummies = df2.drop(columns = ['Marital status', 'Course', 'Previous qualification', \
                                   'Nacionality', 'Mother\'s qualification', 'Father\'s qualification', \
                                   'Mother\'s occupation', 'Father\'s occupation',])
    train, test = train_test_split(just_dummies, test_size=.2, random_state=123, stratify=df.Target)
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.Target)
    return train, test, validate


def model_tests(train, validate):
    x_train = train.drop(columns = ['Target'])
    y_train = train['Target']
    x_val = validate.drop(columns = ['Target'])
    y_val = validate['Target']
    
    #scaling data
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    seed = 42
    clf = DecisionTreeClassifier(max_depth = 6, random_state = 42)
    clf.fit(x_train, y_train)
    accuracy = pd.DataFrame({'model' : ['decision tree'], 'baseline': [2209 / (2209 + 1421)], 'train accuracy' : \
             [clf.score(x_train, y_train)], 'validate accuracy' : [clf.score(x_val, y_val)]})
    knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')
    knn.fit(x_train, y_train)
    knndf = pd.DataFrame({'model' : ['KNN'], 'baseline': [2209 / (2209 + 1421)], 'train accuracy' : \
             [knn.score(x_train, y_train)], 'validate accuracy' : [knn.score(x_val, y_val)]})
    accuracy = pd.concat([knndf, accuracy])
    
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=5,
                            n_estimators=100,
                            max_depth=11, 
                            random_state=123)
    rf.fit(x_train, y_train)
    randomforestdf = pd.DataFrame({'model' : ['random forest'], 'baseline': [2209 / (2209 + 1421)], 'train accuracy' : \
             [rf.score(x_train, y_train)], 'validate accuracy' : [rf.score(x_val, y_val)]})
    accuracy = pd.concat([accuracy, randomforestdf])
    return accuracy

def testdf(df):
    x_train = df.drop(columns = ['Target'])
    y_train = df['Target']
    #scaling data
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=5,
                            n_estimators=100,
                            max_depth=11, 
                            random_state=123)
    rf.fit(x_train, y_train)
    return rf.score(x_train, y_train)