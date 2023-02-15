# University graduation/dropout Predictions
 
# Project Description

Not everyone goes to college and not everyone who goes finished either. An observational study was conducted on university students in europe to gather data about students who graduated and/or dropped out. The purpose of this project is to analyze this data and come up with a machine learning model that can predict whether or not a student with graduate.

# Project Goal
 
* Discover features that have the strongest influence on graduation.

* Use features to develop a machine learning model to predict graduation.

* This information can be used to identify students who are at risk of dropping out and providing them with additional resources/services.
 
# Initial Thoughts
 
My initial informal hypothesis is that nationality will have no influence, age will have no influence, and parent education level with have an influence on graduation.
 
# The Plan
 
* Aquire data from kaggle.com

   * Create Engineered columns from existing data:

       * marital status
       * course
       * daytime/evening classes
       * previous qualification
       * nationality
       * mother's qualification
       * father's qualification
       * mother's occupation
       * father's occupation
       * educational special needs
       * tuition fees up to data
       * gender
       * scholarship holder
       * age at enrollment
       * target
 
* Explore data in search of drivers of graduation

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|marital status| martial status of student|
|course| field of study of student|
|Daytime/evening attendence| 1 or 0, 1 meaning daytime classes and 0 evening classes|
|Previous qualifications| previous educational achievement of student|
|Nacionality| Nationality of student|
|Mother's qualifications| education level of student's mother|
|Father's qualification| education level of student's father|
|Mother's occupation| occupation of student's mother|
|Father's occupation| occupation of student's father|
|Education special needs| 1 or 0, 1 meaning student needs special education modifications or accomodations 0 meaning none|
|Tuition fees up to date| 1 or 0, 1 meaning fees are up to date, 0 meaning student has outstanding balance|
|Gender| 1 or 0, 1 meaning man, 0 meaning woman|
|Scholarship holder| 1 or 0, 1 meaning student has scholarship, 0 meaning none|
|Age at enrollment| age in years of student when they enrolled|
|Target| indicates whether or not student graduated or dropped out|

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from kaggle
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* There is a relationship between student's marital status and graduation
* A relationship exists between age at enrollment and graduation. Younger student's are more likely to graduate
* There is no signifigant relationship between nationality and whether or not a student graduates
* Father's education level seems to have a relationship with graduation
* Graduation and mother's occupation have a relatively weak relationship
 
# Recommendations
* Provide extra support to first generation college students
* Provide extra support to students who are married
* Encourage younger people to apply and take courses to increase graduation rate
