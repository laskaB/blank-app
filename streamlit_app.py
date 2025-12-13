import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Project Framingham Heart Disease - Group 7")
st.write(
    "The Framingham Heart Study is a long-term study of cardiovascular disease that identified risks factors and their join effects." \
    " There are 4434 patients, each of them with 3 examination periods every 6 years. The outcome is: angina pectoris, myocardial infraction, Atherothrombotic Infarction or Cerebral Hemorrhage (Stroke) or death."
)

#data loading
cvd = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

#first look at the data
st.dataframe(cvd.head())

#observations about the data
st.write(
    "The dataset contains 39 features inclding: demographic informations, clinical measurements, physiological risks factors, lifestyle, medications, baseline disease status, and follow up outcomes." 

)

#formulation of the research question
st.title(
    'Can we predict wheater a patient dies based on his baseline health indicators?'
)

st.markdown(
    """
    To answer this research question, we created a subset of the original dataframe:

    - only patients from period 1 were selected
    - all the variables related to time were dropped 
    - HDL and LDL cholesterol were dropped as they were not available for period 1


    """
)

#we performed metadata enrichment to add more context to the data and its visualization
#metadata enrichment
cvd['SEX'] = cvd['SEX'].replace({2: 'female', 1:'male'})
cvd['CURSMOKE'] = cvd['CURSMOKE'].replace({0: 'not current smoker', 1:'current smoker'})
cvd['DIABETES'] = cvd['DIABETES'].replace({0: 'not a diabetic', 1:'diabetic'})
cvd['BPMEDS'] = cvd['BPMEDS'].replace({0: 'not currently used', 1:'current use'})
cvd['PREVAP'] = cvd['PREVAP'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVSTRK'] = cvd['PREVSTRK'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVMI'] = cvd['PREVMI'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVCHD'] = cvd['PREVCHD'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PREVHYP'] = cvd['PREVHYP'].replace({0: 'free of disease', 1:'prevalent disease'})
cvd['PERIOD'] = cvd['PERIOD'].replace({1: 'period 1', 2:'period 2', 3:'period 3'})
cvd['DEATH'] = cvd['DEATH'].replace({0: 'survived', 1:'died'})
cvd['ANGINA'] = cvd['ANGINA'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['HOSPMI'] = cvd['HOSPMI'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['MI_FCHD'] = cvd['MI_FCHD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['ANYCHD'] = cvd['ANYCHD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['STROKE'] = cvd['STROKE'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['CVD'] = cvd['CVD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['HYPERTEN'] = cvd['HYPERTEN'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
cvd['educ'] = cvd['educ'].replace({1: 'higher secondary', 2:'graduation', 3:'post graduation', 4: 'PhD'})

#subset creation
cvd_death = cvd.loc[(cvd['PERIOD'] == 'period 1')]
cvd_death = cvd_death.drop(columns = ['TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD', 'TIMEHYP', 'HDLC', 'LDLC', 'TIME'])
cvd_death['DEATH'] = cvd_death['DEATH'].replace({0: 'survived', 1:'died'})


#final subest 
st.write("The final subset contains the following number of rows and columns (patients and features respectively):")
cvd_death.shape

#displaying the columns of the final subset
st.write("The final subset contains the following features:")
cvd_death.columns

#statistics of numerical variables
st.write("Here are some statistics of the numerical variables in the final subset:")
st.dataframe(cvd_death.describe())

#addressing missing values
st.write(
    """
    *Missing values handling*

    There are few missing values present in this subset:  
    between *0.02% and 9%* across *7 attributes*.

    It was decided to impute the missing values *after the train/test split*, as common practice.

    *Imputation strategy:*
    - Numerical variables were imputed using *KNN imputation (k = 5)*
    - Categorical variables were imputed using the *mode (most frequent value)*
    """
)
#calculating missing values percentage
missing_percentage = cvd_death.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]

#visual rapresentation of the missing data
fig, ax = plt.subplots()
ax.set(title="Missing data", xlabel="Percent missing", ylabel="Variable", xlim=[0, 10]);
#orizontal bar
bars = ax.barh(missing_percentage.index, missing_percentage.values, color = 'lightblue', edgecolor = 'black')
ax.bar_label(bars);
st.pyplot(fig)

#outliers
st.write(
    """
    *Outliers handling*
    - 
    """
)

num_variables = ['TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']
var_names = {
    'TOTCHOL' : 'Total Cholesterol (mg/dL)',
    'AGE' : 'Age',
    'SYSBP' : 'Systolic Blood Pressure (mmHg)',
    'DIABP' : 'Diastolic Bloop Pressure (mmHg)',
    'CIGPDAY' : 'Number of cigarettes per day',
    'BMI' : 'Body Mass Index',
    'HEARTRTE' : 'Heart Rate',
    'GLUCOSE' : 'Glucose'

}
selected_variable = st.selectbox(
    "Select a numeric variable to visualize",
    num_variables
)

fig, ax = plt.subplots()

sns.boxplot(
    data=cvd_death[selected_variable],
    orient="v",
    ax=ax       
)

ax.set(
    title=var_names[selected_variable],
    xlabel=var_names[selected_variable],
    ylabel='Value'
)

st.pyplot(fig)  

st.write(
    '''
- colesterol lower than 120 and higher than 500 are probably outliers --> impute!
- there are no outliers for age
- systolic blood pressure, possible in a diseased condition
- dyastolic blood pressure, possible in a diseased condition
- bmi higher than 35 reflects an obese condition
- heart rate is plausible
- very high glucose levels indicate diabetes'''
)

st.title("Distribution of numerical variables")
#distribution of the data
selected_hist = st.selectbox("Select a numeric variable to visualize", num_variables, key="hist_selectbox")

# Histogram
fig2, ax = plt.subplots()

ax.hist(cvd_death[selected_hist], edgecolor='black', bins=20, color='lightblue')

ax.set(title=var_names[selected_hist], xlabel=var_names[selected_hist], ylabel='Count')

st.pyplot(fig2)

#visualization of categorical variables

#identification of categorical variables
categorical_variables = ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS','PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'ANGINA', 'HOSPMI', 'MI_FCHD', 'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN','DEATH', 'educ']

#creation fo a dictionary with metadata description
categorical_names = {
    'SEX': 'Sex',
    'CURSMOKE': 'Current Smoking Status',
    'DIABETES': 'Diabetes Status',
    'BPMEDS': 'Use of Blood Pressure Medications',
    'PREVAP': 'Prevalent Angina Pectoris',
    'PREVMI': 'Prevalent Miocardial Infraction',
    'PREVSTRK': 'Prevalent Stroke',
    'PREVHYP': 'Prevalent Hypertension',
    'ANGINA': 'Presence of Angina at the time of collection',
    'HOSPMI': 'Hospitalized for Myocardial Infraction',
    'MI_FCHD': 'Hospitalized Myocardial Infarction or Fatal Coronary Heart Disease',
    'ANYCHD': 'Any Form of Coronary Heart Disease',
    'STROKE': 'Stroke Event Follow-up',
    'CVD': 'Any Cardiovascular Disease Event Follow-up',
    'HYPERTEN': 'Hypertension Follow-up',
    'educ': 'Education Level',
    'PREVCHD': 'Plevalent Coronary Heart Disease',
    'DEATH': 'Death'
}

#barplot to visualize categorical variables
st.title('Categorical variables')
selected_bar = st.selectbox("Select a categorical variable to visualize", categorical_variables, key="barplot")

fig3, ax = plt.subplots()
counts = cvd_death[selected_bar].value_counts()
ax.bar(counts.index, counts.values, edgecolor='black', color = ['lightblue', 'lightpink'])
ax.bar_label(ax.containers[0])
ax.set(title= categorical_names[selected_bar], xlabel= categorical_names[selected_bar], ylabel= 'Count')
st.pyplot(fig3)

#visualization of the target variable 

# Distribution of DEATH
st.write('*Death distribution*')
death_counts = cvd_death['DEATH'].value_counts()

fig, ax = plt.subplots()

ax.pie(death_counts.values, labels=['Alive', 'Dead'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'pink'])

ax.set(title='Distribution of Death Outcome')

# Show in Streamlit
st.pyplot(fig)



#bivariate analysis
st.title('Bivariate analysis')

#categorical variables vs death
selected_cat = st.selectbox("Select a categorical variable to visualize", categorical_variables, key="categorical_barplot")

counts = cvd_death.groupby(['DEATH', selected_cat]).size().unstack()

# Bar plot
ax = counts.plot(kind='bar', edgecolor='black', color=['pink', 'lightblue', 'lightgreen', 'lightyellow'], rot=0)

ax.set(title=categorical_names[selected_cat], xlabel=categorical_names[selected_cat], ylabel='Count')

st.pyplot(ax.figure)

#gropbyfuction to see the difference between died and survived for the categorical variables

st.write('Difference in numerical variables for death and survived')
mean_table = cvd_death.groupby('DEATH')[num_variables].mean()

st.dataframe(mean_table)

# Final dataframe and features
st.title("Preparing the data for our models")
st.write(""" 
         *Features used in the model*
         -
          """)
st.write("Several time-related features were removed from the dataframe, and the dataframe was split into X and y which resulted in these final dataframes:")

cvd_death['DEATH'] = cvd_death['DEATH'].replace({'survived': 0, 'died': 1});
X = cvd_death[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP',
                'PREVMI', 'PREVSTRK', 'PREVHYP', 'ANGINA', 'HOSPMI', 'MI_FCHD',
                'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN', 'educ', 'TOTCHOL', 'AGE',
                'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']]

y = cvd_death['DEATH']

st.write("X (" + str(len(X.columns)) + " features): ", )
st.dataframe(X.head(3))
st.write("y (outcome is " + str(y.name.lower()) + "):")
st.dataframe(y.head(3))

#Train Test Split

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

st.write(""" 
         *Train-Test split*
         -
         The dataframe was split into a training and testing set using sklearn train_test_split with randomstate = 1 and test_size = 0.2)
         This created: 
         - train_X
         - test_X
         - train_y
         - test_y
         """)


st.write("The train sets have " + str(train_X.shape[0]) + " rows, "
         "and the test sets have " + str(test_X.shape[0]) + " rows")

# Imputing missing data
st.write(""" 
         *Missing data imputation*
         -
         The missing data in these train sets were then imputed using:
         - The KNN imputer from sklearn using 5 n-nearest neighbours was applied to numerical features
         - The simple imputer from sklearn using the mode was applied to catagorical features
         - The imputers were fitted only on train data and then applied to test to avoid data leakage
          """)

from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np

# Create list of numeric and catagorical features:
NumCols = train_X.select_dtypes(include = "number").columns
CatCol = train_X.select_dtypes(include=['object']).columns

#See missing data
MissingCheck = train_X.isnull().sum()
MissingCols = MissingCheck[MissingCheck > 0]
MissingCols = MissingCols.rename("Missing count")

# Getting a list for later use
missing_col_names = MissingCols.index.tolist()

#Displaying missing data
st.write("""
         The tables and graphs below show train_X for the full picture since test_X had less columns with missing values
         
         Columns with missing values:
         
         """)
st.write(MissingCols)

#Copies so they can still be plotted after imputations to compare
train_X_NoImpute = train_X.copy()

# KNN imputation
imputer = KNNImputer(n_neighbors = 5)
train_X[NumCols] = imputer.fit_transform(train_X[NumCols])
test_X[NumCols] = imputer.transform(test_X[NumCols])

# Mode imputation
cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
train_X[CatCol] = cat_imputer.fit_transform(train_X[CatCol])
test_X[CatCol] = cat_imputer.transform(test_X[CatCol])

# Creating a select box for a feature to check
selected_discheck = st.selectbox("Select a feature to check if distributions after imputation are the similar and have not been affected:", 
                                 missing_col_names,
                                 key="discheck")

#Checking if a histogram or bar graph should be displayed by seeing if the column is numerical or catagorical
if selected_discheck in NumCols: 
    # Making a single figure that has two graphs made next to eachother
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

    #Graph one before impute
    ax1.hist(train_X_NoImpute[selected_discheck],
             bins = 10, 
             edgecolor='black', 
             color = 'lightblue')
    ax1.set(
        title = "Distribution before imputation",
        xlabel = var_names[selected_discheck],
        ylabel = "Value"
    )

    #Graph two after impute
    ax2.hist(train_X[selected_discheck],
            bins = 10, 
            edgecolor='black', 
            color = 'lightblue')
    ax2.set(
        title = "Distribution after imputation",
        xlabel = var_names[selected_discheck],
        ylabel = "Value"
    )

    st.pyplot(fig)
else:
    # Making a single figure that has two graphs made next to eachother
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    
    # Creating counts before and after the imputation
    counts_before = train_X_NoImpute[selected_discheck].value_counts().sort_index()
    counts_after = train_X[selected_discheck].value_counts().sort_index()

    #Graph one before impute
    ax1.bar(counts_before.index,
            counts_before.values, 
             edgecolor='black', 
             color = ['lightblue', "lightpink"])
    ax1.set(
        title = "Count before imputation",
        xlabel = categorical_names[selected_discheck],
        ylabel = "Value"
    )

    #Graph two after impute
    ax2.bar(counts_after.index,
            counts_after.values, 
             edgecolor='black', 
             color = ['lightblue', "lightpink"])
    ax2.set(
        title = "Count after imputation",
        xlabel = categorical_names[selected_discheck],
        ylabel = "Value"
    )

    st.pyplot(fig)

#Encoding

train_X = pd.get_dummies(train_X, columns=CatCol, drop_first=True, dtype=float)
test_X = pd.get_dummies(test_X, columns=CatCol, drop_first=True, dtype=float)

#Checking if the columns still match and no different dummies were made. Converted to a set so the order is not important
if set(train_X.columns) == set(test_X.columns):
    st.write("""
         *Encoding*
         -
         All catagorical variables were encoded using get_dummies from pandas, 
         where the first dummy created is dropped since this would be redundant. 
         The datatype is also changed to float so it can be used in the models. 
         """)
else:
    st.write("COLUMNS DONT MATCH") #To visually check if everything is right

# Scaling
from sklearn.preprocessing import StandardScaler

# Loop to make a scaler for every column and then fits it on train and applies it to test and train
for column in NumCols:
    scaler = StandardScaler()
    scaler.fit(train_X[[column]]) 
    train_X[column] = scaler.transform(train_X[[column]])
    test_X[column] = scaler.transform(test_X[[column]])

st.write("""
         *Scaling*
         -
         All numerical features were scaled using the StandardScaler from sklearn. 
         This was first fitted on the train set and then the train and test 
         set were scaled using this scaler to avoid data leakage.
         """)

# Feature selection
st.title("Feature selection")
st.write("""
         Feature selection was done on the basis of several (combined) methods:
         - Statistical importance (Filter, Wrapper)
         - Related features
         - Important features in literature

        *Filter feature selection*
         -
         """)

st.write("""
        The ANOVA filter ranks features by testing if mean values differ
        significantly between target groups. Each feature is evaluated seperately
        so the ranking depends only on the statistical test, not on the predictive model 
        or interactions between features.
         """)

# Select best features based on ANOVA score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

best_features = SelectKBest(score_func = f_classif, # f_classif is ANOVA F-score
                            k = 'all') # uses all features, instead of default 10
fit = best_features.fit(train_X, train_y)

# Making it into a DF so it can be plotted well
featureScores = pd.DataFrame(
    data = fit.scores_,
    index = list(train_X.columns),
    columns = ['ANOVA Score'])

# Plotting
fig, ax = plt.subplots(figsize = (5,7))
sns.heatmap(featureScores.sort_values(by = "ANOVA Score", ascending = False), annot = True)
plt.title("Filter feature selection (ANOVA)");
st.pyplot(fig)

# Wrapper selection
st.write("""
         *Wrapper feature selection*
         -
         Since wrapper feature selection looks at the variables in relation to eachother, 
         the type of model is also infuential in the resulting ranking
         """)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# Whenever we select models or change anything, the whole code would rerun and this would take a 
# long time. 
# I wanted to save eveything as a dataframe first, because I thought that would fix it like in colab,
# but in streamlit it doesnt work because eveything is rerun top to bottom.
# So I found this function that caches the data and that seems to work.
@st.cache_data
def compute_feature_rankings(train_X, train_y):
    #LogSelectionWrapper
    estimator = LogisticRegression(class_weight='balanced', max_iter=1000)
    selector = RFE(estimator,
                n_features_to_select=1, # Ranking now goes to 1 instead of having top 5
                step=1)
    selector = selector.fit(train_X, train_y)
    featureRankingWrapperLog = pd.DataFrame(
        data=selector.ranking_,
        index = list(train_X.columns),
        columns=['Feature ranking'])  

    #SVMSelectorWrapper
    estimator = SVC(kernel="linear", class_weight="balanced")
    selector = RFE(estimator,
                n_features_to_select=1, 
                step=1)
    selector = selector.fit(train_X, train_y)
    featureRankingWrapperSVM = pd.DataFrame(
        data=selector.ranking_,
        index = list(train_X.columns),
        columns=['Feature ranking'])  

    #RFCSelectorWrapper
    estimator = RandomForestClassifier(n_estimators=200, random_state=42, max_depth = 10) # Added maxdepth because it will take a long time otherwise, for the final RFC it will not be needed
    selector = RFE(estimator,
                n_features_to_select=1,
                step = 1)
    selector = selector.fit(train_X, train_y)
    featureRankingWrapperRFC = pd.DataFrame(
        data=selector.ranking_,
        index = list(train_X.columns),
        columns=['Feature ranking'])  
    return (
        featureRankingWrapperLog,
        featureRankingWrapperSVM,
        featureRankingWrapperRFC
    )

#All rankings are already run even if not selected to reduce load time and we can use them later on for modeling
featureRankingWrapperLog, featureRankingWrapperSVM, featureRankingWrapperRFC = compute_feature_rankings(train_X, train_y)

#Extracting top15 lists for later 
Top15LogWrap = featureRankingWrapperLog.sort_values(by = "Feature ranking", ascending = True).head(15).index.tolist()
Top15SVMWrap = featureRankingWrapperSVM.sort_values(by = "Feature ranking", ascending = True).head(15).index.tolist()
Top15RFCWrap = featureRankingWrapperRFC.sort_values(by = "Feature ranking", ascending = True).head(15).index.tolist()
Top15Filter = featureScores.sort_values(by = 'ANOVA Score', ascending = False).head(15).index.tolist()

#Selecting which to display
modeltypes = ["Logistic Regression", "SVM", "RFC"]
SelectedModelWrap = st.selectbox("Please select a model:", 
                            modeltypes,
                            key="modelselwrap")
    
#Function for heatmap display of wrapper
def WrapperHeatmap(SelectedModelWrapDF):
    fig, ax = plt.subplots(figsize = (5,7))
    sns.heatmap(SelectedModelWrapDF.sort_values(by = "Feature ranking", ascending = True), annot = True)
    plt.title("Wrapper feature selection ranking");
    st.pyplot(fig)

#Displaying right heatmap
if SelectedModelWrap == "Logistic Regression":
    WrapperHeatmap(featureRankingWrapperLog)
elif SelectedModelWrap == "SVM":
    WrapperHeatmap(featureRankingWrapperSVM)
else: 
    WrapperHeatmap(featureRankingWrapperRFC)

st.write("""
         *Custom features set*
         - 

        Most features such as age, diabetes and other clinical markers prevalent in 
        literature also showed up as the best predictors for out models. Thus most of these were included. 
        Features that did not consitantly rank high such as smoking, cigarettes per day and heartrate, 
        are still important in literature and are kept in.
        
         Consistantly low ranking features are:
        - REVSTRK and PREVMI were redundant once stronger variables capturing cardiovascular and 
         cerebrovascular events (e.g., MI_FCHD, CVD_did occur, STROKE_did occur) were included, 
         leading to collinearity without improving prediction.
        - BPMEDS showed the lowest importance overall due to strong confounding
          (medication can imply both higher risk and successful treatment), providing inconsistent 
            and non-interpretable signals.
        - Education features are also constantly ranking low, but this does not allign with
          literature since education can be a predictor of death. This it is kept in
        
        Therefore only REVSTRK, PREVMI and BPMEDS were excluded to improve model stability and interpretability.
         """)

# Modelling
st.title("Modelling")
st.write("""
        In this section, different models were made by selecting different
        model types and feature sets to find the best model.
         
        The models (from SKlearn):
        - Support Vector Classifier with a linear kernel
        - Logistig Regression
        - Random Forest Classifier with 200 estimators, random state 1 and a max depth of 10
        - All models use a balanced class weight
         
         Performance is judged on:
         - Accuracy
         - Precision
         - Recall
         - F1-score
         """)

SelectedModel = st.selectbox("Please select a model:", 
                            modeltypes,
                            key="modelsel")

subsets = ["All features", "Top 15 wrapper features", "Top 15 filter features", "Final selection"] # NEEDS TO BE UPDATED
SelectedSubset = st.selectbox("Please select the features to be used:", 
                            subsets, 
                            key = "subsetsel")
def ModelOutput(modelselectionanswer, subsetselectionanswer):
    #Checkign selected model
    if modelselectionanswer == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced')
    elif modelselectionanswer == "SVM":
        model = SVC(class_weight='balanced', kernel="linear")
    else:
            model = RandomForestClassifier(n_estimators=200, random_state=1,  class_weight="balanced", max_depth= 10)

    if subsetselectionanswer == "All features":
        subsetused = train_X.columns
    #Check to see what model is used, because wrapper changes per model
    elif subsetselectionanswer == "Top 15 wrapper features":
        if modelselectionanswer == "Logistic Regression":
            subsetused = Top15LogWrap
        elif modelselectionanswer == "SVM":
            subsetused = Top15SVMWrap
        else:
            subsetused = Top15RFCWrap
    elif subsetselectionanswer == "Top 15 filter features":
        subsetused = Top15Filter
    else:
        train_X_custom = train_X.drop(columns = [
            "PREVSTRK_prevalent disease",
            "PREVMI_prevalent disease",
            "BPMEDS_not currently used"])
        subsetused = train_X_custom.columns

    #Everything after this for models, needs to be done for every model
    model.fit(train_X[subsetused], train_y)
    pred_y = model.predict(test_X[subsetused])

    from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

    #This does not display nicely like it does in google colab
    Dict = (classification_report(test_y, pred_y, output_dict=True))  #Output into a dict, otherwise st.table will give an arror about it being a string
    Fig = ConfusionMatrixDisplay.from_estimator(model, test_X[subsetused], test_y)
    Fig = Fig.figure_ 

    #Since its now a dictionairy, we have to change the keys by removing and replacing the old ones
    #All the others are also done to keep the order the same. Otherwise survived and died would be at the end.
    #There is probably a more efficient way to do this but I could not find it
    Acc = Dict["accuracy"]
                            
    Dict["survived"] = Dict.pop("0")
    Dict["died"] = Dict.pop("1")
    Dict.pop("accuracy")
    Dict["macro avg"] = Dict.pop("macro avg")
    Dict["weighted avg"] = Dict.pop("weighted avg")

    st.write("Accuracy = " + str(round(Acc, 4)))
    st.table(Dict)
    st.pyplot(Fig)

ModelOutput(SelectedModel, SelectedSubset)

st.write("""From this we concluded that out best model was a RFC using out 
         costum final feature set""")


st.write("""
         *Cross validation*
         - 
         
         To check if this final model also performs well over different train-test splits, 
         we also performed cross validation.

         We did this by using cross_validate from sklearn with the following cross validate method:
         - RepeatedKFold from sklearn
         - 5 splits
         - 10 repeats
         - Randomstate of 1

        Final model cross validation performance:
        
         """)

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate

# New final model since the last is only in the function and not saved outside
ModelFinal = RandomForestClassifier(n_estimators=200, random_state=1,  class_weight="balanced", max_depth= 10)

# Scoring methods that are used, macro is because of class inbalance
scoring = ["accuracy", "roc_auc", "f1_macro", "precision_macro", "recall_macro"]

#cross validation used
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)

#Same code as before but now outside of the function
# and if statements so it is saved and can be used
train_X_crossval = train_X.drop(columns = [
            "PREVSTRK_prevalent disease",
            "PREVMI_prevalent disease",
            "BPMEDS_not currently used"])

# This is again a step that takes long so its cached,
#  nothing is passed to the function because that is giving 
# an error that it cannot be hashed?
# So now eveything is directly in the function already, we just need to run it
@st.cache_data
def run_cv():
    return cross_validate(
        ModelFinal,
        train_X_crossval,
        train_y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

# Running it once
scores_initial = run_cv()

# making it a dataframe so I can use mean() and std()
scores_df = pd.DataFrame(scores_initial)

# Extracting the scores from that dataframe, might be a better way than this manual way
MeanAcc = scores_df["test_accuracy"].mean()
MeanRoc = scores_df["test_roc_auc"].mean()
Meanf1 = scores_df["test_f1_macro"].mean()
MeanPrecision_macro = scores_df["test_precision_macro"].mean()
MeanRecall_macro = scores_df["test_recall_macro"].mean()
StdAcc = scores_df["test_accuracy"].std()
StdRoc = scores_df["test_roc_auc"].std()
Stdf1 = scores_df["test_f1_macro"].std()
StdPrecision_macro = scores_df["test_precision_macro"].std()
StdRecall_macro = scores_df["test_recall_macro"].std()

# Writing it down in a nice and rounded format
st.write("Accuracy: " + str(round(MeanAcc, 4)) + " ± " + str(round(StdAcc, 3)))
st.write("ROC AUC: " + str(round(MeanRoc, 4)) + " ± " + str(round(StdRoc, 3)))
st.write("F1 Macro: " + str(round(Meanf1, 4)) + " ± " + str(round(Stdf1, 3)))
st.write("Precision Macro: " + str(round(MeanPrecision_macro, 4)) + " ± " + str(round(StdPrecision_macro, 3)))
st.write("Recall Macro: " + str(round(MeanRecall_macro, 4)) + " ± " + str(round(StdRecall_macro, 3)))