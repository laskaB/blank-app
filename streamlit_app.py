#import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Project Framingham Heart Disease - MAI3002")
st.markdown("##### ***This project was developed by Group 7: Veronica Scolari (i6350511), Bjørn Rejhons (i6324753), and Lucia Mas León (i6353881).***")

##SECTION 1: GENERAL DESCRIPTION OF THE STUDY
with st.expander("## GENERAL DESCRIPTION OF THE FRAMINGHAM HEART DISEASE STUDY"):

    st.markdown("##  GENERAL OVERVIEW OF THE STUDY")

    st.write(
        "The Framingham Heart Study is the **first prospective study** of cardiovascular disease (CVD). It was used to identified CVD risks factors and their combined effects. " \
        "The study began in 1948 among a population of free-living subjects in the community of Framingham, Massachusetts. Initially, 5209 people were enrolled. In the subset analyzed, there are **4434 patients**, each of them with **three examination periods** aproximately every 6 years. Each partecipant was followed for 24 years to detect the following outcomes: **angina pectoris, myocardial infraction, Atherothrombotic Infarction or Cerebral Hemorrhage (Stroke) or Death**. (1)"
    )
    #source: https://search.r-project.org/CRAN/refmans/riskCommunicator/html/framingham.html

    #data loading
    cvd = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

    st.markdown("###  *Dataset Description - Raw Dataset*")

    #first look at the data
    st.dataframe(cvd.head())

    #observations about the data
    st.write(
        "The dataset contains **39 features** including: demographic information, clinical measurements, lifestyle, medications, prevalent diseases, and follow up outcomes."
    )
    st.write("The dataset includes multiple records for a single person (RANDID) and each row represents one **clinical examination**.")

## SECTION 2: RESEARCH QUESTION AND SUBSET CREATION
with st.expander("## RESEARCH QUESTION AND SUBSET CREATION"):

    st.markdown('## ***To what extenct can baseline health indicators predict all-cause death?***')

    st.markdown(
        """
        This research aims to understand if baseline health indicators can predict all cause mortality with good performance. The objective is to assess the predictive power of these feature to flag subgroups at risk, not to create a tool for clinical decision making.

        To answer this research question, a subset of the original dataframe was created:

        - only patients from period 1 were selected,
        - the features related to time and other possible outcomes were not included,
        - HDL and LDL cholesterol were dropped as they were not available in period 1.
        """
    )

    #we performed metadata enrichment to add more context to the data and its visualization
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
    cvd['ANGINA'] = cvd['ANGINA'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['HOSPMI'] = cvd['HOSPMI'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['MI_FCHD'] = cvd['MI_FCHD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['ANYCHD'] = cvd['ANYCHD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['STROKE'] = cvd['STROKE'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['CVD'] = cvd['CVD'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['HYPERTEN'] = cvd['HYPERTEN'].replace({0: 'did not occur during followup', 1:'did occur during followup'})
    cvd['educ'] = cvd['educ'].replace({1: 'higher secondary', 2:'graduation', 3:'post graduation', 4: 'PhD'})

    #subset creation: only partecipants from period 1 were selected
    cvd_death = cvd.loc[(cvd['PERIOD'] == 'period 1')]
    #all the columns related to time, outcomes, HDL and LDL were removed
    cvd_death = cvd_death.drop(columns = ['TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD', 'TIMEHYP', 'HDLC', 'LDLC', 'TIME', 'PERIOD', 'TIMEDTH', 'ANGINA', 'HOSPMI', 'MI_FCHD', 'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN'])
    cvd_death['DEATH'] = cvd_death['DEATH'].replace({0: 'survived', 1:'died'})

    #final subest shape
    st.write("The final subset contains the following number of rows and columns (patients and features respectively):")
    cvd_death.shape

    #displaying the columns of the final subset
    st.write("The final subset contains the following features:")
    st.write(pd.DataFrame(cvd_death.columns, columns = ['Features']))

##SECTION 3: EXPLORATORY DATA ANALYSIS
with st.expander("## EXPLORATORY DATA ANALYSIS"):
    #OVERVIEW
    st.markdown(
        """
        ### OVERVIEW:
        1. Descriptive statistics
        2. Missing Values Analysis
        3. Outlier Detection
        4. Erroneous Data Detection
        5. Distribution of Numerical Variables
        6. Exploration of Categorical Variables
        7. Target Variable Exploration
        8. Bivariate Analysis
    """
    )


    #1. Descriptive statistics
    st.markdown('###  *Descriptive Statistics*')

    # table showing descriptive statistics for numerical variables
    st.write("Descriptive statistics of the **numerical variables**:")
    st.dataframe(cvd_death.describe())

    # table showing descriptive statistics for categorical variables
    st.write("Descriptive statistics of the **categorical variables**:")
    st.dataframe(cvd_death.describe(include = 'object'))

    #interpretation
    st.markdown(
        """
        In this subset with patients from Period 1, 56% are women and 51% non-smokers.
        The majority are not diabetic and free of prevalent diseases.
        Most partecipants are middle-aged, with a mean of 50 years old.
        The mean cholesterol is 237 mg/dL, which is borderline high, and the mean BMI is 25.8 (overweight).
        The mean systolic (133 mmHg) and diastolic (83 mmHg) blood pressure suggest potential hypertensive individuals.
        The mean glucose is 82 mg/dL, which represents normal levels.
        A mean of 9 cigarettes per day is observed, even if many participants report zero.
        It is possible to observe that the largest education group is secondary education.
        """
    )


    #2. Addressing Missing Values
    st.markdown("###  *Missing Values*")

    #calculating missing values percentages
    missing_percentage = cvd_death.isnull().mean() * 100
    missing_percentage = missing_percentage[missing_percentage > 0]

    #visual rapresentation of the missing data
    fig, ax = plt.subplots()
    ax.set(title = "Missing data",
            xlabel = "Percent missing",
            ylabel = "Variable",
            xlim = [0, 10]);

    bars = ax.barh(missing_percentage.index,
                   missing_percentage.values,
                   color = 'lightblue',
                   edgecolor = 'black')

    ax.bar_label(bars);
    st.pyplot(fig)

    #explanation
    st.write(
        """
        There are a few missing values present in this subset: between 0.02% and 9% in 7 features.

        * Some self-reported variables (education 2.5%, blood pressure medication 1.4%, cigarettes per day 0.7%) are missing probably due to incomplete reporting.
        * Glucose shows the highest missingness, 8.95%.
        * Clinical measurements (Total cholesterol 1.2%, BMI 0.4%, Heart Rate 0.02%) show only minimal missingness, probably due to measurement errors or reporting mistakes.

        It was decided to impute the missing values after the train/test split.

        *Imputation strategy*:

        * **Numerical variables** were imputed using K-Nearest Neighbours imputation (k = 5).
        * **Categorical variables** were imputed using the mode (most frequent value).
        """
    )


    #3. Outliers
    st.markdown('###  *Outliers*')

    # numerical variables
    num_variables = ['TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']

    # description of numerical variables
    num_names = {
        'TOTCHOL' : 'Total Cholesterol (mg/dL)',
        'AGE' : 'Age',
        'SYSBP' : 'Systolic Blood Pressure (mmHg)',
        'DIABP' : 'Diastolic Bloop Pressure (mmHg)',
        'CIGPDAY' : 'Number of cigarettes per day',
        'BMI' : 'Body Mass Index',
        'HEARTRTE' : 'Heart Rate',
        'GLUCOSE' : 'Glucose'

    }

    #slectbox to visulize a boxplot showing possible outliers
    selected_variable = st.selectbox(
        "Select a numeric variable to visualize:",
        num_variables
    )

    fig, ax = plt.subplots()
    sns.boxplot(data = cvd_death[selected_variable], orient = "v", ax = ax)
    ax.set(title = num_names[selected_variable], xlabel = num_names[selected_variable], ylabel = 'Value')
    st.pyplot(fig)

    #explanation
    st.write(
        """
        - Cholesterol levels higher than 500 mg/dL may indicate not only outliers, but also measurement errors.
        These extreme values are rare and usually related only to a few specific medical conditions (https://pubmed.ncbi.nlm.nih.gov/24404629/).
        Values lower than 500 mg/dL are physiologically possible and could be correlated to increased risk of CVD.
        - There are no outliers for age.
        - Systolic blood pressure higher than 180 mmHg indicate a hypertensive crisis, and high values are associated with CVD. However, systolic pressures above 260 mmHg may be related to measurement errors.
        - In diastolic blood pressure values higher than 120 mmHg indicate a hypertensive crisis. These values were not removed because they are physiologically possible (https://www.ncbi.nlm.nih.gov/books/NBK507701/).
        - For the number of cigarettes smoked per day, there were some values outside the IQR. It was decided to clip them at Q3, because even if these values are plausible, they are extremly high.
        - BMI values above 35 reflect an obese condition. BMI values higher than 50 are physiologically plausible.
        - Heart rate values between 40 and 140 bpm are plausible.
        - Very high glucose levels indicate diabetes. Values higher than 180 mg/dL indicate hyperglycemia. Extreme values around 400 mg/dL could be related to uncontrolled diabetes. Values over 300 mg/dL were considered as outliers, because these values imply emergency care. The glucose represented is casual serum glucose, which showed a relationship with increased risk of CVD and mortality (https://drc.bmj.com/content/9/1/e001928).

        ***Imputation strategy***

        Cholesterol, Systolic Blood Pressure and Glucose above the previously stated thresholds were clipped at that specific value.
        The datapoints outside the interquartile range of number of cigarettes smokes per day were clipped at Q3.

        """
        )

    #4. Erroneous Data
    st.markdown('###  *Erroneous Data*')
    st.write("""
             - As reported in the outlier detection, some values are extremly high and might be related to measurement errors or incorrect reporting. Especially in cholesterol, glucose and systolic blood pressure.
             - It was decided to check for internal consistency:
                - partecipants without diabetes are not expected to have glucose levels above 200 mg/dL (https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis#:~:text=Table_title:%20Test%20results%20for%20diagnosis%20of%20prediabetes,Plasma%20Glucose:%20126%20mg/dL%20or%20above%20%7C)
                - non-smokers should smoke 0 cigarettes per day,
                - diastolic blood pressure can't be higher than systolic.
             """)

    #High glucose levels: if a patient doesn't have diabetes but has a glucose higher than 120 mg/dL, this might be an erroneous data.
    cvd_death[(cvd_death['DIABETES'] == 'not a diabetic') & (cvd_death['GLUCOSE'] > 200)]
    st.write('There are not patients without diabetes but with a glucose higher than 200 mg/dL. ')

    #Non smokers should smoke 0 cigarettes per day
    cvd_death[(cvd_death['CURSMOKE'] == 'not current smoker') & (cvd_death['CIGPDAY'] > 0)]
    st.write('None of the non-smokers report to smoke more than zero cigarettes per day.')

    #Diastolic blood pressure higher than systolic
    cvd_death[(cvd_death['SYSBP'] < cvd_death['DIABP'])]
    st.write('There are no partecipants with diastolic blood pressure higher than systolic blood pressure.')


    #5. Distribution of numerical variables
    st.markdown('###  *Distribution of numerical variables*')

    #distribution of the data
    selected_hist = st.selectbox('Select a numeric variable to visualize', num_variables, key = "hist_selectbox")

    # Histogram
    fig2, ax = plt.subplots()
    ax.hist(cvd_death[selected_hist],
            edgecolor = 'black',
            bins = 20,
            color = 'lightblue')

    ax.set(title = num_names[selected_hist],
           xlabel = num_names[selected_hist],
           ylabel = 'Count')
    st.pyplot(fig2)

    #description
    st.write(
        """
        * Cholesterol, systolic blood pressure, and glucose have a right-skewed normal distribution.
        * Age has a slight bimodal distribution.
        * Diastolic blood pressure, BMI, and heart rate display a normal distribution, slightly right-skewed.
        * The number of cigarettes smoked shows a high number of zeros, due to non-smokers. The distribution of the other values roughly resembles a normal distribution.
        """
        )


    #6. Categorical Variables visualization

    #identification of categorical variables
    categorical_variables = ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS','PREVCHD', 'PREVAP', 'PREVMI', 'PREVSTRK', 'PREVHYP', 'DEATH', 'educ']

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
        'educ': 'Education Level',
        'PREVCHD': 'Plevalent Coronary Heart Disease',
        'DEATH': 'Death'
    }

    st.markdown('###  *Categorical variables visualization*')

    #bselectbox to visualize categorical variables
    selected_bar = st.selectbox("Select a categorical variable to visualize", categorical_variables, key="barplot")

    #barplot to visualize categorical variables
    fig3, ax = plt.subplots()
    counts = cvd_death[selected_bar].value_counts()
    ax.bar(counts.index,
           counts.values,
           edgecolor = 'black',
           color = ['lightblue', 'lightpink'])
    ax.bar_label(ax.containers[0])
    ax.set(title= categorical_names[selected_bar], xlabel= categorical_names[selected_bar], ylabel= 'Count')
    st.pyplot(fig3)

    #explanation
    st.markdown(
                """
                The histograms are used to better visualize the population described with the descriptive statistics table.
                The dataset is balanced regarding sex and smoking status.
                However, it shows high imbalance for diabetic status, use of blood pressure medications, prevalent diseases (coronary heart disease, angina pectoris, myocardial infarction, stroke), where only 0.7- 4.4% of the population shows.
                32.3% of the people have prevalent hypertension, and 16.4% report angina at the time of collection.
                The most common education level is higher secondary with 42.2%, followed by graduation with 29.6%.
                """ )

    #7. Visualization of the Target Variable
    st.markdown('###  *Target variable visualization*')

    death_counts = cvd_death['DEATH'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(death_counts.values, labels = ['Alive', 'Dead'], autopct = '%1.1f%%', startangle = 90, colors = ['lightblue', 'pink'])
    st.pyplot(fig)

    st.write(
        """
        65% of participants in the dataset were alive during the study period, while 35% died.
        """)

    #8. Bivariate Analysis (categorical and numerical variables)
    st.markdown('###  *Bivariate analysis*')

    # Selectbox to show categorical variables vs death
    selected_cat = st.selectbox("Select a categorical variable to visualize:", categorical_variables, key = "categorical_barplot")

    # Groupby function to visualize the bivariate analysis: death vs the selected categorical variable
    counts = cvd_death.groupby(['DEATH', selected_cat]).size().unstack()

    # Bar plot
    ax = counts.plot(kind = 'bar',
                     edgecolor = 'black',
                     color = ['pink', 'lightblue', 'lightgreen', 'lightyellow'],
                     rot = 0)
    ax.set(title = categorical_names[selected_cat], xlabel = categorical_names[selected_cat], ylabel = 'Count')
    st.pyplot(ax.figure)

    # Explanation
    st.markdown( """
                From the bivariate analysis, it is possible to observe that 54.4% of the people who died were male, and there was a slightly higher percentage of smokers, 50.8%.
                There was a higher proportion of diabetic people who use blood pressure medication and with prevalent diseases (coronary heart disease, angina pectoris, myocardial infarction, stroke) in the death population.
                Remarkably is that 48.9% of the people who died had hypertension.
                """ )

    # Gropby fuction to see the difference between died and survived for the categorical variables
    st.write('Difference in numerical variables for death and survived:')
    mean_table = cvd_death.groupby('DEATH')[num_variables].mean()
    st.dataframe(mean_table)
    st.markdown("""
                It is possible to observe that all the numerical variables are higher in the death population compared to the survived, indicating that higher values could be related to higher risk of death.
                """)

##SECTION 4: MISSING DATA IMPUTATION, ENCODING, SCALING
with st.expander("## MISSING DATA IMPUTATION, ENCODING, SCALING"):

    # Final dataframe and features
    st.title("Preparing the data for our models")
    st.write("""
            *Features used in the model*
            -
            """)
    st.write("Several time-related features were removed from the dataframe, and the dataframe was split into X and y which resulted in these final dataframes:")

    cvd_death['DEATH'] = cvd_death['DEATH'].replace({'survived': 0, 'died': 1});
    X = cvd_death[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP',
                    'PREVMI', 'PREVSTRK', 'PREVHYP', 'educ', 'TOTCHOL', 'AGE',
                    'SYSBP', 'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE']]

    y = cvd_death['DEATH']

    st.write("X (" + str(len(X.columns)) + " features): ", )
    st.dataframe(X.head(3))
    st.write("y (outcome is " + str(y.name.lower()) + "):")
    st.dataframe(y.head(3))

    # Train Test Split

    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify = y, random_state=1)

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

    # Define physiologically informed thresholds
    CIGPDAY_MAX = train_X['CIGPDAY'].quantile(0.75)
    GLUCOSE_MAX = 300
    TOTCHOL_MAX = 500
    SYSBP_MAX = 260

    train_X['CIGPDAY'] = train_X['CIGPDAY'].clip(upper = CIGPDAY_MAX)
    test_X['CIGPDAY'] = test_X['CIGPDAY'].clip(upper = CIGPDAY_MAX)

    train_X['GLUCOSE'] = train_X['GLUCOSE'].clip(upper = GLUCOSE_MAX)
    test_X['GLUCOSE'] = test_X['GLUCOSE'].clip(upper = GLUCOSE_MAX)

    train_X['TOTCHOL'] = train_X['TOTCHOL'].clip(upper = TOTCHOL_MAX)
    test_X['TOTCHOL'] = test_X['TOTCHOL'].clip(upper = TOTCHOL_MAX)

    train_X['SYSBP'] = train_X['SYSBP'].clip(upper = TOTCHOL_MAX)
    test_X['SYSBP'] = test_X['SYSBP'].clip(upper = TOTCHOL_MAX)


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
                                    key = "discheck")

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
            xlabel = num_names[selected_discheck],
            ylabel = "Value"
        )

        #Graph two after impute
        ax2.hist(train_X[selected_discheck],
                bins = 10,
                edgecolor='black',
                color = 'lightblue')
        ax2.set(
            title = "Distribution after imputation",
            xlabel = num_names[selected_discheck],
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

    train_X = pd.get_dummies(train_X, columns = CatCol, drop_first = True, dtype = float)
    test_X = pd.get_dummies(test_X, columns = CatCol, drop_first = True, dtype = float)

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


##SECTION 5: FEATURE SELECTION AND MODEL TRAINING/TESTING
with st.expander("## FEATURE SELECTION AND MODEL TRAINING/TESTING"):

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

    #explanation
    st.write("""
            ### *Feature selection analysis:*
             Across all the methods tested some core predictors are: age, blood pressure, hypertension, diabetes and prior cardiovascular events.
             Blood pressure medications and education levels score low among all the models.
             - Eventhough education levels scores low statistically, it was decided to 
             include them in the model since they have been proven to be important for 
             predicting death in literature (https://pmc.ncbi.nlm.nih.gov/articles/PMC4435622/#R2)
             - Due to the uncertain nature and the low statistical ranking of blood pressure
              medication it was decided to remove this variable. Medication status could mean a disease burden, 
             but also a lower risk of death in individuals. This is conflicting and difficult to
              interpret for ML models.
             
             Our final custom subset thus uses all (not yet excluded) features except blood pressure medication


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

    subsets = ["All features", "Top 15 wrapper features", "Top 15 filter features", "Final selection"]
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
            train_X_custom = train_X.drop(columns = "BPMEDS_not currently used",)
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

    st.write("""
            All the models showed limited performance in predicting death. 
            It was decided to select the random forest classifier as the best model. 
            It achieved the highest precision for the death class (0.6448)
            and a recall of 0.6032. Eventhough other models had higher recall,
            they showed lower precision.
            This model was selected because it provided the most balance between 
            identifying true deaths, without raising false alarms.
             
            Altough it performed slightly worse, our custom subset was also selected as best
            since it has the best clinically relevant features, and not just the best statistical features. 
            If we selected only based on best performance here, we risk overfitting to our current data.
            """)


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
    train_X_crossval = train_X.drop(columns = "BPMEDS_not currently used",)


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

    st.write(
        """
        # *CONCLUSION*
        The final model achieved a cross-validated ROC AUC of 0.7853 and a F1 score of 0.7171.
        The discriminative performance is low and it can't be used in clinical settings.
        It is possible to conclude that the baseline health indicators alone are not sufficient to predict all cause death.
        Further research could be done on period 2 and 3 to visualize how these follow-up periods modify the performance.
        Additionally, all cause death prediction is challenging as there could be different causes, which are not captured by limited health indicators.
        It might be informative to compare these results to death caused by CVD.
        """
    )