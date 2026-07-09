
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

OUTPUT="images"
os.makedirs(OUTPUT,exist_ok=True)

# LOAD DATA
#----------------------------------------------------
def load_data():
    df= pd.read_csv("dataset/survey lung cancer.csv")
    return df

# INSPECT DATA
#----------------------------------------------------
def inspect_data(df):
    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- TAIL ---")
    print(df.tail())

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- DESCRIBE ---")
    print(df.describe())

    print("\n--- CORR ---")
    print(df.corr(numeric_only=True))

    print("\n--- ISNULL ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATED ---")
    print(df.duplicated().sum())

    print("\n--- COLUMNS ---")
    print(df.columns)

# CLEAR DATA
#----------------------------------------------------
def clear_data(df):
    df.drop_duplicates(inplace=True)
    return df

# BASIC DATA ANALYSIS
#----------------------------------------------------
def basic_anaylsis(df):

    cancer_ratio=df["LUNG_CANCER"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=cancer_ratio.index,y=cancer_ratio.values)
    plt.title("LUNG CANCER CHART")
    plt.xlabel("LUNG CANCER")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/lung_cancer_chart.png")
    plt.show()

    gender_ratio=df["GENDER"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=gender_ratio.index,y=gender_ratio.values)
    plt.xticks(["F", "M"], ['Female', 'Male'],rotation=45,ha="right")
    plt.title("GENDER CHART")
    plt.xlabel("GENDER")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/gender_chart.png")
    plt.show()

    age_ratio=df["AGE"].value_counts()
    plt.figure(figsize=(14,7))
    sb.barplot(x=age_ratio.index,y=age_ratio.values)
    plt.title("AGE CHART")
    plt.xlabel("AGE")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/age_chart.png")
    plt.show()

    smoking_ratio=df["SMOKING"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=smoking_ratio.index,y=smoking_ratio.values)
    plt.title("SMOKING CHART")
    plt.xticks(["1","2"], ['NOT SMOKING', 'SMOKING'],rotation=45,ha="right")
    plt.xlabel("SMOKING")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/smoking_chart.png")
    plt.show()

    YELLOW_FINGERS_ratio=df["YELLOW_FINGERS"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=YELLOW_FINGERS_ratio.index,y=YELLOW_FINGERS_ratio.values)
    plt.xticks(["2","1"], ['YELLOW FINGERS', 'NORMAL FINGERS'],rotation=45,ha="right")
    plt.title("YELLOW FINGERS CHART")
    plt.xlabel("YELLOW FINGERS")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/yellow_fingers_chart.png")
    plt.show()

    ANXIETY_ratio=df["ANXIETY"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=ANXIETY_ratio.index,y=ANXIETY_ratio.values)
    plt.xticks(["2","1"], ['HAVE ANXIETY', 'NORMAL MENTALITY'],rotation=45,ha="right")
    plt.title("ANXIETY CHART")
    plt.xlabel("ANXIETY")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/anxiety_chart.png")
    plt.show()

    PEER_PRESSURE_ratio=df["PEER_PRESSURE"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=PEER_PRESSURE_ratio.index,y=PEER_PRESSURE_ratio.values)
    plt.xticks(["1","2"], ['NO PEER PRESSURE', 'HAVE PEER PRESSURE'],rotation=45,ha="right")
    plt.title("PEER PRESSURE CHART")
    plt.xlabel("PEER PRESSURE")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/peer_pressure_chart.png")
    plt.show()

    CHRONIC_DISEASE_ratio=df["CHRONIC DISEASE"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=CHRONIC_DISEASE_ratio.index,y=CHRONIC_DISEASE_ratio.values)
    plt.xticks(["2","1"], ['YELLOW FINGERS', 'NORMAL FINGERS'],rotation=45,ha="right")
    plt.title("CHRONIC DISEASE CHART")
    plt.xlabel("CHRONIC DISEASE")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/chronic_disease_chart.png")
    plt.show()

    FATIGUE_ratio=df["FATIGUE"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=FATIGUE_ratio.index,y=FATIGUE_ratio.values)
    plt.xticks(["1","2"], ['NORMAL', 'FATIGUE'],rotation=45,ha="right")
    plt.title("FATIGUE CHART")
    plt.xlabel("FATIGUE")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/fatigue_chart.png")
    plt.show()

    ALLERGY_ratio=df["ALLERGY"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=ALLERGY_ratio.index,y=ALLERGY_ratio.values)
    plt.xticks(["1","2"], ['NORMAL', 'HAVE ALLERGY'],rotation=45,ha="right")
    plt.title("ALLERGY CHART")
    plt.xlabel("ALLERGY")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/allergy_chart.png")
    plt.show()

    WHEEZING_ratio=df["WHEEZING"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=WHEEZING_ratio.index,y=WHEEZING_ratio.values)
    plt.xticks(["1","2"], ['NO WHEEZING', 'HAVE WHEEZING'],rotation=45,ha="right")
    plt.title("WHEEZING CHART")
    plt.xlabel("WHEEZING")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/wheezing_chart.png")
    plt.show()

    ALCOHOL_CONSUMING_ratio=df["ALCOHOL CONSUMING"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=ALCOHOL_CONSUMING_ratio.index,y=ALCOHOL_CONSUMING_ratio.values)
    plt.xticks(["1","2"], ['NO ALCOHOL', 'CONSUMING ALCOHOL'],rotation=45,ha="right")
    plt.title("ALCOHOL CONSUMING CHART")
    plt.xlabel("ALCOHOL CONSUMING")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/alcohol_consuming_chart.png")
    plt.show()

    COUGHING_ratio=df["COUGHING"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=COUGHING_ratio.index,y=COUGHING_ratio.values)
    plt.xticks(["1","2"], ['NORMAL', 'COUGHING'],rotation=45,ha="right")
    plt.title("COUGHING CHART")
    plt.xlabel("COUGHING")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/coughing_chart.png")
    plt.show()

    SHORTNESS_OF_BREATH_ratio=df["SHORTNESS OF BREATH"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=SHORTNESS_OF_BREATH_ratio.index,y=SHORTNESS_OF_BREATH_ratio.values)
    plt.xticks(["1","2"], ['NORMAL BREATH', 'SHORTNESS OF BREATH'],rotation=45,ha="right")
    plt.title("SHORTNESS OF BREATH CHART")
    plt.xlabel("SHORTNESS OF BREATH")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/shortness_of_breath_chart.png")
    plt.show()

    SWALLOWING_DIFFICULTY_ratio=df["SWALLOWING DIFFICULTY"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=SWALLOWING_DIFFICULTY_ratio.index,y=SWALLOWING_DIFFICULTY_ratio.values)
    plt.xticks(["1","2"], ['NORMAL SWALLOWING', 'SWALLOWING DIFFICULTY'],rotation=45,ha="right")
    plt.title("SWALLOWING DIFFICULTY CHART")
    plt.xlabel("SWALLOWING DIFFICULTY")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/shallowing_difficulty_chart.png")
    plt.show()

    CHEST_PAIN_ratio=df["CHEST PAIN"].value_counts()
    plt.figure(figsize=(7,7))
    sb.barplot(x=CHEST_PAIN_ratio.index,y=CHEST_PAIN_ratio.values)
    plt.xticks(["1","2"], ['NO CHEST PAIN', 'HAVE CHEST PAIN'],rotation=45,ha="right")
    plt.title("CHEST PAIN CHART")
    plt.xlabel("CHEST PAIN")
    plt.ylabel("NUMBER OF PEOPLE")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/chest_pain_chart.png")
    plt.show()

# DETAILED DATA ANALYSIS
#----------------------------------------------------
def detailed_analysis(df):

    cancervsage = df.groupby("AGE")["LUNG_CANCER"].value_counts().reset_index(name="count")
    plt.figure(figsize=(12,6))
    sb.lineplot(data=cancervsage,x="AGE",y="count",hue="LUNG_CANCER")
    plt.title("Cancer Case Distribution by Gender")
    plt.xlabel("Age")
    plt.ylabel("Number of Lung Cancer Case")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/cancer_vs_age_analaysis.png")
    plt.show()

    plt.figure(figsize=(12,6))
    smoking={1: 'Do Not Smoking', 2: 'Smoking'}
    df["Smoking Case"]=df["SMOKING"].map(smoking)
    agevssmoking = df.groupby("AGE")["Smoking Case"].value_counts().reset_index(name="count")
    sb.barplot(data=agevssmoking,x="AGE",y="count",hue="Smoking Case")
    plt.title("Smoking Distribution by Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Smoking Case")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/smoking_vs_age_analaysis.png")
    plt.show()


    plt.figure(figsize=(12,6))
    anxiety={1: 'Normal Mentality', 2: 'Have Anxiety'}
    df["Anxiety Case"]=df["ANXIETY"].map(anxiety)
    agevssmoking = df.groupby("AGE")["Anxiety Case"].value_counts().reset_index(name="count")
    sb.barplot(data=agevssmoking,x="AGE",y="count",hue="Anxiety Case")
    plt.title("Anxiety Case Distribution by Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Anxiety Case")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/anxiety_vs_age_analaysis.png")
    plt.show()

    plt.figure(figsize=(12,6))
    alcohol={1: 'No Alcohol', 2: 'Consuming Alcohol'}
    df["Alcohol Consuming Case"]=df["ALCOHOL CONSUMING"].map(alcohol)
    agevssmoking = df.groupby("AGE")["Alcohol Consuming Case"].value_counts().reset_index(name="count")
    sb.barplot(data=agevssmoking,x="AGE",y="count",hue="Alcohol Consuming Case")
    plt.title("Alcohol Consuming Case Distribution by Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Alcohol Consuming Case")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/alcohol_vs_age_analaysis.png")
    plt.show()

# MAIN PIPELINE
#----------------------------------------------------
def main():
    df=load_data()
    inspect_data(df)
    df=clear_data(df)
    basic_anaylsis(df)
    detailed_analysis(df)

# RUN
#----------------------------------------------------
if __name__=="__main__":
    main()