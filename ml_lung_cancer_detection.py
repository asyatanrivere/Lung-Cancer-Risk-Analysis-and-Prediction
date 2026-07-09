from lung_cancer_data_analysis import clear_data
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree

# LOAD & CLEAR DATA
df= pd.read_csv("dataset/survey lung cancer.csv")
df=clear_data(df)

# EDIT THE DATAFRAME
d_sex={"M":1,"F":2}
df["GENDER"]=df["GENDER"].map(d_sex)


features=["GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY","CHRONIC DISEASE","FATIGUE","ALLERGY","WHEEZING","ALCOHOL CONSUMING","COUGHING","SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]

x=df[features]
y=df["LUNG_CANCER"]

# TEST & TRAIN GROUPS
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

# DEFINE AND TRAIN THE MODEL
dtree=DecisionTreeClassifier()
dtree=dtree.fit(x_train,y_train)

# MAKE MODEL PREDICT
y_predict=dtree.predict(x_test)

# DECISION TREE PLOT
plt.figure(figsize=(15,9))
plot_tree(dtree,feature_names=features)
plt.savefig("images/decision_tree_plot.png")

# CONFUSION MATRIX
c_matrix=metrics.confusion_matrix(y_test,y_predict)

cm=metrics.ConfusionMatrixDisplay(confusion_matrix=c_matrix)
cm.plot()
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix_plot.png")
plt.show()

# ACCURACY & PRECISION & F1 SCORE
print(f"Accuracy: {metrics.accuracy_score(y_test,y_predict)}")
# Accuracy: 0.8928571428571429
print(f"Precision: {metrics.precision_score(y_test,y_predict,average='macro')}")
# Precision: 0.94
print(f"F1 Score: {metrics.f1_score(y_test,y_predict,average='macro')}")
# F1 Score: 0.801418439716312

