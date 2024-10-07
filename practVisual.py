import matplotlib.pyplot as plt
import re 
import seaborn as sns
import pandas as pd 
import numpy as np

df=pd.read_csv("test.csv")

df.info()
df.describe()

df=df.drop_duplicates()
df=df.fillna(0) 

# practice the regex on the column Name (create three new columns from it )
parentheses_pattern=re.compile(r"\( (.*?) \)")

df[["LastName","Rest"]]=df["Name"].str.split(", ",expand=True)
df["FullName"]=df["Rest"].apply(lambda x :parentheses_pattern.search(x).group(1) if parentheses_pattern.search(x) else None) 
df["Title"]=df["Rest"].str.replace(r"\(.*\)","",regex=True).str.strip()
df.drop(columns=['Rest'],inplace= True)


df["FamilySize"]=df["SibSp"]+df["Parch"]

# save to csv 
df.to_csv("CleanedTitanicData.csv",index=False)

# columns that have a correlation relationship //only numeric columns
numeric_dt = df[['Age','Fare','Pclass']].select_dtypes(include=[float,int])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_dt.corr(),annot=True,cmap="YlGnBu")
plt.show()


sns.jointplot(x="Fare",y="Age",data=df, kind="reg")
plt.show()

grouped_class=df.groupby('Pclass')['Fare'].sum().reset_index()
sns.barplot(x=grouped_class['Pclass'],y=grouped_class['Fare'])
plt.xlabel('Pclass')
plt.ylabel('Sum of Fare')
plt.title(" Sum of Fare per class")
plt.show()

sns.set_style('whitegrid')
embarked=df['Embarked'].value_counts()
x=embarked.values
y=embarked.index
plt.pie(x,labels=y,autopct='%1.1f%%',startangle=90,colors=sns.color_palette('pastel'))
plt.title("Demographics of Passengers ")
plt.show()

#  subplots 
mean_age=np.mean(df['Age'])
min_age=np.min(df['Age'])
max_age=np.max(df['Age'])
mean_member=np.mean(df['FamilySize'])

fig,axs= plt.subplots(1,2,figsize=(10,8))

axs[0].hist(df['Age'],bins=60)
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Values')
axs[0].axhline(mean_age,color="green",linestyle='--',label=f"Mean:{mean_age:.2f} ")
axs[0].axhline(min_age,color="red",linestyle='--',label=f"Min:{min_age} ")
axs[0].axhline(max_age,color="orange",linestyle='--',label=f"Max:{max_age} ")
axs[0].set_title('Distribution of Age')
axs[0].legend()

axs[1].hist(df['FamilySize'],bins=60)
axs[1].set_xlabel('FamilySize')
axs[1].set_ylabel("Values")
axs[1].set_title(" Size of Family Member ")
axs[1].axhline(mean_member,color="green",linestyle="--",label=f"Mean_family:{mean_member:.2f}")
axs[1].legend()

plt.show()


