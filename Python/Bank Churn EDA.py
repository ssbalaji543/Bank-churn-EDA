import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



"DATA COLLECTING"

df = pd.read_csv("C:\\Users\\Welcome\\Documents\\Projects(Data Analysis)\\Project 2- Bank churn EDA\\Dataset\\Bank Churn Dataset.csv")



"DATA UNDERSTANDING"

#print(df.info())
#print(df.head(10))
#print(df.tail(10))
#print(df.describe())
#print(df.duplicated().sum())

#print(df.nunique())
#print(df["Gender"].unique())
#print(df["Geography"].unique())



"DATA TRANSFORMING"

df.drop(columns = ["RowNumber","Surname"], inplace = True)
#print(df.columns)

cols = ["CustomerId", "CreditScore", "Age", "Tenure",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "Exited"]

df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
df["Geography"] = df["Geography"].astype("string")
df["Gender"] = df["Gender"].astype("string")

#print(df.dtypes)



"DATA ANALYSING"

#Overall Churn Count
churn_counts = df['Exited'].value_counts().sort_index()
#print(churn_counts, end='\n\n')


#Churn By Geography
churn_geo = pd.pivot_table(
    df,
    index='Geography',
    columns='Exited',
    aggfunc='size'
)

churn_geo['Total'] = churn_geo.sum(axis=1)
churn_geo['Churn_Rate_%'] = (churn_geo[1] / churn_geo['Total']) * 100
churn_geo['Retention_Rate_%'] =np.where(churn_geo[0]==0, 0, (churn_geo[0] / churn_geo['Total']) * 100)

#print(churn_geo.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Gender
churn_gen = pd.pivot_table(
    df,
    index='Gender',
    columns='Exited',
    aggfunc='size'
)

churn_gen['Total'] = churn_gen.sum(axis=1)
churn_gen['Churn_Rate_%'] = (churn_gen[1] / churn_gen['Total']) * 100
churn_gen['Retention_Rate_%'] =np.where(churn_gen[0]==0, 0, (churn_gen[0] / churn_gen['Total']) * 100)

#print(churn_gen.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Credit Score
churn_crsco = pd.pivot_table(
    df,
    index=pd.cut(
        df['CreditScore'],
        bins=[0, 400, 500, 600, 700, 800, 1000],
        labels=['<400', '400–499', '500–599', '600–699', '700–799', '800+'],
        right=False
    ),
    columns='Exited',
    aggfunc='size',
    observed=False
)

churn_crsco['Total'] = churn_crsco.sum(axis=1)
churn_crsco['Churn_Rate_%'] = (churn_crsco[1] / churn_crsco['Total']) * 100
churn_crsco['Retention_Rate_%'] =np.where(churn_crsco[0]==0, 0, (churn_crsco[0] / churn_crsco['Total']) * 100)

#print(churn_crsco.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Age
churn_age = pd.pivot_table(
    df,
    index=pd.cut(
        df['Age'],
        bins=[0, 25, 35, 45, 55, 65, 75, 85, 100],
        labels=['<25', '25–34', '35–44', '45–54', '55–64', '65-74', '75-84', '85+'],
        right=False
    ),
    columns='Exited',
    aggfunc='size',
    observed=False
)

churn_age['Total'] = churn_age.sum(axis=1)
churn_age['Churn_Rate_%'] = (churn_age[1] / churn_age['Total']) * 100
churn_age['Retention_Rate_%'] =np.where(churn_age[0]==0, 0, (churn_age[0] / churn_age['Total']) * 100)

#print(churn_age.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Tenure
churn_tenure = pd.pivot_table(
    df,
    index='Tenure',
    columns='Exited',
    aggfunc='size'
)

churn_tenure['Total'] = churn_tenure.sum(axis=1)
churn_tenure['Churn_Rate_%'] = (churn_tenure[1] / churn_tenure['Total']) * 100
churn_tenure['Retention_Rate_%'] =np.where(churn_tenure[0]==0, 0, (churn_tenure[0] / churn_tenure['Total']) * 100)

#print(churn_tenure.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Number Of Products
churn_NOP = pd.pivot_table(
    df,
    index='NumOfProducts',
    columns='Exited',
    aggfunc='size',
    fill_value=0
)

churn_NOP['Total'] = churn_NOP.sum(axis=1)
churn_NOP['Churn_Rate_%'] = (churn_NOP[1] / churn_NOP['Total']) * 100
churn_NOP['Retention_Rate_%'] =np.where(churn_NOP[0]==0, 0, (churn_NOP[0] / churn_NOP['Total']) * 100)

#print(churn_NOP.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Credit Card
churn_crcard = pd.pivot_table(
    df,
    index='HasCrCard',
    columns='Exited',
    aggfunc='size'
)

churn_crcard['Total'] = churn_crcard.sum(axis=1)
churn_crcard['Churn_Rate_%'] = (churn_crcard[1] / churn_crcard['Total']) * 100
churn_crcard['Retention_Rate_%'] =np.where(churn_crcard[0]==0, 0, (churn_crcard[0] / churn_crcard['Total']) * 100)

#print(churn_crcard.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Active Member
churn_active = pd.pivot_table(
    df,
    index='IsActiveMember',
    columns='Exited',
    aggfunc='size'
)

churn_active['Total'] = churn_active.sum(axis=1)
churn_active['Churn_Rate_%'] = (churn_active[1] / churn_active['Total']) * 100
churn_active['Retention_Rate_%'] =np.where(churn_active[0]==0, 0, (churn_active[0] / churn_active['Total']) * 100)

#print(churn_active.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Account Balance
churn_bal = pd.pivot_table(
    df,
    index=pd.cut(
        df['Balance'],
        bins=[0, 1, 50000, 100000, 150000, 200000, 300000],
        labels=['0', '1–50K', '50K–1L', '1L–1.5L', '1.5L–2L', '2L+'],
        right=False
    ),
    columns='Exited',
    aggfunc='size',
    observed=False
)

churn_bal['Total'] = churn_bal.sum(axis=1)
churn_bal['Churn_Rate_%'] = (churn_bal[1] / churn_bal['Total']) * 100
churn_bal['Retention_Rate_%'] =np.where(churn_bal[0]==0, 0, (churn_bal[0] / churn_bal['Total']) * 100)

#print(churn_bal.sort_values('Churn_Rate_%', ascending=False), end='\n\n')


#Churn By Estimated Salary
churn_salary = pd.pivot_table(
    df,
    index=pd.cut(
        df['EstimatedSalary'],
        bins=[1, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000],
        labels=['1–25K', '25K-50K', '50K–75K',  '75K-1L', '1L–1.25L', '1.25L–1.5L', '1.5L-1.75L', '1.75L-2L'],
        right=False
    ),
    columns='Exited',
    aggfunc='size',
    observed=False
)

churn_salary['Total'] = churn_salary.sum(axis=1)
churn_salary['Churn_Rate_%'] = (churn_salary[1] / churn_salary['Total']) * 100
churn_salary['Retention_Rate_%'] =np.where(churn_salary[0]==0, 0, (churn_salary[0] / churn_salary['Total']) * 100)

#print(churn_salary.sort_values('Churn_Rate_%', ascending=False), end='\n\n')



"DATA VISUALISING"

#Overall Churn Rate
plt.pie(
    churn_counts,
    labels=['Retention', 'Churn'],
    colors=['#4caf50', '#f44336'],
    autopct=lambda p: f'{int(round(p*churn_counts.sum()/100))}\n({p:.1f}%)',
    startangle=90,
    labeldistance=1.15,
    pctdistance=0.75,
    wedgeprops={'width': 0.4,
                'edgecolor': 'white',
                'linewidth': 2}
)

plt.title('Overall Customer Churn Rate')
plt.show()


#Churn Rate by Categorical Features
plt.figure(figsize=(10,6))

#Subplot 1
plt.subplot(2,2,1)
plt.pie(
    churn_gen['Churn_Rate_%'],
    labels=churn_gen.index,
    colors=['#fc03b1', '#03d3fc'],
    autopct='%1.1f%%',
    startangle=90,
    labeldistance=1.15,
    pctdistance=0.75,
    wedgeprops={'width': 0.4,
                'edgecolor': 'white',
                'linewidth': 2}
)

plt.title('Gender-wise Churn Rate')

#Subplot 2
plt.subplot(2,2,2)
plt.pie(
    churn_geo['Churn_Rate_%'],
    labels=churn_geo.index,
    colors=['#66dbf2', '#f266d1', '#db66f2'],
    autopct='%1.1f%%',
    startangle=90,
    labeldistance=1.15,
    pctdistance=0.75,
    wedgeprops={'width': 0.4,
                'edgecolor': 'white',
                'linewidth': 2}
)

plt.title('Geography-wise Churn Rate')

#Subplot 3
plt.subplot(2,2,3)
plt.pie(
    churn_active['Churn_Rate_%'],
    labels=['Inactive','Active'],
    colors=['#029bfa', '#fa025d'],
    autopct='%1.1f%%',
    startangle=90,
    labeldistance=1.15,
    pctdistance=0.75,
    wedgeprops={'width': 0.4,
                'edgecolor': 'white',
                'linewidth': 2}
)

plt.title('Active vs Inactive Member Churn Rate')

#Subplot 4
plt.subplot(2,2,4)
plt.pie(
    churn_crcard['Churn_Rate_%'],
    labels=['No','Yes'],
    colors=['#66baf2', '#f266e2'],
    autopct='%1.1f%%',
    startangle=90,
    labeldistance=1.15,
    pctdistance=0.75,
    wedgeprops={'width': 0.4,
                'edgecolor': 'white',
                'linewidth': 2}
)

plt.title('Has Creditcard Churn Rate')

#Overall Title
plt.suptitle("Churn Rate of Categorical Features", fontsize=16, fontweight='bold')
#Adjusting Subplots
plt.subplots_adjust(
    left=0.15,
    right=0.85,
    bottom=0,
    top=0.8,
    wspace=0,
    hspace=0.25
)

plt.show()


#Churn Rate by Numerical Features
plt.figure(figsize=(18,10))

#Subplot 1
plt.subplot(2,3,1)
sns.barplot(
    x=churn_crsco.index,
    y=churn_crsco['Churn_Rate_%'],
    order=churn_crsco.sort_values(
        'Churn_Rate_%',
        ascending=False
    ).index,
    color="#8b02fa",
    edgecolor="black"
)
plt.title("Churn Rate by Credit Score")
plt.ylabel("Churn Rate (%)")
plt.xlabel(" ")
plt.xticks(rotation=45)

#Subplot 2
plt.subplot(2,3,2)
sns.barplot(
    x=churn_age.index,
    y=churn_age['Churn_Rate_%'],
    order=churn_age.sort_values(
        'Churn_Rate_%',
        ascending=False
    ).index,
    color="#8b02fa",
    edgecolor="black"
)
plt.title("Churn Rate by Age")
plt.ylabel("Churn Rate (%)")
plt.xlabel(" ")
plt.xticks(rotation=45)

#Subplot 3
plt.subplot(2,3,3)
sns.barplot(
    x=churn_tenure.index,
    y=churn_tenure['Churn_Rate_%'],
    order=churn_tenure.sort_values(
        'Churn_Rate_%',
        ascending=False
    ).index,
    color="#8b02fa",
    edgecolor="black"
)
plt.title("Churn Rate by Tenure")
plt.ylabel("Churn Rate (%)")
plt.xlabel(" ")
plt.xticks()

#Subplot 4
plt.subplot(2,3,4)
sns.barplot(
    x=churn_bal.index,
    y=churn_bal['Churn_Rate_%'],
    order=churn_bal.sort_values(
        'Churn_Rate_%',
        ascending=False
    ).index,
    color="#8b02fa",
    edgecolor="black"
)
plt.title("Churn Rate by Account Balance")
plt.ylabel("Churn Rate (%)")
plt.xlabel(" ")
plt.xticks(rotation=45)

#Subplot 5
plt.subplot(2,3,5)
sns.barplot(
    x=churn_NOP.index,
    y=churn_NOP['Churn_Rate_%'],
    order=churn_NOP.sort_values(
        'Churn_Rate_%',
        ascending=False
    ).index,
    color="#8b02fa",
    edgecolor="black"
)
plt.title("Churn Rate by No of Products")
plt.ylabel("Churn Rate (%)")
plt.xlabel(" ")
plt.xticks()

#Subplot 6
plt.subplot(2,3,6)
sns.barplot(
    x=churn_salary.index,
    y=churn_salary['Churn_Rate_%'],
    order=churn_salary.sort_values(
        'Churn_Rate_%',
        ascending=False
    ).index,
    color="#8b02fa",
    edgecolor="black"
)
plt.title("Churn Rate by Salary")
plt.ylabel("Churn Rate (%)")
plt.xlabel(" ")
plt.xticks(rotation=45)

#Overall Title
plt.suptitle("Churn Rate of Numerical Features", fontsize=16, fontweight='bold')
#Adjusting Subplots
plt.subplots_adjust(
    left=0.07,
    right=0.97,
    bottom=0.12,
    top=0.885,
    wspace=0.3,
    hspace=0.495
)

plt.show()


#Distribution of Categorical Features
plt.figure(figsize=(18,10))

#Subplot 1
plt.subplot(1,4,1)
sns.countplot(x='Gender',
              data=df,
              color='#b50d3f',
              edgecolor='black',
              linewidth=1)
plt.title("Distribution of Gender")
plt.xlabel("")
plt.ylabel("Count")

#Subplot 2
plt.subplot(1,4,2)
sns.countplot(x='Geography',
              data=df,
              color='#b50d3f',
              edgecolor='black',
              linewidth=1)
plt.title("Distribution of Geography")
plt.xlabel("")
plt.ylabel("Count")

#Subplot 3
plt.subplot(1,4,3)
sns.countplot(x='HasCrCard',
              data=df,
              color='#b50d3f',
              edgecolor='black',
              linewidth=1)
plt.title("Distribution of HasCrCard")
plt.xlabel("")
plt.xticks([0, 1], ['No', 'Yes'])
plt.ylabel("Count")

#Subplot 4
plt.subplot(1,4,4)
sns.countplot(x='IsActiveMember',
              data=df,
              color='#b50d3f',
              edgecolor='black',
              linewidth=1)
plt.title("Distribution of IsActiveMember")
plt.xlabel("")
plt.xticks([0, 1], ['No', 'Yes'])
plt.ylabel("Count")

#Overall Title
plt.suptitle("Distributions of Categorical Features", fontsize=16, fontweight='bold')
#Adjusting Subplots
plt.subplots_adjust(
    left=0.07,
    right=0.95,
    bottom=0.20,
    top=0.75,
    wspace=0.9,
    hspace=0.3
)

plt.show()


#Distributions of Numerical Features
plt.figure(figsize=(18,10))

#Subplot 1
plt.subplot(2,3,1)
sns.histplot(df['CreditScore'],
             bins=20,
             kde=True,
             stat='density',
             color='green',
             linewidth=0.5,
             line_kws={'color':'darkgreen', 'linewidth':1},
             kde_kws={'bw_adjust': 1.4})
plt.title("Distribution of Credit Score")
plt.xlabel("")
plt.ylabel("Density")

#Subplot 2
plt.subplot(2,3,2)
sns.histplot(df['Age'],
             bins=20,
             kde=True,
             stat='density',
             color='red',
             linewidth=0.5,
             line_kws={'color':'darkred', 'linewidth':1},
             kde_kws={'bw_adjust': 1.4})
plt.title("Distribution of Age")
plt.xlabel("")
plt.ylabel("Density")

#Subplot 3
plt.subplot(2,3,3)
sns.countplot(x='Tenure',
              data=df,
              color='orange',
              edgecolor='black',
              linewidth=0.5)
plt.title("Distribution of Tenure")
plt.xlabel("")
plt.ylabel("Count")

#Subplot 4
plt.subplot(2,3,4)
sns.histplot(df['Balance'],
             bins=20,
             kde=True,
             stat='density',
             color='purple',
             linewidth=0.5,
             line_kws={'color':'darkpurple', 'linewidth':1},
             kde_kws={'bw_adjust': 1.4})
plt.title("Distribution of Account Balance")
plt.xlabel("")
plt.ylabel("Density")

#Subplot 5
plt.subplot(2,3,5)
sns.histplot(df['EstimatedSalary'],
             bins=20,
             kde=True,
             stat='density',
             color='blue',
             linewidth=0.5,
             line_kws={'color':'darkblue', 'linewidth':1},
             kde_kws={'bw_adjust': 1.4})
plt.title("Distribution of Salary")
plt.xlabel("")
plt.ylabel("Density")

#Subplot 6
plt.subplot(2,3,6)
sns.countplot(x='NumOfProducts',
              data=df,
              color='yellow',
              edgecolor='black',
              linewidth=0.5)
plt.title("Distribution of NumOfProducts")
plt.xlabel("")
plt.ylabel("Count")

#Overall Title
plt.suptitle("Distributions of Numerical Features", fontsize=16, fontweight='bold')
#Adjusting Subplots
plt.subplots_adjust(
    left=0.07,
    right=0.97,
    bottom=0.05,
    top=0.85,
    wspace=0.3,
    hspace=0.3
)

plt.show()


#Churn and Retention Rate by Categorical Features(using Stackedbar plot)
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

# Colors
retained_color = "#4caf50"
churned_color = "#f44336"

plots = [
    (churn_gen, "Churn Rate by Gender"),
    (churn_geo, "Churn Rate by Geography"),
    (churn_crcard, "Churn Rate by Credit Card"),
    (churn_active, "Churn Rate by Active Member")
]

for ax, (data, title) in zip(axes, plots):
    ax.bar(
        data.index,
        data['Retention_Rate_%'],
        label='Retained',
        color=retained_color,
        edgecolor='black',
        linewidth=1
    )
    ax.bar(
        data.index,
        data['Churn_Rate_%'],
        bottom=data['Retention_Rate_%'],
        label='Churned',
        color=churned_color,
        edgecolor='black',
        linewidth=1
    )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Percentage")

    if set(data.index) == {0, 1}:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_xlabel("")
    else:
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index)
        ax.set_xlabel("")

# Common Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper right',
    ncol=1,
    bbox_to_anchor=(0.9, 0.95),
    fontsize=11,
    frameon=False
)

#Overall Title
fig.suptitle(
    "Churn and Retention Rate by Categorical Features",
    fontsize=16,
    fontweight='bold',
    y=0.98
)
#Adjusting Subplots
plt.subplots_adjust(
    left=0.15,
    right=0.85,
    bottom=0.05,
    top=0.8,
    wspace=0.3,
    hspace=0.35
)

plt.show()


#Churn and Retention Rate by Numerical Features(using Stackedbar plot)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Colors
retained_color = "#4caf50"
churned_color = "#f44336"

plots = [
    (churn_crsco, "Churn Rate by Credit Score", 45),
    (churn_age, "Churn Rate by Age", 45),
    (churn_tenure, "Churn Rate by Tenure", 0),
    (churn_bal, "Churn Rate by Account Balance", 45),
    (churn_NOP, "Churn Rate by No of Products", 0),
    (churn_salary, "Churn Rate by No of Products", 45)
]

for ax, (data, title, rot) in zip(axes, plots):
    ax.bar(
        data.index,
        data['Retention_Rate_%'],
        label='Retained',
        color=retained_color,
        edgecolor='black',
        linewidth=1
    )
    ax.bar(
        data.index,
        data['Churn_Rate_%'],
        bottom=data['Retention_Rate_%'],
        label='Churned',
        color=churned_color,
        edgecolor='black',
        linewidth=1
    )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=rot)
    

# Common Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper right',
    ncol=1,
    bbox_to_anchor=(0.95, 0.95),
    fontsize=11,
    frameon=False
)
#Overall Title
fig.suptitle(
    "Churn and Retention Rate by Numerical Features",
    fontsize=16,
    fontweight='bold',
    y=0.98
)
#Adjusting Subplots
plt.subplots_adjust(
    left=0.05,
    right=0.95,
    bottom=0.12,
    top=0.8,
    wspace=0.3,
    hspace=0.6
)

plt.show()


#Churn Distribution Across Customer Attributes
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

#Gender
#Subplot 1
sns.countplot(data=df[df["Exited"]==1],
              x='Gender',
              hue='HasCrCard',
              palette={1 :'#015257', 0 :'#ab0202'},
              ax=axes[0, 0],
              edgecolor='black',
              linewidth=1
)
axes[0, 0].set_title("Churn by Gender X Credit Card")

#Subplot 2
sns.countplot(data=df[df["Exited"]==1],
              x='Gender',
              hue='IsActiveMember',
              palette={1 :'#015257', 0 :'#ab0202'},
              ax=axes[0, 1],
              edgecolor='black',
              linewidth=1
)
axes[0, 1].set_title("Churn by Gender X Active")


#Geography
#Subplot 3
sns.countplot(data=df[df["Exited"]==1],
              x='Geography',
              hue='HasCrCard',
              palette={1 :'#015257', 0 :'#ab0202'},
              ax=axes[1, 0],
              edgecolor='black',
              linewidth=1
)
axes[1, 0].set_title("Churn by Geography X Credit Card")

#Subplot 4
sns.countplot(data=df[df["Exited"]==1],
              x='Geography',
              hue='IsActiveMember',
              palette={1 :'#015257', 0 :'#ab0202'},
              ax=axes[1, 1],
              edgecolor='black',
              linewidth=1
)
axes[1, 1].set_title("Churn by Geography X Active")


for ax in axes.flat:
    ax.legend_.remove()
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlabel('')

#Common Legend
handles1, labels1 = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles1,
    ['No', 'Yes'],
    title='Status',
    loc='upper right',
    bbox_to_anchor=(0.85, 0.95),
    ncol=1,
    frameon=False
)

#Overall Title
fig.suptitle(
    "Churn Distribution Across Customer Attributes",
    fontsize=16,
    fontweight='bold',
    y=0.98
)
#Adjusting Subplots
plt.subplots_adjust(
    left=0.15,
    right=0.85,
    bottom=0.05,
    top=0.8,
    wspace=0.3,
    hspace=0.4
)
plt.show()
