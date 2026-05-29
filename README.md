## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
NAME: SRIRAM.S

REGNO: 212225240155

```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
print(df)
```

<img width="579" height="233" alt="{34D3F069-7140-4F0D-8C4B-1CCA815BC03C}" src="https://github.com/user-attachments/assets/099d23f8-fa8a-4392-8585-1604d850449c" />


```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```

<img width="176" height="205" alt="{A70B5AD8-557F-4E77-8AAE-AF4E703B8DA9}" src="https://github.com/user-attachments/assets/0bb9f9f8-18be-45c4-a369-dbae4f56e1fb" />


```
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
```

<img width="505" height="326" alt="{CD615ADC-ACFE-47E8-81CF-CDB22CB8E43C}" src="https://github.com/user-attachments/assets/5ef96a0c-f895-4d93-b3cb-e659649bdf77" />


```
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```

<img width="453" height="326" alt="{3F3E6351-4AA7-4C56-A3E2-A8EDCBB68C49}" src="https://github.com/user-attachments/assets/f26801dd-ef08-4220-b0e5-df345ea68efe" />


```
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```

<img width="450" height="322" alt="{75E7694A-DCBD-4E6E-AA9E-DE40D0F6F108}" src="https://github.com/user-attachments/assets/ea0901f8-d015-40dc-9efc-b2eee1ae93ff" />


```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```


<img width="560" height="328" alt="{829A9EBF-420E-4DD6-8A5D-8308E6239398}" src="https://github.com/user-attachments/assets/04f8dd04-1eae-4747-bf68-37beece6bed3" />



```
pd.get_dummies(df,columns=['City'])
```

<img width="790" height="329" alt="{9C0303F5-BD10-4418-8362-FFC7342DADF4}" src="https://github.com/user-attachments/assets/a40f419f-4899-497b-b998-138f20282cf2" />


```
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
nd
```

<img width="253" height="319" alt="{6925B0B0-BC37-4D98-B1E1-4DA8374694B6}" src="https://github.com/user-attachments/assets/2d7b8712-8e78-4fe6-a992-ace463be0b9a" />


```
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```

<img width="522" height="320" alt="{03090596-2EAA-4255-80C1-4F9073BF23D8}" src="https://github.com/user-attachments/assets/f7b9c4cd-67ab-4653-bb72-bac0eb938b25" />


```
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
```


<img width="437" height="322" alt="{807D8D9C-BCE0-4108-8348-1358C305A984}" src="https://github.com/user-attachments/assets/371dbf9c-9d9f-4fcb-a93a-50badfc0f66b" />


```
df = pd.read_csv('Data_to_Transform.csv')
df
```


<img width="706" height="381" alt="{F393738D-435E-4156-B68D-ECD68873033F}" src="https://github.com/user-attachments/assets/687733fe-4fed-4b89-8db0-f187d4857313" />



```
df.skew()
```

<img width="323" height="104" alt="{431D3D0A-6FB7-4DAA-B39A-D90935ED3246}" src="https://github.com/user-attachments/assets/0d417d59-3b68-467a-b848-bf788cbdadc6" />



```
np.log(df["Highly Positive Skew"])
```


<img width="517" height="240" alt="{E60B980F-57EB-4015-9CAE-D559D2D577FC}" src="https://github.com/user-attachments/assets/5952a32b-b5ce-48f9-a51c-d1a89272f5f7" />


```
np.reciprocal(df["Moderate Positive Skew"])
```


<img width="570" height="236" alt="{F4B9723A-A7E5-4A38-A8B4-5622FC8CCECE}" src="https://github.com/user-attachments/assets/d3210899-55b1-4d3b-9768-8d104b26047b" />



```
np.sqrt(df["Highly Positive Skew"])
```

<img width="527" height="241" alt="{F4A53E1E-9C6B-4505-9C4B-D54A60E52583}" src="https://github.com/user-attachments/assets/bd1a97b9-a84e-4ca5-84e2-d2a1cd04f2d2" />


```
np.square(df["Highly Positive Skew"])
```

<img width="530" height="237" alt="{425D75C2-8C24-4DA8-9285-FCA4F404103B}" src="https://github.com/user-attachments/assets/6f16c443-24b3-4181-8501-e5f9e95b6f60" />


```
from scipy import stats
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="912" height="399" alt="{1AC66548-8DAA-43F5-9016-D7EA4307BD5B}" src="https://github.com/user-attachments/assets/91ab0f6f-09fd-48ea-b249-89c7069629ed" />


```
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```


<img width="1129" height="413" alt="{96064E40-BA5A-40E0-9C3D-553F02D954BD}" src="https://github.com/user-attachments/assets/4042689a-a2a4-4606-97f9-f49f4b59e334" />


```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```


<img width="1121" height="412" alt="{C4F58A63-C8AC-45E5-9F25-D8DE369522BB}" src="https://github.com/user-attachments/assets/b653c8f7-dc97-451c-8e34-889f31b7ca0b" />



```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```


<img width="823" height="472" alt="{A463B02D-350E-4E27-B9E3-885D0569879B}" src="https://github.com/user-attachments/assets/678e9cf9-f1fb-4dcd-9a4e-8e940657fd8d" />



```
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```


<img width="716" height="496" alt="{10FBCFBF-F941-442D-9EA7-8BCE4240D9AB}" src="https://github.com/user-attachments/assets/86acbea7-ca3e-4757-afd1-d5cae450c108" />


```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```

<img width="756" height="492" alt="{CAD9EA8A-CF99-4952-8F5F-6A1851701103}" src="https://github.com/user-attachments/assets/d090d82e-cab4-4673-a08c-a67ddf348f78" />



```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```

<img width="790" height="485" alt="{5EF97BF2-DAA7-4339-B312-45EF2AB65BBC}" src="https://github.com/user-attachments/assets/87d10e39-f7e5-439c-9ee4-a147289e7153" />



```
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```

<img width="742" height="485" alt="{68E3EFF8-3FC8-4CE3-9423-DF6BA677AD23}" src="https://github.com/user-attachments/assets/77afcaee-136b-4683-a2b6-7b0d012deff4" />



```
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```


<img width="711" height="479" alt="{4CCA8E94-3258-4164-A148-71576DDE89A6}" src="https://github.com/user-attachments/assets/0eeda47b-1e9a-4d44-8e26-07a7764de71f" />



```
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```


<img width="725" height="478" alt="{DE9A6DE7-61C6-474B-91F9-529168A416B0}" src="https://github.com/user-attachments/assets/9b7fb51e-340b-4667-bb5f-3320a139dc8b" />



```
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```


<img width="700" height="473" alt="{22D0BBD6-583E-496B-9B27-60DBFEDA9A12}" src="https://github.com/user-attachments/assets/db3d965d-6a7a-4b18-9f83-cd6b60170a4f" />


```
pd.concat([CC,new],axis = 1)
```


<img width="521" height="334" alt="{CACEF928-45B3-4500-80E3-29B8288460AD}" src="https://github.com/user-attachments/assets/74b43312-164f-4b03-bcf0-cffc51e1fdc8" />

# RESULT:
Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.
 

       
