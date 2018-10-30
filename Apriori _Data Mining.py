
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[2]:


df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')


# In[3]:


df.head()


# In[4]:


df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)


# In[5]:


df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]


# In[6]:


basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


# In[7]:


basket.head()


# In[8]:


basket.iloc[:,[0,1,2,3,4,5,6,7]].head()


# In[9]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# In[10]:


basket_sets = basket.applymap(encode_units)


# In[11]:


basket_sets.drop('POSTAGE', inplace=True, axis=1)


# In[12]:


basket_sets.head()


# In[13]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)


# In[14]:


frequent_itemsets.head()


# In[15]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


# In[16]:


rules


# In[17]:


rules[ (rules['lift'] >= 7) &
       (rules['confidence'] >= 0.7) ]


# In[18]:


basket['ALARM CLOCK BAKELIKE PINK'].sum()


# In[19]:


basket['ALARM CLOCK BAKELIKE IVORY'].sum()


# In[20]:


basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


# In[21]:


basket_sets2 = basket2.applymap(encode_units)


# In[22]:


basket_sets2.drop('POSTAGE', inplace=True, axis=1)


# In[23]:


frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)


# In[24]:


rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2


# In[25]:


rules2[ (rules2['lift'] >= 5) &
        (rules2['confidence'] >= 0.6) ]

