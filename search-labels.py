import pandas as pd

df = pd.read_csv('file-names/image-labels.csv')
df.drop(df.loc[df['Organ'] != 'Upper GI'].index, inplace=True)
df.drop(df.loc[df['Classification'] != 'anatomical-landmarks'].index, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)
print(df['Finding'].value_counts())
df.to_csv('file-names/filtered-names/image-labels.csv')



