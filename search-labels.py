import pandas as pd

df = pd.read_csv('file-names/image-labels.csv')
df.drop(df.loc[df['Organ'] != 'Upper GI'].index, inplace=True)
df.drop(df.loc[df['Classification'] != 'anatomical-landmarks'].index, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)
print(df['Finding'].value_counts())
df.to_csv('file-names/filtered-names/image-labels.csv')

lis1 = []
df = pd.read_csv('file-names/video-labels.csv')
df.drop(df.loc[df['Organ'] != 'Upper GI'].index, inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)
for i in range(df.shape[0]):
    if (df['Classification 1'].iloc[i] != 'Anatomical landmarks') \
        & (df['Classification 2'].iloc[i] != 'Anatomical landmarks'):
            lis1.append(i)
df.drop(df.index[lis1], inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)
print(df['Finding 1'].value_counts())
print(df['Finding 2'].value_counts())
df.to_csv('file-names/filtered-names/video-labels.csv')
