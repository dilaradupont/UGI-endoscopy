# import pandas as pd

# df = pd.read_csv('file-names/image-labels.csv')
# df.drop(df.loc[df['Organ'] != 'Upper GI'].index, inplace=True)
# df.drop(df.loc[df['Classification'] != 'anatomical-landmarks'].index, inplace=True)
# df.reset_index(drop=True, inplace=True)
# print(df.shape)
# print(df['Finding'].value_counts())
# df.to_csv('file-names/filtered-names/image-labels.csv')

# lis1 = []
# df = pd.read_csv('file-names/video-labels.csv')
# df.drop(df.loc[df['Organ'] != 'Upper GI'].index, inplace=True)
# df.reset_index(drop=True, inplace=True)
# print(df.shape)
# for i in range(df.shape[0]):
#     if (df['Classification 1'].iloc[i] != 'Anatomical landmarks') \
#         & (df['Classification 2'].iloc[i] != 'Anatomical landmarks'):
#             lis1.append(i)
# df.drop(df.index[lis1], inplace=True)
# df.reset_index(drop=True, inplace=True)
# print(df.shape)
# print(df['Finding 1'].value_counts())
# print(df['Finding 2'].value_counts())
# df.to_csv('file-names/filtered-names/video-labels.csv')

from PIL import Image
from PIL.ExifTags import TAGS
  
# open the image
image = Image.open("pylorus/0a2ed767-ac9e-4922-b25c-9eb3dae85f66.jpg")
  
# extracting the exif metadata
exifdata = image.getexif()
  
# looping through all the tags present in exifdata
for tagid in exifdata:
      
    # getting the tag name instead of tag id
    tagname = TAGS.get(tagid, tagid)
  
    # passing the tagid to get its respective value
    value = exifdata.get(tagid)
    
    # printing the final result
    print(f"{tagname:25}: {value}")



