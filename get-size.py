from PIL import Image
import pandas as pd

def image_names_to_df(row1, row2):
    df = pd.read_csv('file-names/filtered-names/image-labels.csv')
    df = df.iloc[row1:row2, 1:2]
    image_list = df['Video file'].tolist()
    df.to_csv('test-files/image-files.csv')
    return image_list, df

def get_size(image_list, df1, landmark):
    w = []
    h = []

    for i in range(len(image_list)):
        im = Image.open(f'images/anatomical-landmarks/{landmark}/{image_list[i]}.jpg')
        w.append(im.size[0])
        h.append(im.size[1])
    df_size = pd.DataFrame(list(zip(df1['Video file'], w, h)), columns =['Video file', 'Width', 'Height'])
    return df_size

def main():
    image_list, df1 = image_names_to_df(0, 999)
    df_size1 = get_size(image_list, df1, 'pylorus')
    image_list, df2 = image_names_to_df(999, 1763)
    df_size2 = get_size(image_list, df2, 'retroflex-stomach')
    image_list, df3 = image_names_to_df(1763, 2695)
    df_size3 = get_size(image_list, df3, 'z-line')
    df_size = pd.concat([df_size1, df_size2, df_size3], axis=0, join="outer")
    df_size.reset_index(drop=True, inplace=True)
    df_size.to_csv('test-files/image-size.csv')

if __name__ == "__main__":
    main()


