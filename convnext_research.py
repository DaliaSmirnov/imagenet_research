import pandas as pd

df_resnet50 = pd.read_parquet('/Users/daliasmirnov/PycharmProjects/pythonProject/ConvneXt/prediction_df_resnet50.parq')
df_convnext = pd.read_parquet('/Users/daliasmirnov/PycharmProjects/pythonProject/ConvneXt/prediction_df.convnext.parq')

class_mapping = dict(zip(df_resnet50['class'], df_resnet50['description']))
df_convnext = df_convnext.sort_values(['img_path', 'probability'], ascending=False)
df_resnet50 = df_resnet50.sort_values(['img_path', 'probability'], ascending=False)


def compare_preds(img):
    true_class = list(df_convnext[df_convnext['img_path'].str.endswith(img)]['true_class'])[0]
    print('the real class is:',class_mapping[true_class])
    for i in range(10):
        convnext_preds = list(df_convnext[df_convnext['img_path'].str.endswith(img)]['description'])
        resnet50_preds = list(df_resnet50[df_resnet50['img_path'].str.endswith(img)]['description'])
        print(str(i) + ':  ' + 'convnext: ' + convnext_preds[i] +'   resnet50:' + resnet50_preds[i])

if __name__ == '__main__':
    while True:
        img_name = input("Please enter image name: ")
        if img_name == "bye":
            break
        compare_preds(img_name + '.JPEG')