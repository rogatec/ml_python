# Load libraries
from pandas import read_csv

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape (dimension of the dataset)
print(dataset.shape)
print('--------------------------')
# head (peek on the data)
print(dataset.head(20))
print('--------------------------')
# descriptions (statistical summary)
print(dataset.describe())
print('--------------------------')
# class distribution
print(dataset.groupby('class').size())
print('--------------------------')
