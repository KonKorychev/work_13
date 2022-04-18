# Задача 5. Предсказание уровня экспрессии белка
**Цель:**
<br/>Предсказать экспрессию белков (`target`) по приведенным данным для отложенной выборки. Ответы в отложенной выборке `test` даны для самостоятельной валидации.

```python
# Импорт основных библиотек
import numpy as np
import pandas as pd

# Импорт библиотеки по изучению сетевых структур
import networkx as nx

# Импорт библиотеки обработки естественного языка
from gensim.models import Word2Vec

# Импорт библиотек машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Импорт библиотеки машинного обучения графов
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

# Импорт библиотек построения диаграмм и графиков
import matplotlib.pyplot as plt
import seaborn as sns

# Определение режима вывода диаграмм
%matplotlib inline

# Определение стиля вывода диаграмм
plt.rc('axes', grid=True)
plt.rc('grid', linewidth=0.5)
```

## Загрузка исходных данных
```python
# Загрузка обучающего набора данных (уровень экспрессии белков)
protein_train = pd.read_csv('train.csv', low_memory=False)
protein_train.head()
```
![png](Images/table01.jpg)

```python
# Загрузка тестового набора данных (уровень экспрессии белков)
protein_test = pd.read_csv('test.csv', low_memory=False)
protein_test.head()
```
![png](Images/table02.jpg)

```python
# Загрузка данных рёбер графа (взаимодействие белков)
protein_edges = pd.read_csv('edges.csv', low_memory=False)
protein_edges.head()
```
![png](Images/table03.jpg)

## Разведочный анализ
```python
# Вывод размеров обучающей и тестовой выборок
print('Обучающая выборка:{};  Тестовая выборка:{}'.format(protein_train.shape, protein_test.shape))
```
```
Обучающая выборка:(8000, 2);  Тестовая выборка:(2000, 2)
```

```python
# Построение графа NetworkX для отражения структуры взаимодействия белков
protein_graph = nx.from_pandas_edgelist(protein_edges, 'node_1', 'node_2')
print('Узлов: {};  Рёбер: {}'.format(protein_graph.number_of_nodes(), protein_graph.number_of_edges()))
```
```
Узлов: 10000;  Рёбер: 594174
```

```python
# Определение схемы вывода диаграмм
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Вывод полного графа взаимодействия белков
nx.draw_spring(protein_graph, node_size=15, width=0.1, linewidths=0, ax=axes[0])

# Вывод сокращенного варианта графа на основе степени важности узлов
degree_centrality = pd.Series(nx.degree_centrality(protein_graph))
subgraph = protein_graph.subgraph(degree_centrality[degree_centrality > 0.06].index)
nx.draw_spring(subgraph, node_size=15, width=0.1, linewidths=0, ax=axes[1])

# Вывод заголовков диаграмм
axes[0].set_title('Полный граф взаимодействия белков')
axes[1].set_title('Упрощенный граф взаимодействия белков')

# Отображение диаграмм
plt.show()
```
![png](Images/chart01.jpg)

## Предобработка данных
#### Извлечение признаков с помощью NetworkX
```python
# Извлечение основных признаков узлов из графа (степень важности, посредничества, рейтинг узла)
degree_centrality = pd.Series(nx.degree_centrality(protein_graph))
betweenness_centrality = pd.Series(nx.betweenness_centrality(protein_graph, k=30))
pagerank = pd.Series(nx.pagerank(protein_graph))

# Вывод размерности признаков
print(degree_centrality.shape, betweenness_centrality.shape, pagerank.shape)
```
```
(10000,) (10000,) (10000,)
```

```python
# Формирование датасета извлеченных признаков
node_features = pd.concat(
    [degree_centrality, betweenness_centrality, pagerank], axis=1
).reset_index()

node_features.columns = ['node'] + ['x' + str(i) for i in range(3)]

# Вывод извлеченных признаков
node_features.head(3)
```
![png](Images/table04.jpg)

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```


