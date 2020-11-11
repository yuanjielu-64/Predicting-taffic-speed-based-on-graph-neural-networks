import pandas as pd
import numpy as np
from collections import defaultdict

#df = pd.read_csv(r'Z:/project/data/tmc.csv')
df = pd.read_csv('/Users/yuanjielu/Desktop/python/project/data/tmc.csv')

#df_adj = pd.read_excel(r'Z:/project/data/adj_tyson.xlsx')
df_adj = pd.read_excel('/Users/yuanjielu/Desktop/python/project/data/adj_tyson.xlsx')
tmc = list(df.tmc)

adj = pd.DataFrame(np.zeros(shape = (131,131)), index = tmc, columns = tmc)
#print(adj)

edge = []

for i in range(len(df_adj)):
    a = df_adj['start'].iloc[i]
    b = df_adj['end'].iloc[i]

    a_index = tmc.index(a)
    b_index = tmc.index(b)
    a_info = df[df['tmc'] == a]['miles'].values
    b_info = df[df['tmc'] == b]['miles'].values
    edge.append([a, b,(a_info + b_info) / 2])
    #print(a_index,b_index)
    adj.iat[a_index,b_index] = 1


#for i in range(len(adj)):
#    adj.iat[i, i] = 1

#print(edge)

adj.to_csv('/Users/yuanjielu/Desktop/python/project/data/adj.csv')

class Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = defaultdict(list)
    self.distances = {}

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    #self.edges[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance

def dijsktra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes:
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distances[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path

g = Graph()
adj_1 = pd.DataFrame(np.zeros(shape = (131,131)), index = tmc, columns = tmc)
for i in tmc:
    g.add_node(i)

for j in edge:
    g.add_edge(j[0],j[1],j[2][0].round(6))

for i in tmc:
    a = dijsktra(g,i)
    #print(a[0])
    #print(a[0].keys())
    k = list(a[0].keys())
    v = list(a[0].values())
    a_index = tmc.index(i)

    for m in range(len(k)):
        b_index = tmc.index(k[m])
        c = v[m]
        if v[m] == 0:
            adj_1.iat[a_index, b_index] = 0
        else:
            adj_1.iat[a_index, b_index] = v[m].round(1)


print(adj_1)

adj_1.to_csv('/Users/yuanjielu/Desktop/python/project/data/adj_1.csv')

