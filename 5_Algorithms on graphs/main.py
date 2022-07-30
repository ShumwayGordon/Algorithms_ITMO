import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


def convert_to_adjacency(matrix):
    start = 0
    res = []
    lst = []
    n = len(matrix)

    for i in range(n):
        res.append(lst*n)
    while start < n:
        y = matrix[start]
        for i in range(len(y)):
            if y[i] == 1:
                res[start].append(i)
        start += 1
    return res


def convert_to_matrix(graph):
    matrix = []
    for i in range(len(graph)):
        matrix.append([0]*len(graph))
        for j in graph[i]:
            matrix[i][j] = 1
    return matrix


def main():
    vert = 100
    edges = 200

    G = nx.generators.gnm_random_graph(vert, edges)
    print(G)

    adj_matr = nx.adjacency_matrix(G, weight=None)
    with np.printoptions(edgeitems=100):
        print("Adjacency matrix: \n", adj_matr.todense())

    print("Adjacency list:")
    for line in nx.generate_adjlist(G):
        with np.printoptions(edgeitems=100):
            print(line)

    print()
    print("Depth First Search:")
    A = list(nx.connected_components(G))
    for i in range(len(A)):
        print("Component #", i+1)
        iterator = iter(A[i])
        item0 = next(iterator, None)
        print(list(nx.dfs_preorder_nodes(G, source=item0)))
        print()

    source = random.randint(0, vert-1)
    destin = random.randint(0, vert - 1)

    print(list(nx.bfs_edges(G, source=source)))
    print()
    steps = nx.shortest_path_length(G=G, source=source, target=destin)
    print("The distance between nodes #", source, "and #", destin, "=", steps)
    path_sh = list(nx.shortest_path(G=G, source=source, target=destin))
    print(path_sh)

    plt.figure()
    ax = plt.gca()
    ax.set_title("Randomly generated graph")
    nx.draw(G, pos=nx.spring_layout(G), node_color='lightgreen', ax=ax, with_labels=True)
    plt.show()


if __name__ == "__main__":
    main()