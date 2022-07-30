import random
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import timeit
import pyamaze


def list_2_matr(agj_list, V):
    matrix = [[0 for j in range(V)]
              for i in range(V)]
    for i in range(V):
        for u,v,w in agj_list:
            matrix[u][v] = w
            matrix[v][u] = w
    return matrix


class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
        self.graph_2 = [[0 for column in range(vertices)]
                      for row in range(vertices)]


    def addEdge(self, u, v, w):
        if any(a[0] == u and a[1] == v for a in self.graph) or any(a[0] == v and a[1] == u for a in self.graph):
            return False
        else:
            self.graph.append([u, v, w])
            return True


    def printArr(self, dist):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}".format(i, dist[i]))

    def printSolution(self, dist):
        print("Vertex Distance from Source using Dijkstra alg")
        for node in range(self.V):
            print(node, ": ", dist[node])


    def minDistance(self, dist, sptSet):
        min = sys.maxsize
        min_index = 0
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index


    def dijkstra(self, src):
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True

            for v in range(self.V):
                if self.graph_2[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph_2[u][v]:
                    dist[v] = dist[u] + self.graph_2[u][v]

        self.printSolution(dist)


    def BellmanFord(self, src):
        dist = [float("Inf")] * self.V
        dist[src] = 0
        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return

        self.printArr(dist)

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():

    # task 1

    num_vert = 100
    num_edges = 500
    max_dist = 10

    g = Graph(num_vert)
    added_edges = 0
    while added_edges < num_edges:
        u = random.randint(0,num_vert-1)
        v = random.randint(0,num_vert-1)
        while u == v:
            v = random.randint(0, num_vert - 1)
        w = random.randint(1,max_dist)
        if g.addEdge(u,v,w):
            added_edges += 1

    start_v = random.randint(0, num_vert-1)
    #print("start vertice = ", start_v)

    adj_matr = list_2_matr(g.graph, num_vert)
    g.graph_2 = adj_matr

    #print(g.graph)
    #for i in range(num_vert):
    #    print(adj_matr[i])

    #g.dijkstra(start_v)
    #print()

    #g.BellmanFord(start_v)

    G = nx.from_numpy_matrix(np.array(adj_matr))

    time_bf = 0
    for _ in range(10):
        start = timeit.default_timer()
        pred, dist = nx.single_source_bellman_ford(G, start_v)
        time_bf = time_bf + (timeit.default_timer() - start) / 10

    time_dj = 0
    for _ in range(10):
        start = timeit.default_timer()
        pred2, dist2 = nx.single_source_dijkstra(G, start_v)
        time_dj = time_dj + (timeit.default_timer() - start) / 10

    """
    sorted_dist2 = sorted(pred2.items())
    print("Vertex Distance from Source using Dijkstra")
    [print(a[0], ": ", a[1]) for a in sorted_dist2]

    print("Vertex Distance from Source using Bellman-Ford alg")
    sorted_dist = sorted(pred.items())
    [print(a[0], ": ", a[1]) for a in sorted_dist]

    print()
    print("Average Dijkstra alg time: ", time_dj)

    print()
    print("Average Bellman-Ford alg time: ", time_bf)

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    plt.show()
    """

    # task 2

    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    maze = [[maze[j][i] for j in range(len(maze))] for i in range(len(maze[0])-1,-1,-1)]

    flag = True
    s_x = 0
    s_y = 0
    while flag:
        s_x = random.randint(0, 9)
        s_y = random.randint(0, 19)
        if maze[s_x][s_y] == 0:
            flag = False

    flag = True
    e_x = 0
    e_y = 0
    while flag:
        e_x = random.randint(0, 9)
        e_y = random.randint(0, 19)
        if maze[e_x][e_y] == 0:
            flag = False

    start = (s_x, s_y)
    end = (e_x, e_y)

    path = astar(maze, start, end)
    print("start: ", start)
    print("end: ", end)
    print("found path length: ", len(path))
    print(path)

    for a in path:
        maze[a[0]][a[1]] = 0.4

    maze[start[0]][start[1]] = 2
    maze[end[0]][end[1]] = 2

    plt.pcolormesh(maze, cmap='pink')
    #plt.axes().set_aspect('equal')  # set the x and y axes to the same scale
    #plt.axes().invert_yaxis()  # invert the y-axis so the first row of data is at the top
    plt.xticks([])  # remove the tick marks by setting to an empty list
    plt.yticks([])  # remove the tick marks by setting to an empty list

    plt.show()

if __name__ == "__main__":
    main()