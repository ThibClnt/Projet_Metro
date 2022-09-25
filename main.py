import matplotlib.backend_bases
import matplotlib.pyplot as plt
import math
import os


############################################
# Représentation des données - Abstraction #
############################################


class Vertex:
    """
    Représente le sommet d'un graphe. Peut avoir un parent, ou "représentant", nommé root.
    Rank représente l'étage dans l'arbre, pour l'algorithme de Kruskal.
    """

    def __init__(self, name=None, line=None, branch=None, terminus=None):
        self.name = name
        self.root = self
        self.rank = 0
        self.line = line
        self.branch = branch
        self.terminus = terminus

    def __repr__(self):
        return f" Vertex : {self.name} ; parent : {self.root.name} ; "


class Graph:
    """
    Objet qui représente un graphe orienté, avec des méthodes utilitaires pour réaliser l'algorithme de Kruskal
    """

    def __init__(self, vertices):
        """
        :param vertices: Iterable[Vertex]
        """
        # Arêtes du graphe sous la forme : {pt_de_depart: {pt_darrive: poids}}
        self.edges = {
            vertex: dict() for vertex in vertices
        }
        self.vertices = vertices
        self.acpm_fait = False  # Vrai lorsque l'acpm a été calculé

    def add_vertex(self, vertex):
        self.vertices += [vertex]
        self.edges[vertex] = dict()

    def add_edge(self, origin, dest, weight):
        """
        Ajoute une arête au graphe
        :param origin: Vertex
        :param dest: Vertex
        :param weight: float
        """
        if origin not in self.edges.keys() or dest not in self.edges.keys():
            raise AttributeError(f"({origin.name} ; {dest.name}) n'est pas une paire de sommets valide.")

        self.edges[origin][dest] = weight

    def find(self, vertex):
        """
        Trouve la racine d'un ensemble de points
        :param vertex : Vertex
        :return: Vertex (root)
        """
        if vertex.root == vertex:
            return vertex
        return self.find(vertex.root)

    def union(self, x, y):
        """
        Rassemble les arbres de x et de y
        :param x: Vertex
        :param y: Vertex
        """
        xroot = self.find(x)
        yroot = self.find(y)

        if xroot.rank < yroot.rank:
            xroot.root = yroot
        else:
            yroot.root = xroot
            if x.rank == y.rank:
                x.rank += 1


class NonOrientedGraph(Graph):
    """
    Représente un graphe non orienté.
    Hérite de Graph (graphe orienté), mais chaque arête ajoutée l'est dans les deux sens.
    """

    def add_edge(self, origin, dest, weight):
        """
        Ajoute une arête au graphe. Chaque arête est ajoutée dans les deux sens
        :param origin: Vertex
        :param dest: Vertex
        :param weight: float
        """
        super().add_edge(origin, dest, weight)
        super().add_edge(dest, origin, weight)


############################################
# Extraction des données - Parsing         #
############################################

# TODO : Changer l'adresse du fichier en chemin relatif

def data_graph():
    """ Retourne directement le Graphe du métro"""

    graph_metro = NonOrientedGraph([])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metro.txt"), "r") as file:
        for line1 in file:
            if "V " in line1 and "num_sommet" not in line1:
                line_vertex = line1.split()
                val = line_vertex[-3][1]
                val_branch = line_vertex[-1]
                term = line_vertex[-2][1:]
                vertex = Vertex(name=" ".join(line_vertex[2:-3]), line=val, branch=val_branch, terminus=term)
                graph_metro.add_vertex(vertex)
            elif "E " in line1 and "num_sommet1" not in line1:
                line_edge = line1.split()
                graph_metro.add_edge(graph_metro.vertices[int(line_edge[1])], graph_metro.vertices[int(line_edge[2])],
                                     int(line_edge[3]))
        return graph_metro


def data_coord():
    dict_coord = dict()
    """ structure : dict(nom sommet : (x,y)) """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pospoints.txt"), "r") as file:
        for line in file:
            line = line.split(";")
            vertex_name = line[2].split("\n")[0]
            vertex_name = " ".join(vertex_name.split("@"))
            dict_coord[(line[0], line[1])] = vertex_name
    return dict_coord


############################################
# Algorithmes                              #
############################################
def kruskal(graph: Graph):
    if graph.acpm_fait:
        return

    edges = []
    result = []
    for pt_depart in graph.edges.keys():
        for pt_arrivee in graph.edges[pt_depart].keys():
            edges.append([pt_depart, pt_arrivee, graph.edges[pt_depart][pt_arrivee]])

    edges.sort(key=lambda e: e[2])

    while len(edges) != 0:
        orig, dest, _ = edges.pop(0)
        if graph.find(orig) != graph.find(dest):
            graph.union(orig, dest)
            for v in graph.edges.keys():
                if v.name == orig.name:
                    graph.union(v, orig)
                if v.name == dest.name:
                    graph.union(v, dest)

            result.append((orig, dest))

    return result


# TODO: Dijskra: Renommage variables; Adaptation types de données ; Encapsulation
def argmin_dict(dist):
    min_ = math.inf
    argmin = "inf"
    for i in dist:
        if dist[i] <= min_:
            min_ = dist[i]
            argmin = i
    return argmin


def dijkstra(graph, summits):
    vertices = list(graph.keys())
    dist = {}
    pred = {}
    for i in vertices:
        dist[i] = math.inf
        pred[i] = None

    dist[summits] = 0
    vertices_copy = vertices

    while vertices_copy:
        dist_f = {}
        for i in vertices_copy:
            dist_f[i] = dist[i]

        u = argmin_dict(dist_f)
        del vertices_copy[vertices_copy.index(u)]

        set1 = set(vertices_copy)
        set2 = set(list(graph[u].keys()))
        intersect = list(set1 & set2)

        for i in intersect:
            if dist[i] > dist[u] + graph[u][i]:
                dist[i] = dist[u] + graph[u][i]
                pred[i] = u

        del dist_f[u]
    return dist, pred


def shortest_path(summits, target_vertex, pred):
    arrive = pred[target_vertex]
    path = [arrive, target_vertex]
    while arrive != summits:
        arrive = pred[arrive]
        path = [arrive] + path
    return path


def utilisation_dijkstra(pointa, pointb):
    print(pointa, pointb)


############################################
# Interface graphique                      #
############################################


class Gui:
    PCC = 1
    ACPM = 2

    def __init__(self, path):
        # Initialisation de l'affichage
        self.impath = path
        self._init_display()

        self.start = None
        self.end = None
        self.mode = Gui.PCC

        self.graph = data_graph()
        self.pos = data_coord()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _init_display(self):
        im = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.impath))
        self.fig, self.ax = plt.subplots(figsize=plt.figaspect(im))
        self.fig.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        self.ax.imshow(im)

    def reset_display(self):
        im = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.impath))
        self.ax.imshow(im)

    def on_click(self, event):
        if self.mode != Gui.PCC:
            return

        ex, ey = event.xdata, event.ydata
        for coord, vertex in self.pos.items():
            x, y = float(coord[0]), float(coord[1])
            if ((x - ex) ** 2 + (y - ey) ** 2) < 12:
                if event.button == matplotlib.backend_bases.MouseButton.LEFT:
                    self.start = vertex
                elif event.button == matplotlib.backend_bases.MouseButton.RIGHT:
                    self.end = vertex

                if self.start is not None and self.end is not None:
                    utilisation_dijkstra(self.start, self.end)
                break

    def switch_mode(self):
        self.mode = 3 - self.mode

        if self.mode == Gui.ACPM:
            self.display_acpm()
        elif self.mode == Gui.PCC:
            self.start, self.end = None, None

    def display_acpm(self):
        acpm = kruskal(self.graph)

        for v, p in acpm:
            vx, vy, px, py = 0, 0, 0, 0

            for coord, vertex in self.pos.items():
                if v.name == vertex:
                    vx, vy = int(coord[0]), int(coord[1])
                    break

            for coord, vertex in self.pos.items():
                if p.name == vertex:
                    px, py = int(coord[0]), int(coord[1])
                    break

            plt.plot((vx, px), (vy, py), color='blue', linewidth=2)


if __name__ == '__main__':
    Gui("metrof_r.png").switch_mode()
    plt.show()

