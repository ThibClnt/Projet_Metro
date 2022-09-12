import matplotlib.patches
import matplotlib.pyplot
import math


############################################
# Extraction des données - Parsing         #
############################################

# TODO : Changer l'adresse du fichier en chemin relatif
def data():
    """
    # Retourne 4 listes : noms des stations, sommet 1, sommet 2, temps entre sommet 1 et 2
    """
    list_station = []
    vertex_1 = []
    vertex_2 = []
    time = []
    with open(r"C:\Users\kekel\Documents\Efrei\Maths pour l'info\metro.txt", "r") as file:
        for line in file:
            if "V " in line and "num_sommet" not in line:
                line_vertex = line.split()
                list_station += [line_vertex[2]]
            elif "E " in line and "num_sommet1" not in line:
                line_edge = line.split()
                vertex_1 += [int(line_edge[1])]
                vertex_2 += [int(line_edge[2])]
                time += [int(line_edge[3])]

    return list_station, vertex_1, vertex_2, time


############################################
# Représentation des données - Abstraction #
############################################


class Vertex:
    """
    Représente le sommet d'un graphe. Peut avoir un parent, ou "représentant", nommé root.
    Rank représente l'étage dans l'arbre, pour l'algorithme de Kruskal.
    """
    _unnamed_count = 0  # Compte le nombre d'objets non nommés, afin de leur donner un nom par défaut

    def __init__(self, name=None):
        self.name = Vertex._unnamed_count if name is None else name
        if name is None:
            Vertex._unnamed_count += 1
        self.root = None
        self.rank = 0


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
            vertex.name: dict() for vertex in vertices
        }

    def add_edge(self, origin, dest, weight):
        """
        Ajoute une arête au graphe
        :param origin: Vertex
        :param dest: Vertex
        :param weight: float
        """
        if origin not in self.edges.keys() and dest not in self.edges.keys():
            raise AttributeError(f"({origin} ; {dest}) n'est pas une paire de sommets valide.")

        self.edges[origin][dest] = weight

    def find(self, vertex):
        """
        Trouve la racine d'un ensemble de points
        :param vertex : Vertex
        :return: Vertex (root)
        """
        if vertex.root is None:
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

        if xroot == yroot:
            if x.rank < y.rank:
                x.root = y.root
            else:
                y.root = x.root
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
# Lien données -> interface graphique      #
############################################

# TODO : Doc
class Stop(Vertex):
    """
    Représente un arrêt de métro, i.e. un sommet du graphe, qui peut être représenté sur la carte comme un point rouge,
    et cliqué.
    """

    def __init__(self, x, y, name=None):
        """
        :param x: float - position horizontale, en px, sur l'image
        :param y: float - position verticale, en px, sur l'image
        :param name: str | None - nom de l'arrêt. Un id est donné par défaut, si aucun nom n'est donné
        """
        super().__init__(name)
        self.x, self.y = x, y
        self.artist = None

    def draw(self, canvas: matplotlib.pyplot.Axes, radius=1):
        """
        Dessine l'arrêt sur la carte
        :param canvas: matplotlib.pyplot.Axes - Objet graphique sur lequel dessiner l'arrêt
        :param radius: float - rayon, en px, du cercle représentant l'arrêt
        """
        canvas.add_artist(matplotlib.patches.Circle((self.x, self.y), radius, color='red'))

    def collide_with(self, x, y, radius=1):
        """
        Pour tester si l'arrêt a été cliqué
        :param x: float - position, en px, de l'évenement
        :param y: float - position, en px, de l'évenement
        :param radius: float - rayon d'action, en px, de l'arrêt
        """
        return self.x - radius < x < self.x + radius and self.y - radius < y < self.y + radius


############################################
# Algorithmes                              #
############################################
# TODO : Kruskal


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


############################################
# Interface graphique                      #
############################################


############################################
# Point de départ & Tests                  #
############################################
if __name__ == '__main__':
    ax: matplotlib.pyplot.Axes = matplotlib.pyplot.gca()
    matplotlib.pyplot.axis((0, 987, 0, 952))
    ax.set_aspect('equal')

    stops = [
        Stop(x, y)
        for x, y in ((907, 682),
                     (892, 669),
                     (876, 652))
    ]
    g = NonOrientedGraph(stops)

    for v in stops:
        v.draw(ax, 5)

    matplotlib.pyplot.show()
