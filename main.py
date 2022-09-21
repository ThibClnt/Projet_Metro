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
    _unnamed_count = 0  # Compte le nombre d'objets non nommés, afin de leur donner un nom par défaut

    def __init__(self, name=None, line=None, branch=None, terminus=None):
        self.name = Vertex._unnamed_count if name is None else name
        if name is None:
            Vertex._unnamed_count += 1
        self.root = None
        self.rank = 0
        self.line = line
        self.branch = branch
        self.terminus = terminus


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
            vertex : dict() for vertex in vertices
        }
        self.vertices = vertices
    
    def add_vertex(self,vertex):
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
# Extraction des données - Parsing         #
############################################

# TODO : Changer l'adresse du fichier en chemin relatif

def data_graph(): 
    """ Retourne directement le Graphe du métro"""


    Graph_metro = NonOrientedGraph([])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"metro.txt"), "r") as file:
        for line1 in file:
            if "V " in line1 and "num_sommet" not in line1:
                line_vertex = line1.split()
                print(line_vertex)
                val = line_vertex[-3][1]
                val_branch = line_vertex[-1]
                term = line_vertex[-2][1:]
                vertex = Vertex(name = " ".join(line_vertex[2:-3]), line = val, branch = val_branch, terminus = term)
                Graph_metro.add_vertex(vertex)
            elif "E " in line1 and "num_sommet1" not in line1:
                line_edge = line1.split()
                Graph_metro.add_edge(Graph_metro.vertices[int(line_edge[1])],Graph_metro.vertices[int(line_edge[2])],int(line_edge[3]))
        return Graph_metro

            
def data_coord():
    dict_coord = dict()
    """ structure : dict(nom sommet : (x,y)) """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"pospoints.txt"), "r") as file:
        for line in file:
            line = line.split(";")
            vertex_name = line[2].split("\n")[0]
            vertex_name = " ".join(vertex_name.split("@"))
            dict_coord[(line[0], line[1])] = vertex_name
    return dict_coord
    







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
        self._init_display(path)

        self.start = None
        self.end = None
        self.mode = Gui.PCC

        self.pos = data_coord()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def _init_display(self, path):
        im = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
        self.fig, self.ax = plt.subplots(figsize=plt.figaspect(im))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.imshow(im)
        plt.axis('off')

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


if __name__ == '__main__':
    Gui("metrof_r.png")
