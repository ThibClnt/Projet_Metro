import matplotlib.patches
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

    def __init__(self, name=None, line = None,  branch = None, terminus = None):
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
                val = line_vertex[-3][1]                # Numero de ligne
                val_branch = line_vertex[-1]            # 0,1 ou 2 en fonctions des embranchements
                term = line_vertex[-2][1:]              # True or False en str
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
            if vertex_name in dict_coord:
                dict_coord[vertex_name] += [(line[0], line[1])]
            else:
                dict_coord[vertex_name] =[(line[0], line[1])]
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


def dijkstra(graph, summit):
    vertices = list(graph.keys())
    dist = {}
    pred = {}
    for i in vertices:
        dist[i] = math.inf
        pred[i] = None

    dist[summit] = 0
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


def shortest_path(summit, target_vertex, pred):
    arrive = pred[target_vertex]
    path = [arrive, target_vertex]
    while arrive != summit:
        arrive = pred[arrive]
        path = [arrive] + path
    return path


############################################
# Interface graphique                      #
############################################


############################################
# Point de départ & Tests                  #
############################################
# if __name__ == '__main__':
#     ax: matplotlib.pyplot.Axes = matplotlib.pyplot.gca()
#     matplotlib.pyplot.axis((0, 987, 0, 952))
#     ax.set_aspect('equal')

#     stops = [
#         Stop(x, y)
#         for x, y in ((907, 682),
#                      (892, 669),
#                      (876, 652))
#     ]
#     g = NonOrientedGraph(stops)

#     for v in stops:
#         v.draw(ax, 5)

#     matplotlib.pyplot.show()
 





A = data_graph()
for i in A.edges:
    print(i.terminus)



# def onclick(event):
#     c_x = event.xdata
#     c_y = event.ydata
#     print(c_x)
#     print(c_y)
#     print("\n")
#     return c_x,c_y

# im = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),"metrof_r.png"))
# fig, ax = plt.subplots(figsize = plt.figaspect(im))
# fig.canvas.mpl_connect('button_press_event', onclick)
# fig.subplots_adjust(0,0,1,1)
# ax.imshow(im)
# # plt.axis('off')

# # ax.scatter(589,696, s= 10)
# plt.show()



############################################
# Dijkstra                                 #
############################################

def terminus(graph):
    liste_terminus = []
    for vertex in graph.keys():
        if vertex.terminus == "True":
            liste_terminus += [[vertex.name,vertex.line,vertex.branch]]
    return liste_terminus

# Interface renvoie nom des arrets de départ et d'arrivé"

def utilisation_dijkstra(nom_depart, nom_arrive):
    # Liste des arrets de meme station mais pas meme ligne (ex : Chatellet)
    graph_metro = data_graph()
    liste_depart = []
    liste_arrive = []
    for vertex in graph_metro.edges.keys():
        if vertex.name == nom_depart:
            liste_depart += [vertex]
        if vertex.name == nom_arrive:
            liste_arrive += [vertex]

    temps = 0
    for vertex in liste_depart:
        tuple = dijkstra(graph_metro.edges, vertex) # tuple = (dist, pred)
        for vertex_1 in nom_arrive:
            if tuple[0][vertex_1] < temps:
                temps = tuple[0][vertex_1]
                pred = tuple[1]
                vertex_depart = vertex
                vertex_arrive = vertex_1

    

    return shortest_path(vertex,vertex_1,pred)
    

    # while i < len(shortest_path):
    #     vertex = shortest_path[i]
    #     j = i
    #     while vertex.line == shortest_path[j].line and vertex.branch == shortest_path[j].branch:
    #         j+=1

    #     i = j
    #     print("Prendre la ligne",vertex.line,"direction", terminus.name,"à",vertex.name,"jusqu'à", shortest_path[i].name)


