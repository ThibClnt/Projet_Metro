import matplotlib.backend_bases
import matplotlib.pyplot as plt
import math
import os


############################################
# Représentation des données - Abstraction #
############################################
class Vertex:
    """
    Représente un sommet dans un arbre (peut être utilisé de manière plus général dans un graphe).
    """

    def __init__(self, name=None):
        self.name = name
        self.root = self
        self.rank = 0

    def __repr__(self):
        return f" Vertex : {self.name} ; parent : {self.root.name} ; "


class Graph:
    """
    Objet qui représente un graphe orienté, avec des méthodes utilitaires pour réaliser l'algorithme de Kruskal
    """

    def __init__(self):
        self.edges = {}         # Arêtes du graphe sous la forme : {pt_de_depart: {pt_darrive: poids}}
        self.vertices = list()  # Liste des sommets du graphe
        self.acpm_fait = False  # Vrai lorsque l'acpm a été calculé

    def add_vertex(self, vertex):
        """
        Ajoute un sommet au graphe
        :param vertex: Vertex
        """
        self.vertices += [vertex]
        self.edges[vertex] = dict()

    def add_edge(self, origin, dest, weight):
        """
        Ajoute une arête (donc deux sens) au graphe
        :param origin: Vertex
        :param dest: Vertex
        :param weight: float
        """
        if origin not in self.edges.keys() or dest not in self.edges.keys():
            raise AttributeError(f"({origin.name} ; {dest.name}) n'est pas une paire de sommets valide.")

        self.edges[origin][dest] = weight
        self.edges[dest][origin] = weight


class Arret(Vertex):
    """
    Sommet d'un graphe, représentant un arrêt de transports en commun
    """

    def __init__(self, name=None, line=None, branch=None, terminus=None):
        super().__init__(name)
        self.line = line            # Numéro de ligne
        self.branch = branch        # Identifiant de l'embranchement
        self.terminus = terminus    # Vrai si c'est un terminus - Faux sinon


############################################
# Extraction des données - Parsing         #
############################################

def data_graph():
    """
    Parse le fichier de données metro.txt et retourne un graphe correspondant
    """

    graph_metro = Graph()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metro.txt"), "r", encoding="utf8") as file:
        for line1 in file:
            if "V " in line1 and "num_sommet" not in line1:
                line_vertex = line1.split()
                val = line_vertex[-3][1:]
                val_branch = int(line_vertex[-1])
                term = line_vertex[-2][1:]
                vertex = Arret(name=" ".join(line_vertex[2:-3]), line=val, branch=val_branch, terminus=term)
                graph_metro.add_vertex(vertex)
            elif "E " in line1 and "num_sommet1" not in line1:
                line_edge = line1.split()
                graph_metro.add_edge(graph_metro.vertices[int(line_edge[1])], graph_metro.vertices[int(line_edge[2])],
                                     int(line_edge[3]))

    # Orientation de quelques (rares) arêtes
    eglise = javel = molitor = porte = boulogne = auteuil = chardon = mirabeau = None
    for v in graph_metro.edges.keys():

        if v.name == "Église d'Auteuil":
            eglise = v
        if v.name == "Javel":
            javel = v
        if v.name == "Michel Ange Molitor" and v.line == "10":
            molitor = v
        if v.name == "Porte d'Auteuil":
            porte = v
        if v.name == "Boulogne, Jean Jaurès":
            boulogne = v
        if v.name == "Michel Ange Auteuil" and v.line == "10":
            auteuil = v
        if v.name == "Chardon Lagâche":
            chardon = v
        if v.name == "Mirabeau":
            mirabeau = v

    del graph_metro.edges[javel][mirabeau]
    del graph_metro.edges[mirabeau][chardon]
    del graph_metro.edges[chardon][molitor]
    del graph_metro.edges[eglise][javel]
    del graph_metro.edges[auteuil][eglise]
    del graph_metro.edges[porte][auteuil]
    del graph_metro.edges[boulogne][porte]
    del graph_metro.edges[porte][molitor]
    del graph_metro.edges[molitor][porte]

    return graph_metro


def data_coord():
    """
    Parse le fichier des positions des arrêts. Retourne un dictionnaire qui associe à chaque position (x, y) le nom de
    l'arrêt qui s'y trouve.
    """
    dict_coord = dict()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pospoints.txt"), "r", encoding="utf8") as file:
        for line in file:
            line = line.split(";")
            vertex_name = line[2].split("\n")[0]
            vertex_name = " ".join(vertex_name.split("@"))
            dict_coord[(line[0], line[1])] = vertex_name
    return dict_coord


############################################
# Kruskal                                  #
############################################
def find(vertex):
    """
    Trouve la racine d'un arbre
    :param vertex : Vertex
    :return: Vertex (root)
    """
    if vertex.root == vertex:
        return vertex
    return find(vertex.root)


def union(x, y):
    """
    Rassemble les arbres de x et de y
    :param x: Vertex
    :param y: Vertex
    """
    xroot = find(x)
    yroot = find(y)

    # Comparaison des "rangs". Cette notion permet d'obtenir un arbre équilibré. La racine de la fusion des deux arbres
    # est la racine de l'arbre avec le plus d'"étages". Un arbre désiquilibré augmenterait le temps de recherche d'une
    # racine (dans la fonction find())
    if xroot.rank < yroot.rank:
        xroot.root = yroot
    else:
        yroot.root = xroot
        if x.rank == y.rank:
            x.rank += 1


def kruskal(graph: Graph):
    """
    Algorithme de Kruskal : retourne l'acpm du graphe passé en paramètre (en tant qu'une liste d'arêtes). Une arête est
    représentée comme un tuple (sommet1, sommet2)
    :param graph: Graph
    :rtype: list(tuple())
    """

    # Si l'acpm a déjà été fait pour le graphe, pas besoin de recommencer
    if graph.acpm_fait:
        return

    edges = []          # Liste des arêtes à traiter, triées par poids croissant
    result = []         # Liste des arêtes de l'acpm
    for pt_depart in graph.vertices:
        for pt_arrivee in graph.edges[pt_depart].keys():
            edges.append([pt_depart, pt_arrivee, graph.edges[pt_depart][pt_arrivee]])

    edges.sort(key=lambda e: e[2])

    while len(edges) != 0:
        # Chaque arête est traitée à tour de rôle (poids min en premier)
        # Une arête traitée est retirée de la liste 'edges'
        orig, dest, _ = edges.pop(0)

        # Si les parents des arbres auquel appartiennent les deux sommets de l'arête sont les mêmes, les arbres sont les
        # mêmes, ce qui signifie que l'ajout d'une telle arête formerait un cycle
        if find(orig) != find(dest):
            union(orig, dest)

            # Car un arêt est représenté en plusieurs sommets de mêmes noms, on traite tous les sommets homonymes
            for v in graph.vertices:
                if v.name == orig.name:
                    union(v, orig)
                if v.name == dest.name:
                    union(v, dest)

            result.append((orig, dest))

    return result


############################################
# Dijkstra                                 #
############################################
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


def recherche_vertex(name, graph):
    for v in graph.edges.keys():
        if v.name == name:
            return v


def liste_terminus(graph):
    liste_terminus = []
    for vertex in graph.edges.keys():
        if vertex.terminus == "True":
            liste_terminus += [[vertex, vertex.line, vertex.branch]]
    return liste_terminus


def terminus(branch, ligne, liste_terminus):
    for liste in liste_terminus:
        if liste[1] == ligne and liste[2] == branch:
            return liste[0]


def find_terminus(station1, station2, graph):
    liste_term = liste_terminus(graph)
    station = station2
    station_pred = [station1]

    if station1.branch != station2.branch:  # Si branches différentes entre S1 et S2 on récupère directement le terminus
        return terminus(station2.branch, station2.line, liste_term)

    if station2.name == "Javel" and station1.name == "Mirabeau":
        return recherche_vertex("Gare d'Austerlitz", graph)

    while station.terminus != "True":
        for vertex in graph.edges[station]:

            if station.name == "Mirabeau" and (vertex.name == "Église d'Auteuil" or vertex.name == "Javel"):
                return recherche_vertex("Gare d'Austerlitz", graph)

            if station.name == "Porte d'Auteuil" and (
                    vertex.name == "Boulogne, Jean Jaurès" or vertex.name == "Michel Ange Molitor"):
                return recherche_vertex("Boulogne, Pont de Saint-Cloud, Rond Point Rhin et Danube", graph)

            if vertex not in station_pred and vertex.line == station.line:

                if vertex.branch == station.branch:  # On parcours la ligne jusqu'à trouver un terminus
                    station_pred += [station]
                    station = vertex


                else:
                    return terminus(vertex.branch, vertex.line,
                                    liste_term)  # Si on tombe sur un embranchements on peut renvoyer n'importe lequel des terminus

    return station


# Interface renvoie nom des arrets de départ et d'arrivé"

def utilisation_dijkstra(nom_depart, nom_arrive, graph):
    # Liste des arrets de meme station mais pas meme ligne (ex : Chatellet)
    liste_depart = []
    liste_arrive = []
    for vertex in graph.edges.keys():
        if vertex.name == nom_depart:
            liste_depart += [vertex]
        if vertex.name == nom_arrive:
            liste_arrive += [vertex]
    temps = math.inf
    for vertex in liste_depart:
        tuple = dijkstra(graph.edges, vertex)  # tuple = (dist, pred)

        for vertex_1 in liste_arrive:
            if tuple[0][vertex_1] < temps:
                temps = tuple[0][vertex_1]
                pred = tuple[1]
                vertex_depart = vertex
                vertex_arrive = vertex_1

    return temps, shortest_path(vertex_depart, vertex_arrive, pred)


def afficher_temps(tuple_result, graph):
    temps = tuple_result[0]
    sec = temps % 60
    min = (temps - sec) / 60
    print("Le trajet durera", int(min), "m", sec, "s (", temps, "s)")
    print("")


def afficher_texte_parcours(tuple_result, graph):
    shortest_path = tuple_result[1]
    i = 0
    var = 0
    while i < len(shortest_path):

        cond = True
        embranch = False
        while cond and i < len(shortest_path) - 1:
            if shortest_path[i].line == shortest_path[i + 1].line:

                if (shortest_path[i - 1].name == "Mirabeau" and shortest_path[i + 1].name == "Église d'Auteuil") or (
                        shortest_path[i - 1].name == "Porte d'Auteuil" and shortest_path[
                    i + 1].name == "Michel Ange Molitor"):
                    embranch = True
                    cond = False
                elif shortest_path[i].branch == shortest_path[i + 1].branch:  # Cas rien de particulier
                    i += 1
                elif (shortest_path[i - 1].branch == 1 and shortest_path[i + 1].branch == 2) or (
                        shortest_path[i + 1].branch == 1 and shortest_path[
                    i - 1].branch == 2):  # Cas embranchement relou
                    embranch = True
                    cond = False
                elif shortest_path[i + 1].branch == 0 or shortest_path[
                    i].branch == 0:  # Si on arrive sur d'une 0 ou va vers une 0
                    i += 1

                else:
                    cond = False
            else:
                cond = False
        terminus = find_terminus(shortest_path[i - 1], shortest_path[i], graph).name
        print("Prendre la ligne", shortest_path[i].line, "direction", terminus, "de", shortest_path[var].name,
              "jusqu'a", shortest_path[i].name)
        # Permettre de ne pas sauter un arret en cas d'embranchement
        if embranch:
            var = i
        else:
            var = i + 1

        i += 1


############################################
# Interface graphique                      #
############################################


class App:
    PCC = 1
    ACPM = 2

    def __init__(self, path):
        # Initialisation de l'affichage
        self.impath = path
        self._init_display()

        self.start = None
        self.end = None
        self.mode = App.PCC

        self.graph = data_graph()
        self.pos = data_coord()
        self.acpm = []      # Variable destinée

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
        if self.mode != App.PCC:
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
                    utilisation_dijkstra(self.start, self.end, self)
                break

    def switch_mode(self):
        self.mode = 3 - self.mode

        if self.mode == App.ACPM:
            self.display_acpm()
        elif self.mode == App.PCC:
            self.start, self.end = None, None

    def display_acpm(self):
        self.acpm = kruskal(self.graph)

        for v, p in self.acpm:
            vx, vy, px, py = 0, 0, 0, 0

            for coord, vertex in self.pos.items():
                if v.name == vertex:
                    vx, vy = int(coord[0]), int(coord[1])
                    break

            for coord, vertex in self.pos.items():
                if p.name == vertex:
                    px, py = int(coord[0]), int(coord[1])
                    break

            plt.plot((vx, px), (vy, py), color='#0000f0', linewidth=2)


if __name__ == '__main__':

    G = data_graph()

    name_depart = "Dupleix"
    name_arrive = "Porte des Lilas"

    afficher_temps(utilisation_dijkstra(name_depart, name_arrive, G), G)
    afficher_texte_parcours(utilisation_dijkstra(name_depart, name_arrive, G), G)
    """    App("metrof_r.png").switch_mode()
        plt.show()
    """
