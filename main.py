import matplotlib.backend_bases
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
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
        self.edges = {}  # Arêtes du graphe sous la forme : {pt_de_depart: {pt_darrive: poids}}
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
        self.line = line  # Numéro de ligne
        self.branch = branch  # Identifiant de l'embranchement
        self.terminus = terminus  # Vrai si c'est un terminus - Faux sinon


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
    eglise = javel = molitor = porte = boulogne = auteuil = chardon = mirabeau = danube = botzaris = place_des_fetes = pre_st_gervais = None
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
        if v.name == "Danube":
            danube = v
        if v.name == "Botzaris":
            botzaris = v
        if v.name == "Place des Fêtes":
            place_des_fetes = v
        if v.name == "Pré-Saint-Gervais":
            pre_st_gervais = v

    del graph_metro.edges[javel][mirabeau]
    del graph_metro.edges[mirabeau][chardon]
    del graph_metro.edges[chardon][molitor]
    del graph_metro.edges[eglise][javel]
    del graph_metro.edges[auteuil][eglise]
    del graph_metro.edges[porte][auteuil]
    del graph_metro.edges[boulogne][porte]
    del graph_metro.edges[porte][molitor]
    del graph_metro.edges[molitor][porte]
    del graph_metro.edges[botzaris][danube]
    del graph_metro.edges[danube][pre_st_gervais]
    del graph_metro.edges[pre_st_gervais][place_des_fetes]
    del graph_metro.edges[place_des_fetes][botzaris]

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
    graph.acpm_fait = True

    edges = []   # Liste des arêtes à traiter, triées par poids croissant
    result = []  # Liste des arêtes de l'acpm
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
    """
    Rend le sommet qui a pour nom "name"
    Utile dans notre programme pour les terminus de la ligne 10
    """
    for v in graph.edges.keys():
        if v.name == name and v.line == "10":
            return v


def recuperation_terminus(graph):
    """
    Retourne une liste des terminus. Terminus de la forme [arrêt, ligne, embranchement]
    """
    liste_terminus = []
    for vertex in graph.edges.keys():
        if vertex.terminus == "True":
            liste_terminus += [[vertex, vertex.line, vertex.branch]]
    return liste_terminus


def fonction_terminus(branch, ligne, liste_terminus):
    """
    Rend le terminus en fonction de la ligne et de la branche
    La fonction est utile dans la fonction trouver_terminus quand on observe des lignes avec embranchement
    Elle evite de parcourir toute la ligne
    """
    for liste in liste_terminus:
        if liste[1] == ligne and liste[2] == branch:
            return liste[0]


def trouver_terminus(station1, station2, graph):
    """
    Permet d'obtenir le terminus de la ligne dans le sens station1 vers station2
    """
    liste_terminus = recuperation_terminus(graph)
    station = station2
    station_pred = [station1]

    if station1.branch != station2.branch:  # Si branches différentes entre S1 et S2 on récupère directement le terminus
        return fonction_terminus(station2.branch, station2.line, liste_terminus)

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
                    return fonction_terminus(vertex.branch, vertex.line, liste_terminus)
                    # Si on ne tombe pas sur un embranchement on peut renvoyer n'importe lequel des terminus de la ligne

    return station


def utilisation_dijkstra(nom_depart, nom_arrive, graph):
    # Listes des arrets de meme nom (mais pas de meme ligne) (ex : Chatellet)
    liste_depart = []
    liste_arrive = []
    vertex_depart = vertex_arrive = pred = None

    for vertex in graph.edges.keys():
        if vertex.name == nom_depart:
            liste_depart += [vertex]
        if vertex.name == nom_arrive:
            liste_arrive += [vertex]

    temps = math.inf
    for vertex in liste_depart:
        resultat_dijkstra = dijkstra(graph.edges, vertex)  # tuple : (distances, prédecesseurs)

        for vertex_1 in liste_arrive:
            if resultat_dijkstra[0][vertex_1] < temps:
                temps = resultat_dijkstra[0][vertex_1]
                pred = resultat_dijkstra[1]
                vertex_depart = vertex
                vertex_arrive = vertex_1

    pcc = shortest_path(vertex_depart, vertex_arrive, pred)
    return temps, pcc


def afficher_temps(temps):
    """
    Affiche le temps de trajet en console
    """
    seccondes = temps % 60
    minutes = (temps - seccondes) / 60
    print("Le trajet durera", int(minutes), "m", seccondes, "s (", temps, "s)\n")


def afficher_texte_parcours(pcc, graph):
    """
    Affiche les étapes du trajet en console. pcc est le plus court chemin (liste de Arret) et graph est le graphe
    représentant le réseau de transports
    :param pcc: list
    :param graph: Graph
    """
    print("=== DEBUT DU TRAJET ===")
    i = 0
    var = 0
    while i < len(pcc):

        cond = True
        embranch = False
        while cond and i < len(pcc) - 1:
            if pcc[i].line == pcc[i + 1].line:

                if ((pcc[i - 1].name == "Mirabeau" and pcc[i + 1].name == "Église d'Auteuil") or
                        (pcc[i - 1].name == "Porte d'Auteuil" and pcc[i + 1].name == "Michel Ange Molitor") and i > 0):
                    embranch = True
                    cond = False

                elif ((pcc[i - 1].name == "Place des Fêtes" and pcc[i + 1].name == "Danube") or
                      (pcc[i - 1].name == "Danube" and pcc[i + 1].name == "Place des Fêtes") and i > 0):
                    embranch = True
                    cond = False

                elif pcc[i].branch == pcc[i + 1].branch:  # Cas classique
                    i += 1

                elif ((pcc[i - 1].branch == 1 and pcc[i + 1].branch == 2) or
                      (pcc[i + 1].branch == 1 and pcc[i - 1].branch == 2)):  # Cas embranchement relou
                    embranch = True
                    cond = False

                elif pcc[i + 1].branch == 0 or pcc[i].branch == 0:  # Si on arrive d'un embranchement ou qu'on quitte un embranchement
                    i += 1

                else:
                    cond = False
            else:
                cond = False
        terminus = trouver_terminus(pcc[i - 1], pcc[i], graph).name
        print("Prendre la ligne", pcc[i].line, "direction", terminus, "de", pcc[var].name,
              "jusqu'a", pcc[i].name)
        # Permettre de ne pas sauter un arret en cas d'embranchement
        if embranch:
            var = i
        else:
            var = i + 1

        i += 1
    print("=== FIN DU TRAJET ===\n")


############################################
# Interface graphique                      #
############################################
class App:
    """
    Coeur de l'application
    """
    PCC = 1         # Mode de recherche du plus court chemin
    ACPM = 2        # Mode d'affichage de l'arbre couvrant de poids minimum

    def __init__(self, path):
        # Initialisation de l'affichage
        self.impath = path
        self._init_display()
        plt.axes(self.ax)

        self.start = None
        self.end = None
        self.mode = App.PCC

        self.graph = data_graph()
        self.pos = data_coord()
        self.acpm = []  # Variable destinée

        self.click_callback = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.keyboard_callback = self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def _init_display(self):
        """
        Initialisation de l'affichage, avec l'image du réseau de transport
        """
        im = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.impath))
        self.fig, self.ax = plt.subplots(figsize=plt.figaspect(im))
        self.fig.subplots_adjust(0, 0, 1, 1)
        plt.axis('off')
        self.ax.imshow(im)

    def reset_display(self):
        """
        Réinitialisation de l'affichage, lors d'un changement de mode
        """
        while self.ax.lines:
            self.ax.lines.pop()
        plt.draw()

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
                    self.reset_display()
                    temps, pcc = utilisation_dijkstra(self.start, self.end, self.graph)
                    afficher_texte_parcours(pcc, self.graph)
                    self.afficher_trajet(pcc)
                break

    def on_press(self, event):
        if event.key == ' ':
            self.switch_mode()

    def radio_clicked(self, label):
        if label == "Plus court chemin" and self.mode == App.ACPM:
            self.switch_mode()
        elif label == "Arbre couvrant de poids minimum" and self.mode == App.PCC:
            self.switch_mode()

    def switch_mode(self):
        """
        Changement de mode
        """
        print(f"Changement de {self.mode} à {3 - self.mode}")
        self.mode = 3 - self.mode
        self.reset_display()

        if self.mode == App.ACPM:
            self.display_acpm()
        elif self.mode == App.PCC:
            self.start, self.end = None, None

    def recherche_pos_point(self, name):
        """
        Retourne la position (x, y) du point dont le nom est name
        """
        x, y = 0, 0
        for coord, vertex in self.pos.items():
            if name == vertex:
                x, y = int(coord[0]), int(coord[1])
                break
        return x, y

    def display_acpm(self):
        """
        Affichage de l'arbre couvrant de poids minimum
        """
        if not self.graph.acpm_fait:
            self.acpm = kruskal(self.graph)

        for v, p in self.acpm:
            vx, vy = self.recherche_pos_point(v.name)
            px, py = self.recherche_pos_point(p.name)
            plt.plot((vx, px), (vy, py), color='#0000f0', linewidth=2)

        plt.draw()

    def afficher_trajet(self, pcc):
        """
        Dessin du trajet
        """
        x0, y0, x1, y1 = 0, 0, 0, 0
        x0, y0 = self.recherche_pos_point(pcc[0].name)

        for i in range(len(pcc) - 1):
            if i != 0:
                x0, y0 = x1, y1

            x1, y1 = self.recherche_pos_point(pcc[i + 1].name)
            plt.plot((x0, x1), (y0, y1), color='#0000f0', linewidth=2)
        plt.draw()


if __name__ == '__main__':
    app = App("metrof_r.png")
    plt.show()
