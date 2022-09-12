import pytest
from main import Vertex, Graph, NonOrientedGraph

def test_Vertex_name():
    A = Vertex("vertex")
    assert A.name == "vertex"

def test_Vertex_rank():
    u1 = Vertex("u1")
    u2 = Vertex("u2")
    u3 = Vertex("u3")
    G = NonOrientedGraph([u1,u2,u3])
    G.add_edge(u1, u2, 3)
    G.add_edge(u2, u3, 10)
    G.add_edge(u1, u3, 7)
    assert G.modelisation_graph() == {"u1": {"u2" : 3, "u3" : 7}, "u2": {"u1" : 3, "u3" : 10}, "u3": {"u1" : 7, "u2" : 10}}

