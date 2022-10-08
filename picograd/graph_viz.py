from abc import ABC, abstractmethod
from graphviz import Digraph

from picograd.engine import Var


class ComputationalGraphViz(ABC):
    """
        ComputationalGraphViz is the building block object for quick
        visualization of a computational graph. Picograd dynamically
        builds a computational graph and keeps track of the operations
        made while building the computational graph.
        Performing depth-first-search over the computational graph, ComputationalGraphViz
        is able to build a trace from a root node. Usually the root node
        is simply the final result of an operation. For MLP, it is usually the loss function.
    """

    def __init__(self):
        self._nodes = set()
        self._edges = set()

    def create_graph(self, root: Var, rankdir: str = 'LR') -> Digraph:
        """
        Builds the computational graph and displays it
        :param root: root node of type Var
        :param rankdir: TB (top to bottom graph) | LR (left to right)
        :return: Digraph
        """

        self._build_trace(root)
        graph = self._build_graph(rankdir=rankdir)
        return graph

    def _build_trace(self, node: Var) -> None:
        """Performs a recursive depth-first search over the computational graph."""
        if node not in self._nodes:
            self._nodes.add(node)
            for child in node.children:
                self._edges.add((child, node))
                self._build_trace(child)

    @abstractmethod
    def _build_graph(self, rankdir: str) -> Digraph:
        raise NotImplementedError("build_graph() must be implemented for subclasses!")


class ForwardGraphViz(ComputationalGraphViz):

    def __init__(self):
        super(ForwardGraphViz, self).__init__()

    def _build_graph(self, rankdir: str = 'LR') -> Digraph:
        """
        Builds the forward graph
        :param rankdir: TB (top to bottom graph) | LR (left to right)
        :return: Digraph
        """

        assert rankdir in ['LR', 'TB'], f"Unexpected rankdir argument (TB, LR available). Got {rankdir}."
        graph = Digraph(format='png', graph_attr={'rankdir': rankdir})

        for n in self._nodes:
            uid = str(id(n))
            name = n.label if n.label != "" else (n.op + '_res' if n.op else n.label)
            # for any value in the graph, create a rectangular ('record') node for it
            graph.node(name=uid, label=f"{name} | data: {n.data:.4f} | grad: {n.grad:.4f}", shape='record')
            if n.op:
                graph.node(name=uid + n.op, label=n.op)
                graph.edge(uid + n.op, uid)

        for n1, n2 in self._edges:
            # connect n1 to the op node of n2
            if n2.op:
                graph.edge(str(id(n1)), str(id(n2)) + n2.op)
            else:
                graph.edge(str(id(n1)), str(id(n2)))

        return graph
