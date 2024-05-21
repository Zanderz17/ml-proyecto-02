import numpy as np

"""
Código adaptado y modificado según la plantilla enviada por el profesor en la clase 30/04/24
y tomando como **referencia** el siguiente video: https://www.youtube.com/watch?v=sgQAhG5Q7iY
"""

class Nodo:
 #Define what your data members will be
  def __init__(self, X, Y, index=None, threshold=None):
    self.X = X
    self.Y = Y
    self.index = index
    self.threshold = threshold
    self.left = None
    self.right = None

  def IsTerminal(self):
    # return true if this node has the same labels in Y
    return np.all(self.Y == self.Y.iloc[0])


  def BestSplit(self):
    # Determine the best split for the node data based on Gini impurity
    best_index = None
    best_value = None
    best_score = float("inf")

    for index in range(self.X.shape[1]):  # iterate over each feature
      values = self.X[:, index]
      for value in np.unique(values):
        left_mask = values <= value
        right_mask = values > value
        left_y = self.Y[left_mask]
        right_y = self.Y[right_mask]

        # Calculate weighted Gini for each split
        left_gini = self.Gini(left_y)
        right_gini = self.Gini(right_y)
        weighted_gini = (len(left_y) / len(self.Y)) * left_gini + (len(right_y) / len(self.Y)) * right_gini

        if weighted_gini < best_score:
          best_score = weighted_gini
          best_index = index
          best_value = value

    return best_index, best_value, best_score

  def Entropy(self):
    _, counts = np.unique(self.Y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = - np.sum(probabilities * np.log2(probabilities))
    return entropy

  def Gini(self, Y):
    _ , counts = np.unique(Y, return_counts=True)
    #print(counts)
    probabilities = counts / counts.sum()
    #print(probabilities)
    gini = 1 - np.sum(probabilities**2)
    return gini

class DT:
    def __init__(self, X, Y):
        self.m_Root = None
        self.X = X
        self.Y = Y

    def create_DT(self):
        self.m_Root = self._create_tree(self.X, self.Y)

    def _create_tree(self, X, Y):
        node = Nodo(X, Y)
        if node.IsTerminal():
            return node

        # Utiliza BestSplit que a su vez utiliza Gini dentro de la clase Nodo
        index, value, _ = node.BestSplit()
        if index is None:
            return node

        # Crear nodos hijos basados en la mejor división
        left_mask = X[:, index] <= value
        right_mask = X[:, index] > value
        left_child = self._create_tree(X[left_mask], Y[left_mask])
        right_child = self._create_tree(X[right_mask], Y[right_mask])

        # Configurar el nodo actual con los detalles de la división y los nodos hijos
        node.index = index
        node.threshold = value
        node.left = left_child
        node.right = right_child
        return node

    def predict(self, x):
        # Predice la clase de una observación utilizando el árbol creado
        return self._predict_node(self.m_Root, x)

    def _predict_node(self, node, x):
        if node.left is None and node.right is None:
            # Nodo terminal, retorna la clase más frecuente
            node_y_int = node.Y.astype(np.int64)
            return np.bincount(node_y_int).argmax()
        elif x[node.index] <= node.threshold:
            return self._predict_node(node.left, x)
        else:
            return self._predict_node(node.right, x)
