import numpy as np

def h(X, w, b):
    return np.dot(X, w.T) + b

def loss(y, y_aprox, C, W):
    termino_regularizacion = 0.5 * np.linalg.norm(W)**2
    termino_perdida = C * np.sum(np.maximum(0, 1 - y * y_aprox))
    costo_total = termino_regularizacion + termino_perdida
    return costo_total

def derivatives(x, y, y_aprox, w, b, C):
    F = y * y_aprox
    dL_dw = np.zeros_like(w)
    dL_db = 0
    condition = F < 1
    if np.any(condition):
        dL_dw = w + C * np.sum(-y[condition][:, np.newaxis] * x[condition], axis=0)
        dL_db = C * np.sum(-y[condition])
    else:
        dL_dw = w
    return dL_dw, dL_db

def Update(y, y_aprox, w, b, db, dw, alpha, C):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def training(X, Y, C, alpha, epochs):
    w = np.array([np.random.rand() for i in range(X.shape[1])])
    b = np.random.rand()
    error = []
    for i in range(epochs):
        Y_aprox = h(X, w, b)
        dw, db = derivatives(X, Y, Y_aprox, w, b, C)
        w, b = Update(Y, Y_aprox, w, b, db, dw, alpha, C)
        L = loss(Y, Y_aprox, C, w)
        error.append(L)
    return w, b, error

def training_multiclass(X, Y, C, alpha, epochs, num_classes):
    classifiers = []
    for i in range(num_classes):
        Y_binary = np.where(Y == i, 1, -1)
        w, b, error = training(X, Y_binary, C, alpha, epochs)
        classifiers.append((w, b))
    return classifiers

def testing(X, classifiers):
    y_aprox = np.zeros((X.shape[0], len(classifiers)))
    for i, (w, b) in enumerate(classifiers):
        y_aprox[:, i] = h(X, w, b)
    return np.argmax(y_aprox, axis=1)