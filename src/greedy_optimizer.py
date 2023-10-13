import numpy as np

def search_optimal_point(space: np.array, precision: float, score_function):
    """Obtém o ponto otimizado da função de pontuação, para o qual ela possui valor 
    máximo, sobre o espaço de busca S discretizado de forma homogênea, com precisão
    p (tamanho de partição). 

    Args:
        space (np.array): espaço de busca, composto de n intervalos (I_1, I_2, ..., I_n),
        em que I_n = [a_i, b_i]
        precision (float): tamanho das partições sobre cada dimensão de S
        score_function (_type_): 

    Returns:
        np.array: ponto otimizado
    """
    discrete_space = discretize_space(space, precision)
    
    optimal_point = discrete_space[0]
    maximum_score = 0
    
    # obtém a pontuação para cada ponto do espaço de busca discretizado
    for point in discrete_space:
        score = score_function(point)
        
        # atualiza o ponto otimizado
        if score > maximum_score:
            optimal_point = point
            maximum_score = score
    
    return optimal_point


def discretize_space(space: np.array, partition_size: float) -> np.array:
    """Discretiza um espaço n-dimensional S em uma coleção discreta de pontos S* = 
    {p_1, p_2, p_3, ...}, a partir de cortes de tamanho e, em cada direção x_i

    Args:
        space (np.array): espaço a ser dividido, composto de n intervalos (I_1, I_2, ..., I_n),
        em que I_n = [a_i, b_i]
        partition_size (float): tamanho das partições em cada dimensão x_i de S

    Returns:
        np.array: coleção de pontos do espaço discretizado  S* = (p_1, p_2, p_3, ...)
    """
    splited_intervals = []
    
    for interval in space:
        splited_int = split_interval(interval, partition_size)
        splited_intervals.append(splited_int)
    
    points = get_all_points(splited_intervals)    
    
    return np.array(points)


def get_all_points(vectors_list: np.array) -> np.array:
    """Dada uma coleção de vetores (v_0, v_1, v_2, ...), v_i = (x_1, x_2, x_3, ...)
    obtém todos os pontos formados pela combinação de suas coordenadas

    Args:
        vectors_list (np.array): lista de vetores (v_0, v_1, v_2, ...), cujas
        coordenadas deseja-se combinar umas com as outras.

    Returns:
        np.array: lista de todos pontos (x_1, x_2, x_3, ...), cujas as coordenadas
        x_i são do vetor v_i 
    """
    v_0 = vectors_list[0]
    points = []
    
    # caso de base, com vectors_list = [V]
    if len(vectors_list) == 1:
        return vectors_list
    
    # montar todos os pontos possíveis com cada coordenada de V[0]
    for coordinate in v_0:
        points_without_v_0 = get_all_points(vectors_list[1:])[0]
        
        # para cada coordenada de v[0], combinar com as coordenadas dos demais
        for point_without_v_0 in points_without_v_0:
            new_point = np.array([coordinate, point_without_v_0])
            points.append(new_point)
    
    return np.array(points)


def split_interval(interval: np.array, partition_size: float) -> np.array:
    """Particiona um intervalo real [a, b] em n pedaços de tamanho s = (b-a)/n, formando 
    um conjunto de pontos da forma {a, a + s, a + 2s, a + 3s, ..., a + (n-1)s}

    Args:
        interval (np.array): _description_
        partition_size (float): _description_

    Returns:
        np.array: _description_
    """
    a, b = interval[0], interval[1]
    
    n_partitions = int((b-a)/partition_size)
    partitions = np.zeros(n_partitions)
    
    for i in range(n_partitions):
        partitions[i] = a + ((i*(b-a))/n_partitions)
    
    return partitions


def test():
    search_space = np.array([[0, 3], [5, 8]])
    
    def reward_funct(x:np.array):
        s = 0
        for e in x: s = s - e
        return s
    
    opt = search_optimal_point(search_space, 0.1, reward_funct)
    
    print(opt)

test()