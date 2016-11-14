import numpy
def create_page_rank_markov_chain(links, damping_factor=0.15):
    links = numpy.array(links)
    N = links.max() + 1  
    deg = numpy.zeros(N,dtype=int) 
    for p in links:
        deg[p[0]] += 1
        
    prob_matrix = numpy.zeros([N,N])
    for p in links:
        prob_matrix[p[0]][p[1]] += (1 - damping_factor)/deg[p[0]]
    
    for i in range(N):
        for j in range(N):
            if deg[i] != 0:
                prob_matrix[i][j] += damping_factor/N
            else :
                prob_matrix[i][j] += 1.0/N
    
    return numpy.matrix(prob_matrix)


def page_rank(links, start_distribution, damping_factor=0.15, 
              tolerance=10 ** (-7), return_trace=False):
    
    prob_matrix = create_page_rank_markov_chain(links, 
                                                damping_factor=damping_factor)
    distribution = numpy.matrix(start_distribution)
    trace = []
    if return_trace:
        trace.append(distribution)
    while True :
        distribution2 = distribution * prob_matrix
        if return_trace :
            trace.append(distribution2)
        
        delta = numpy.abs(distribution2 - distribution)
        eps = numpy.max(delta)
        if eps < tolerance:
            break
            
        distribution = distribution2
    
    if return_trace:
        return numpy.array(distribution).ravel(), numpy.array(trace)
    else:
        return numpy.array(distribution).ravel()

