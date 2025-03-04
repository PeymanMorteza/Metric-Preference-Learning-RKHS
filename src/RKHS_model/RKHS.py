import numpy as np
import os
from src.RKHS_model.kernels import *

class RKHS:
    def __init__(self,kernel,coefs,vecs):
        self.kernel=kernel
        self.coefs=coefs
        self.vecs=vecs
    def __mul__(self,b):
        total=0
        #print(self.coefs)
        for i in range(len(self.coefs)):
            for j in range(len(b.coefs)):
                total+=self.coefs[i]*b.coefs[j]*self.kernel(self.vecs[i],b.vecs[j])
        return total
    def __add__(self,b):
        filtered_vecs = [vec for vec, coef in zip(self.vecs+b.vecs, self.coefs+b.coefs) if (abs(coef) >1e-3) ]
        filtered_coefs = [coef for coef in self.coefs+b.coefs if (abs(coef) >1e-3)]
        from collections import defaultdict

        # Dictionary to accumulate coefficients for each vector
        vector_dict = defaultdict(float)

        for vec, coef in zip(filtered_vecs, filtered_coefs):
            # Convert vector to tuple to use it as a dictionary key
            vec_tuple = tuple(vec)
            vector_dict[vec_tuple] += coef

        # Convert dictionary back to lists
        new_vecs = [np.array(list(vec)) for vec in vector_dict.keys()]
        new_coeffs = list(vector_dict.values())

        return RKHS(self.kernel,new_coeffs,new_vecs)
    def __rmul__(self,c):
        return RKHS(self.kernel,[c*coef for coef in self.coefs],self.vecs)
    def __repr__(self):
        return "coefficients:"+str(self.coefs)+"\n"+"Vectors"+str(self.vecs)

def projection(u,v): #projection of v on u in RKHS
        return (u*v)*(1.0/(u*u))*u

def gram_schmidt(vectors):
    
    # Initialize an empty list to store orthogonalized vectors
    ortho_set = []
    vector_set=[]
    
    # Process the first vector in the set
    ortho_set.append((1.0/((vectors[0]*vectors[0])**0.5))*vectors[0]) 
    vector_set.append(vectors[0])
    
    # Process the remaining vectors
    for i in range(1, len(vectors)):
        #print("i",i)
        v = vectors[i]
        # Compute the orthogonal component of v with respect to previous vectors
        v_len=len(vector_set)
        for j in range(v_len):
            #print(ortho_set[j])
            #print("olen",len(ortho_set[v_len-1]))
            proj=projection(ortho_set[j], v)
            v = ((-1)*proj)+v
        # If the resulting vector is not negligible, add it to the orthogonal set
        if (abs(v*v))**(0.5) > 1e-3:
            my_v=(1.0/ (abs(v*v))**0.5)*v
            #print("my_v",my_v)
            ortho_set.append(my_v)
            vector_set.append(vectors[i]) 
    return ortho_set,vector_set
    
def alpha_reper(ortho_set,vector):
    alpha=[]
    for i in range(len(ortho_set)):
        alpha.append(ortho_set[i]*vector)
    return alpha