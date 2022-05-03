#!/usr/bin/env python
# -*- coding: utf-8 -*-


r"""Python module for the Euclidean Random Assignment Problem (ERAP).

## DOCUMENTATION (including licences) TO BE WRITTEN.

References
----------

[1] M, D'Achille "Statistical properties of the Euclidean random assignment problem"
PhD Thesis, Université Paris-Saclay, 2020.

[2] D. Benedetto, E. Caglioti, S. Caracciolo, M. D’Achille, G. Sicuro, and A. Sportiello, “Random Assignment Problems on 2d Manifolds,” J. Stat. Phys., vol. 183, no. 2, p. 34, May 2021.


"""

import pandas as pd
import numpy as np
from lapjv import lapjv
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import math, random



__author__ = 'Matteo D\'Achille'
__copyright__ = 'Copyright Matteo D\'Achille and [other contributors]'
__credit__ = 'Matteo D\'Achille [other contributors]'
__status__ = 'beta'
__version__ = '0.0.1'


## GENERAL UTILS

def isSquare(n):
    ## See: https://stackoverflow.com/questions/2489435/check-if-a-number-is-a-perfect-square
    ## Trivial checks
    if type(n) != int:  ## integer
        return False
    if n < 0:      ## positivity
        return False
    if n == 0:      ## 0 pass
        return True

    ## Reduction by powers of 4 with bit-logic
    while n&3 == 0:
        n=n>>2

    ## Simple bit-logic test. All perfect squares, in binary,
    ## end in 001, when powers of 4 are factored out.
    if n&7 != 1:
        return False

    if n==1:
        return True  ## is power of 4, or even power of 2


    ## Simple modulo equivalency test
    c = n%10
    if c in {3, 7}:
        return False  ## Not 1,4,5,6,9 in mod 10
    if n % 7 in {3, 5, 6}:
        return False  ## Not 1,2,4 mod 7
    if n % 9 in {2,3,5,6,8}:
        return False
    if n % 13 in {2,5,6,7,8,11}:
        return False

    ## Other patterns
    if c == 5:  ## if it ends in a 5
        if (n//10)%10 != 2:
            return False    ## then it must end in 25
        if (n//100)%10 not in {0,2,6}:
            return False    ## and in 025, 225, or 625
        if (n//100)%10 == 6:
            if (n//1000)%10 not in {0,5}:
                return False    ## that is, 0625 or 5625
    else:
        if (n//10)%4 != 0:
            return False    ## (4k)*10 + (1,9)


    ## Babylonian Algorithm. Finding the integer square root.
    ## Root extraction.
    s = (len(str(n))-1) // 2
    x = (10**s) * 4

    A = {x, n}
    while x * x != n:
        x = (x + (n // x)) >> 1
        if x in A:
            return False
        A.add(x)
    return True

def aggiusta_v2(x,l=1):
    if (x>l/2.):
        return x-l
    elif(x<-l/2.):
        return x+l
    else:
        return x


Ncycles = lambda l:len(set(reduce(lambda l,_:[min(x,l[x])for x in l],l,l)))

def radial_profile(data, center):
    # See https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile



def aa(d):
    '''Computes the parameter \alpha as a function of d'''
    return (2**(d-1)-1)**(1./d)

def z2ri(z):
    return np.array([z.real, z.imag])

def rd4(probscum):
    '''Samples the index by inverse transform sampling (probscum)'''
    rdu = np.random.uniform(0, 1);
    i = 0;
    while(rdu > probscum[i]):
        i+=1
    return(i)


# DISTANCE FUNCTIONS

def dist_toro(l1,l2,blues,reds):
    """Calculates the squared euclidean distance on the torus"""

    proj_x = distance.cdist(blues.T[0].reshape(-1,1),reds.T[0].reshape(-1,1))
    proj_y = distance.cdist(blues.T[1].reshape(-1,1),reds.T[1].reshape(-1,1))

    cost_matrix = (np.minimum(proj_x,l1*np.ones_like(proj_x)-proj_x))**2+(np.minimum(proj_y,l2*np.ones_like(proj_y)-proj_y))**2

    return cost_matrix

def dist_toro_alpha(l1,l2,blues,reds,alpha=1):
    """Calculates the squared euclidean distance on the torus"""

    proj_x = distance.cdist(blues.T[0].reshape(-1,1),reds.T[0].reshape(-1,1))
    proj_y = distance.cdist(blues.T[1].reshape(-1,1),reds.T[1].reshape(-1,1))

    cost_matrix = (np.minimum(proj_x,l1*np.ones_like(proj_x)-proj_x))**2+alpha*(np.minimum(proj_y,l2*np.ones_like(proj_y)-proj_y))**2

    return cost_matrix

def dist_cilindro(l1,l2,blues,reds):
    """Calculates the squared euclidean distance on the cylinder"""

    proj_x = distance.cdist(blues.T[0].reshape(-1,1),reds.T[0].reshape(-1,1))
    proj_y = distance.cdist(blues.T[1].reshape(-1,1),reds.T[1].reshape(-1,1))

    cost_matrix = (np.minimum(proj_x,l1*np.ones_like(proj_x)-proj_x))**2+proj_y**2

    return cost_matrix

def dist_moebius(l1,l2,blues,reds):
    """Calculates the squared euclidean distance on the Moebius strip"""

    b_xx = blues.T[0].reshape(-1,1)
    b_yy = blues.T[1].reshape(-1,1)


    r_xx = reds.T[0].reshape(-1,1)
    r_yy = reds.T[1].reshape(-1,1)


    eu_dist = distance.cdist(b_xx,r_xx)**2+distance.cdist(b_yy,r_yy)**2 # Non gira attorno

    m_dist_plus = distance.cdist(b_xx,r_xx+l1)**2+distance.cdist(b_yy,l2-r_yy)**2 # Ha girato a destra

    m_dist_minus = distance.cdist(b_xx,r_xx-l1)**2+distance.cdist(b_yy,l2-r_yy)**2 # Ha girato intorno a sinistra

    cost_matrix  = np.min(np.array([eu_dist,m_dist_plus,m_dist_minus]),axis=0)

    return cost_matrix

def dist_klein(l1,l2,blues,reds):
    """Calculates the squared euclidean distance on the Klein bottle"""

    b_xx = blues.T[0].reshape(-1,1) ## Blue Projections
    b_yy = blues.T[1].reshape(-1,1)


    r_xx = reds.T[0].reshape(-1,1) ## Red Projections
    r_yy = reds.T[1].reshape(-1,1)


    eu_dist = distance.cdist(b_xx,r_xx)**2+distance.cdist(b_yy,r_yy)**2 # Non gira attorno

    m_1_plus = distance.cdist(b_xx,r_xx+l1)**2+distance.cdist(b_yy,l2-r_yy)**2 # Ha girato a destra

    m_1_minus = distance.cdist(b_xx,r_xx-l1)**2+distance.cdist(b_yy,l2-r_yy)**2 # Ha girato intorno a sinistra

    m_2_plus = distance.cdist(b_xx,r_xx)**2+distance.cdist(b_yy,r_yy+l2)**2 # Ha girato intorno sopra

    m_2_minus = distance.cdist(b_xx,r_xx)**2+distance.cdist(b_yy,r_yy-l2)**2 # Ha girato intorno sotto


    m_ne =  distance.cdist(b_xx,r_xx+l1)**2+distance.cdist(b_yy,l2-r_yy+l2)**2

    m_nw =  distance.cdist(b_xx,r_xx-l1)**2+distance.cdist(b_yy,l2-r_yy+l2)**2

    m_se =  distance.cdist(b_xx,r_xx+l1)**2+distance.cdist(b_yy,l2-r_yy-l2)**2

    m_sw =  distance.cdist(b_xx,r_xx-l1)**2+distance.cdist(b_yy,l2-r_yy-l2)**2


    cost_matrix  = np.min(np.array([eu_dist,m_1_plus,m_1_minus,m_2_plus,m_2_minus,m_ne,m_nw,m_se,m_sw]),axis=0)

    return cost_matrix

def dist_RP2(l1,l2,blues,reds):
    """Calculates the squared euclidean distance on the real projective plane RP2"""

    b_xx = blues.T[0].reshape(-1,1) ## Blue Projections
    b_yy = blues.T[1].reshape(-1,1)


    r_xx = reds.T[0].reshape(-1,1) ## Red Projections
    r_yy = reds.T[1].reshape(-1,1)


    eu_dist = distance.cdist(b_xx,r_xx)**2+distance.cdist(b_yy,r_yy)**2 # Non gira attorno

    E = distance.cdist(b_xx,r_xx+l1)**2+distance.cdist(b_yy,l2-r_yy)**2 # Ha girato a destra

    W = distance.cdist(b_xx,r_xx-l1)**2+distance.cdist(b_yy,l2-r_yy)**2 # Ha girato intorno a sinistra

    N = distance.cdist(b_xx,l1-r_xx)**2+distance.cdist(b_yy,r_yy+l2)**2 # Ha girato intorno sopra

    S = distance.cdist(b_xx,l1-r_xx)**2+distance.cdist(b_yy,r_yy-l2)**2 # Ha girato intorno sotto


    NE =  distance.cdist(b_xx,2*l1-r_xx)**2+distance.cdist(b_yy,2*l2-r_yy)**2

    NW =  distance.cdist(b_xx,-r_xx)**2+distance.cdist(b_yy,2*l2-r_yy)**2

    SE =  distance.cdist(b_xx,2*l1-r_xx)**2+distance.cdist(b_yy,-r_yy)**2

    SW =  distance.cdist(b_xx,-r_xx)**2+distance.cdist(b_yy,-r_yy)**2


    cost_matrix  = np.min(np.array([eu_dist,E,W,N,S,NE,NW,SE,SW]),axis=0)

    return cost_matrix


def dist_s1(blues,reds,r=1./np.pi):
    return (r*np.arccos(1.-distance.cdist(blues,reds,'cosine')))**2


def dist_sphere(blues,reds,radius=1./np.sqrt(4*np.pi)):
    return (radius*np.arccos(1.-distance.cdist(blues,reds,'cosine')))**2


def dist_antidisk(blues,reds,radius=1./np.sqrt(np.pi)):
    d_ref = distance.cdist(blues,reds,'sqeuclidean')
    d_turn = distance.cdist(blues,np.zeros_like(blues),'sqeuclidean')\
           + distance.cdist(np.zeros_like(reds),reds,'sqeuclidean')\
           + 2*radius*distance.cdist(blues,reds,'euclidean')\
           + radius**2*np.ones_like(d_ref)
    return np.min(np.array([d_ref,d_turn]),axis=0)


# ROUTINES FOR GENERATING BLUE AND/OR RED POINTS

def us1(n,r=1./np.pi):
    v = np.random.normal(size=(n,2))
    norm = (np.linalg.norm(v,axis=1)*np.ones_like(v).T).T
    return r*v/norm

def uD1(n,r = 1./np.pi): # random variables following the uniform distribution over a disk
    radii = np.random.uniform(0,r,n)
    thetas = np.random.uniform(0,2*np.pi,n)
    X = np.sqrt(radii)*np.cos(thetas)
    Y = np.sqrt(radii)*np.sin(thetas)
    return np.array([X,Y]).T

def uA1(n,r_larger = 1./np.pi,r_smaller= 0.5/np.pi): # random variables following the uniform distribution over an annuluse 
    radii = np.random.uniform(r_smaller,r_larger,n)
    thetas = np.random.uniform(0,2*np.pi,n)
    X = np.sqrt(radii)*np.cos(thetas)
    Y = np.sqrt(radii)*np.sin(thetas)
    return np.array([X,Y]).T

def us2(n,r=1./np.sqrt(4*np.pi)):
    v = np.random.normal(size=(n,3))
    norm = (np.linalg.norm(v,axis=1)*np.ones_like(v).T).T
    return r*v/norm


def sunflower(n,R=1/np.sqrt(np.pi)):
    indices = np.arange(0, n, dtype=float) + 0.5
    r = np.sqrt(indices/n)
    theta = np.pi * (1 + 5**0.5) * indices
    points = R*np.array([r*np.cos(theta),r*np.sin(theta)])
    return points.T

def sfera_new(n,r=1./np.sqrt(4*np.pi)):
    u = 2*np.random.uniform(size=n)-1
    phis = np.random.uniform(0,2*np.pi,size=n)
    z = u
    x = np.cos(phis)*np.sqrt(1-z**2)
    y = np.sin(phis)*np.sqrt(1-z**2)
    points = np.array([x,y,z]).T
    return r*points

def semisfera_new(n,r=1./np.sqrt(2*np.pi)):
    u = 2*np.random.uniform(size=n)-1
    phis = np.random.uniform(0,2*np.pi,size=n)
    z = u*np.sign(u)
    x = np.cos(phis)*np.sqrt(1-z**2)
    y = np.sin(phis)*np.sqrt(1-z**2)
    points = np.array([x,y,z]).T
    return r*points


def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return (1/(2*np.sqrt(np.pi)))*np.array(points)

def fibonacci_semi_sphere(size,R=1/(np.sqrt(2*np.pi))):
    points = []
    n = 2*size
    offset = 2./n
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + 1.) % n) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r
        #z = z*np.sign(z)

        points.append([x,y,z])

    all_points = R*np.array(points)

    xdata,ydata,zdata = all_points.T

    mask = zdata>=0

    x = xdata[mask]
    y = ydata[mask]
    z = zdata[mask]
    return np.array([x,y,z]).T


def rect_dis(l1,l2,n):
    """n points from the uniform distribution on the rectangle l1xl2
    WARNING: for computational convenience in the distance functions, the case l1=l2=sqrt(n) corresponds by convention to l1=l2=1.
    """
    Xs = l1*np.random.uniform(size=n)
    Ys = l2*np.random.uniform(size=n)
    return np.array([Xs,Ys]).T
def griglia(l1,l2,n):
    if (isSquare(n)):
        onedsize = round(np.sqrt(n))
        x = np.linspace(1./(2*onedsize),1-1./(2*onedsize),onedsize);
        y = np.linspace(1./(2*onedsize),1-1./(2*onedsize),onedsize);
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    else:
        return np.nan

def griglia_v2(l1,l2,n):
    """Generates rectangular lattice at semi-integer coordinates"""
    if (isSquare(n) and l1==1 and l2==1): ## Torus with aspect ratio 1:1
        onedsize = round(np.sqrt(n))
        x = np.linspace(1./(2*onedsize),1-1./(2*onedsize),onedsize);
        y = np.linspace(1./(2*onedsize),1-1./(2*onedsize),onedsize);
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T

    elif (n==l1*l2): ## Torus with arbitrary aspect ratio
        x = l1*np.linspace(1./(2*l1),1-1./(2*l1),l1);
        y = l2*np.linspace(1./(2*l2),1-1./(2*l2),l2);
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    else:
        return np.nan


def griglia_toro(l1,l2,n):
    """Generates a l1xl2 rectangular lattice at ``integer'' coordinates"""
    if (isSquare(n) and l1==1 and l2==1): ## Case of torus with aspect ratio 1:1
        L = int(round(np.sqrt(n)))
        x = np.linspace(0,L-1,L)/L;
        y = np.linspace(0,L-1,L)/L;
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T

    elif (n==l1*l2): ## Case of torus with arbitrary aspect ratio
        x = np.linspace(0,l1-1,l1);
        y = np.linspace(0,l2-1,l2);
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    else:
        return np.nan

def griglia_quadrata_dirichlet(l1,l2,n):
    '''Sia L=sqrt(n). Genero L+2 punti equispaziati in [0,1] (in ogni direzione), e poi rimuovo sia {0} che {1}'''
    if (isSquare(n) and l1==1 and l2==1):
        L = round(np.sqrt(n))
        x = np.linspace(1/2.,L-1/2.,L)/L;
        y = np.linspace(1/2.,L-1/2.,L)/L;
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    elif (n==l1*l2): ## Case of a rectangle with arbitrary aspect ratio
        x = np.linspace(1/2.,l1-1/2.,l1);
        y = np.linspace(1/2.,l2-1/2.,l2);
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    else:
        return np.nan


def griglia_cilindro(l1,l2,n):
    '''Points on the periodic direction at integer coordinates, points on the OBC directions at semi-integer positions'''
    if (isSquare(n) and l1==1 and l2==1):
        L = round(np.sqrt(n))
        x = np.linspace(0,L-1,L)/L;
        y = np.linspace(1/2.,L-1/2.,L)/L;
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    elif (n==l1*l2): ## Case of a rectangle with arbitrary aspect ratio
        x = np.linspace(0,l1-1,l1);
        y = np.linspace(1/2.,l2-1/2.,l2);
        xx,yy = np.meshgrid(x, y)
        return np.array([xx.flatten(),yy.flatten()]).T
    else:
        return np.nan


def general_grid(Lx,Ly):
    delta = 1/(2*Lx)
    x0 = np.linspace(0,Lx-1,Lx)/Lx
    y0 = np.linspace(0,Ly-1,Ly)/Ly
    x, y = np.meshgrid(x0, y0)
    x[1::2] += delta
    return np.array([x.flatten(),y.flatten()]).T




def k_grid(b_points,n,k):
    if(n%k != 0):
        raise ValueError("n is not divisible by k!")
    else:
        choices = np.random.choice(range(n),size=int(n/k))
        germs = b_points[choices]
        r_points = np.repeat(germs,k,axis=0)
    return r_points



def fai_punto_uno(stat_vecs):
    '''Generate a random point uniformly distributed on...'''
    sys = [1,0];

    N = 125

    lnn = int(np.ceil(2*np.log(N)))

    fatt = stat_vecs[0]

    gira = stat_vecs[1]

    vec4 = stat_vecs[2]

    probscum = stat_vecs[3]

    for i in range(lnn):

        rd = rd4(probscum);

        sys = [sys[0]*fatt[rd]*gira[rd],sys[1] + sys[0]*vec4[rd]];

    punto = z2ri(sys[1]);

    if(np.random.randint(0,2) == 1):
        punto[1] = -punto[1];

    return punto

def gen_points_type1(n_points,dim):
    '''Generation of n_points of type1 with dimension dim'''

    a = aa(dim)
    probs = np.array([1.,a**dim,a**dim, 1.])
    probs = probs/np.sum(probs)
    probscum_in = np.cumsum(probs)
    a_pippo = a

    fatt_in = [1./2,a/2.,a/2.,1/2.]
    gira_in = [1.,complex(0,1),complex(0,-1),1.]    ## Here j is the complex unity
    vec4_in = [0/2., 1/2., complex(1,a)/2., 1/2.]

    Vs = [fatt_in,gira_in,vec4_in,probscum_in]
    points = []

    for n in range(n_points):
        points.append(fai_punto_uno(Vs))
    return points



"Koch-Cèsaro like fractals"


def fai_punto_due(stat_vs):
    '''Generate a random point uniformly distributed on...'''

    N = 125

    lnn = int(np.ceil(2*np.log(N)))


    sys = [1,0];

    fatt = stat_vs[0]

    gira = stat_vs[1]

    vec4 = stat_vs[2]

    for i in range(lnn):

        rd = np.random.randint(0, 4);

        sys = [sys[0]*fatt[rd]*gira[rd],sys[1] + sys[0]*vec4[rd]]

    punto = z2ri(sys[1]);

    if(np.random.randint(0,2) == 1):
        punto[1] = -punto[1];

    return punto

def gen_points_type2(n_points,alpha):
    '''Generation of n_points of type2 with bending angle alpha'''

    c_unit = complex(0.0, 1.0)
    fatt_2 = [1./(2. + 2*np.cos(alpha)) for i in range(4)]
    fase_2 = np.exp(c_unit*alpha);
    gira_2 = [1., fase_2, 1./fase_2, 1.];
    vec4_2 = [0., 1., 1 + np.cos(alpha) + c_unit*np.sin(alpha), 1 + 2*np.cos(alpha)]

    Vs = [fatt_2,gira_2,vec4_2]
    points = []

    for n in range(n_points):
        points.append(fai_punto_due(Vs))
    return points


def d_H_cesaro(alpha):

    """Hausdorff dimension of type 2 fractal

    Parameters
    ----------
    alpha: float
        bending angle
    Returns
    -------
    d_hauss: float
        Haussdorf dimension
    """

    d_hauss = 2./(1+np.log(1+np.cos(alpha))/np.log(2))
    return d_hauss

def alpha_cesaro(dH):
    """Bending angle of Cesàro fractal of given Haussdorf dimension

    Parameters
    ----------
    dH: float
        Haussdorf dimension
    Returns
    -------
    alpha: float
        bending angle
    """
    alpha = np.arccos(2**((2.-dH)/dH)-1.)
    return alpha


## Operations on fields on a square lattice (assuming PBC)


def nabla_x_minus(field_comp):
    return field_comp-np.roll(field_comp,1,axis=0) # Matrix -  matrix  shifted `left' (axis=0) by one (-1)

def nabla_x_plus(field_comp):
    return np.roll(field_comp,-1,axis=0)-field_comp # Matrix -  matrix  shifted `left' (axis=0) by one (-1)


def nabla_y_minus(field_comp):
    return field_comp-np.roll(field_comp,1,axis=1) # Matrix -  matrix  shifted `left' (axis=0) by one (-1)

def nabla_y_plus(field_comp):
    return np.roll(field_comp,-1,axis=1)-field_comp # Matrix -  matrix  shifted `left' (axis=0) by one (-1)


def Div(mux,muy):
    """Discrete divergence operator according to the new Andrea's receipt (18-3-2020) """
    A = np.roll(mux,(-1,-1),axis=(0,1))+np.roll(mux,(-1,0),axis=(0,1))-np.roll(mux,(0,0),axis=(0,1))-np.roll(mux,(0,-1),axis=(0,1))
    B = np.roll(muy,(-1,-1),axis=(0,1))-np.roll(muy,(-1,0),axis=(0,1))-np.roll(muy,(0,0),axis=(0,1))+np.roll(muy,(0,-1),axis=(0,1))
    return (A+B)/2.

def Rot(mux,muy):
    """Discrete curl operator according to the new Andrea's receipt (18-3-2020) """
    A = -np.roll(mux,(-1,-1),axis=(0,1))+np.roll(mux,(-1,0),axis=(0,1))+np.roll(mux,(0,0),axis=(0,1))-np.roll(mux,(0,-1),axis=(0,1))
    B = np.roll(muy,(-1,-1),axis=(0,1))+np.roll(muy,(-1,0),axis=(0,1))-np.roll(muy,(0,0),axis=(0,1))-np.roll(muy,(0,-1),axis=(0,1))
    return (A+B)/2.



## Phats and other utils for rectangular lattices


def shift1(n):
    L = int(np.sqrt(n))
    mat = [[np.exp(-1j*np.pi*n1/float(L)) for q in range(L)] for n1 in range(L)]
    return np.array(mat)

def shift2(n):
    L = int(np.sqrt(n))
    mat = [[np.exp(-1j*np.pi*n2/float(L)) for n2 in range(L)] for p in range(L)]
    return np.array(mat)


def phatsquare(n):

    L = int(np.sqrt(n))
    phatq = [[4*(np.sin(np.pi*p/L)**2+np.sin(np.pi*q/L)**2) for p in range(L)] for q in range(L)]
    phatq[0][0]=1
    return np.array(phatq)


def phat1(n):
    L = int(np.sqrt(n))
    mat = [[2*np.sin(p*np.pi/float(L)) for q in range(L)] for p in range(L)]
    return np.array(mat)

def phat2(n):
    L = int(np.sqrt(n))
    mat = [[2*np.sin(q*np.pi/float(L)) for q in range(L)] for p in range(L)]
    return np.array(mat)

def shiftrot(n):
    L = int(np.sqrt(n))
    mat = [[np.exp(1j*np.pi*(n1+n2)/L) for n2 in range(L)] for n1 in range(L)]
    return np.array(mat)


def phat_1(n):
    L = int(np.sqrt(n))
    mat = [[2*np.sin(p*np.pi/float(L)) for q in range(L)] for p in range(L)]
    return np.array(mat)

def phat_1_r(l1,l2,n):
    mat = [[2*np.sin(p*np.pi/float(l1)) for q in range(l2)] for p in range(l1)]
    return np.array(mat)

def phat_2(n):
    L = int(np.sqrt(n))
    mat = [[2*np.sin(q*np.pi/float(L)) for q in range(L)] for p in range(L)]
    return np.array(mat)

def phat_2_r(l1,l2,n):
    mat = [[2*np.sin(q*np.pi/float(l2)) for q in range(l2)] for p in range(l1)]
    return np.array(mat)


def squared_phat(n):
    L = int(np.sqrt(n))
    phatq = [[4*(np.sin(np.pi*p/L)**2+np.sin(np.pi*q/L)**2) for p in range(L)] for q in range(L)]
    return np.array(phatq)


def modified_squared_phat(n):
    """Rotated laplacian in p space"""
    return squared_phat(n)-(phat_1(n)**2*phat_2(n)**2)/2.



def inv_phatq_r(l1,l2,n):
    return np.nan_to_num(1./(phat_1_r(l1,l2,n)**2+phat_2_r(l1,l2,n)**2),posinf=0,neginf=0)



def hatp(n):
    L = int(np.sqrt(n))
    hatpq = [[4*(np.sin(np.pi*p/L)**2+np.sin(np.pi*q/L)**2) for p in range(L)] for q in range(L)]
    return np.array(hatpq)

def hatpmod(n):
    appo = hatp(n)
    appo[0][0] = 1
    return appo


## THE INSTANCE CLASS

class Instance:
    '''An instance of the ERAP at d=2.

    Attributes:
        n     number of points
        p     cost function exponent (default p=2)
        l1    length of side along x axis
        l2    length of side along y axis
    '''

    def __init__(self, n, p=2,\
                 model='toro', kind='pp',\
                 l1=1,l2=1,\
                 solve=True,\
                 blues = np.nan,\
                 reds =  np.nan,\
                 k = np.nan,\
                 dH = 1.666,\
                 s = 1,\
                 l1_blue=1, l2_blue=1, l1_red=1, l2_red=1,\
                 Radius_larger_blue=1,Radius_larger_red=1,Radius_smaller_blue=0,Radius_smaller_red=0): 
        ''' Standard constructor for an Instance object

        Parameters:
        n: int
            number of points
        model: string (optional, default='toro')
            specifies the kind of boundary conditions: quadrato, toro, cilindro, moebius, klein, RP2, Blue_square_Red_rectangle, Blue_disk_Red_disk, Blue_disk_Red_annuluse  
        l1: float (optional, default=1)
            length of domain side along 'x' axis
        l2: float (optional, default=1)
            length of domain side along 'y' axis
        p: float (optional, default=2)
            the energy-distance exponent in the model
        solve: bool (default=True)
            whether to solve the instance at instantiation or not (useful for debug)
        kind: string (default='pp')
            decides the kind of randomness, between Poisson-Posson ('pp'), Grid-Poisson ('gp'), k Grid-Poisson ('kgp')
        '''
        self.n = n
        self.l1 = l1
        self.l2 = l2
        self.p = p
        self.model = model
        self.s = s # Attractive (s=-1) or repulsive (s=+1) where applicable
        self.l1_blue = l1_blue
        self.l2_blue = l2_blue
        self.l1_red = l1_red
        self.l2_red = l2_red
        self.Radius_larger_blue=Radius_larger_blue
        self.Radius_larger_red=Radius_larger_red
        self.Radius_smaller_blue=Radius_smaller_blue
        self.Radius_smaller_red=Radius_smaller_red
        ## Allocation of blue points
        if (np.isnan(blues).any()): ## New instance is generated according to kind
            if (kind=='pp'):     
                if (model=='Blue_square_Red_rectangle'):
                    self.blues = rect_dis(self.l1_blue,self.l2_blue,n) 
                elif (model=='Blue_disk_Red_disk'):
                    self.blues = uD1(n,r=Radius_larger_blue) 
                elif (model=='Blue_disk_Red_annuluse'):
                    self.blues = uA1(n,r_larger=Radius_larger_blue,r_smaller=Radius_smaller_blue) 
                else:
                    self.blues = rect_dis(l1,l2,n) ## Easiest one
                self.kind = 'pp'

            elif (kind=='gp'):
                if(not isSquare(n) and not self.l1*self.l2 == self.n):
                    raise ValueError('n should be a square in the Grid-Poisson problem!')
                    ## TODO: this only tells the user that some error is going on (i.e. n not a square), but still allocates the object
                else:
                    if (self.model=='toro' or self.model=='klein' or self.model=='RP2'):
                        #self.blues = griglia(l1,l2,n)
                        #self.blues = griglia_v2(l1,l2,n)
                        self.blues = griglia_toro(l1,l2,n)

                    elif (self.model=='cilindro' or self.model=='moebius'):
                        self.blues = griglia_cilindro(l1,l2,n)

                    elif (self.model=='quadrato'):
                        self.blues = griglia_quadrata_dirichlet(l1,l2,n)

                    self.kind = 'gp'

            elif (kind=='kgp'):
                if(not isSquare(n)):
                    raise ValueError('n should be a square in the k-Grid-Poisson problem!')

            elif (kind=='sierpinski'):

                self.blues = np.array(gen_points_type1(n,dH))

            elif (kind=='kochcesaro'):

                self.blues = np.array(gen_points_type2(n,alpha_cesaro(dH)))

            else:

                    self.blues = griglia_v3(l1,l2,n)

        else:  ## That is, if you input blue, allocates the input blues
            if(blues.shape[0]!=self.n):
                raise ValueError('n and of number of input blue points should be at least identical!')
            else:
                self.blues = blues
                self.kind ='gp'

        ## Allocation of red points

        if (np.isnan(reds).any()): ## If red points are not given, generate them according to model

            if (kind=='pp' or kind =='gp'):
                if (model=='Blue_square_Red_rectangle'):
                    self.reds = rect_dis(self.l1_red,self.l2_red,n) 
                elif (model=='Blue_disk_Red_disk'):
                    self.reds = uD1(n,r=Radius_larger_red) 
                elif (model=='Blue_disk_Red_annuluse'):
                    self.reds = uA1(n,r_larger=Radius_larger_red,r_smaller=Radius_smaller_red) 
                else:
                    self.reds = rect_dis(l1,l2,n)

            if (kind=='kgp'):

                if (np.isnan(k)):
                    raise ValueError('Please specify k in the k-Grid-Poisson problem!')
                else:
                    self.reds = k_grid(self.blues,n,k)
                    self.kind = 'kgp'


            elif (kind=='sierpinski'):

                self.reds = np.array(gen_points_type1(n,dH))
                self.kind = 'sierpinski'
                self.dH = dH

            elif (kind=='kochcesaro'):

                self.reds = np.array(gen_points_type2(n,alpha_cesaro(dH)))
                self.kind = 'kochcesaro'
                self.dH = dH

        else: ## Allocate input reds, after some control

            if(reds.shape[0]!=self.n):
                raise ValueError('n and of number of input red points should be at least identical!')

            else:
                self.reds = reds ## Assign input reds

        self.is_solved = False
        self.has_cmat = False
        if(solve):
            self.solve()

    def describe(self):
        return "An istance at size {} in the {} model".format(self.n, self.model)

    ### BASIC OPERATIONS

    def compute_cost_matrix(self):
        '''
        Computes the cost matrix (with appropriate boundary conditions) to be given to the LAPJV algorithm
        '''
        if (self.p==2):
            if(self.model=='toro'):
                self.cmat = dist_toro(self.l1,self.l2,self.blues,self.reds)
            elif(self.model=='cilindro'):
                self.cmat = dist_cilindro(self.l1,self.l2,self.blues,self.reds)
            elif(self.model=='moebius'):
                self.cmat = dist_moebius(self.l1,self.l2,self.blues,self.reds)
            elif(self.model=='klein'):
                self.cmat = dist_klein(self.l1,self.l2,self.blues,self.reds)
            elif(self.model=='quadrato' or self.model=='Blue_square_Red_rectangle' or self.model=='Blue_disk_Red_disk' or self.model=='Blue_disk_Red_annuluse'):
                self.cmat = distance.cdist(self.blues,self.reds,metric='sqeuclidean')
            elif(self.model=='RP2'):
                 self.cmat = dist_RP2(self.l1,self.l2,self.blues,self.reds)
        else:
            if(self.model=='toro'):
                if (self.p==0):
                    self.cmat = np.log(dist_toro(self.l1,self.l2,self.blues,self.reds)**.5)
                else:
                    self.cmat = (dist_toro(self.l1,self.l2,self.blues,self.reds))**(self.p/2.)
            elif(self.model=='quadrato'or self.model=='Blue_square_Red_rectangle' or self.model=='Blue_disk_Red_disk' or self.model=='Blue_disk_Red_annuluse'):
                if (self.p==0):
                    self.cmat = np.log(distance.cdist(self.blues,self.reds,metric='euclidean'))
                else:
                    self.cmat = distance.cdist(self.blues,self.reds,metric='sqeuclidean')**(self.p/2.)
            elif(self.model=='cilindro'):
                if (self.p==0):
                    self.cmat = np.log(dist_cilindro(self.l1,self.l2,self.blues,self.reds))
                else:
                    self.cmat =  dist_cilindro(self.l1,self.l2,self.blues,self.reds)**(self.p/2.)

            elif(self.model=='moebius'):
                if (self.p==0):
                    self.cmat = np.log(dist_moebius(self.l1,self.l2,self.blues,self.reds))
                else:
                    self.cmat = dist_moebius(self.l1,self.l2,self.blues,self.reds)**(self.p/2.)
            elif(self.model=='klein'):
                if (self.p==0):
                    self.cmat = np.log(dist_klein(self.l1,self.l2,self.blues,self.reds))
                else:
                    self.cmat = dist_klein(self.l1,self.l2,self.blues,self.reds)**(self.p/2.)

            elif(self.model=='RP2'):
                if (self.p==0):
                    self.cmat = np.log(dist_RP2(self.l1,self.l2,self.blues,self.reds))
                else:
                    self.cmat = dist_RP2(self.l1,self.l2,self.blues,self.reds)**(self.p/2.)



        if (self.p<0):

            self.cmat = self.s*self.cmat

        self.has_cmat = True



    def solve(self):
        '''Solves the instance using the Jonker-Volgenant algorithm'''
        if (self.has_cmat):
            self.solution = lapjv(self.cmat)
            self.is_solved = True
        else:
            self.compute_cost_matrix()
            self.solution = lapjv(self.cmat) ## The solution step
            self.is_solved = True
        if (self.l1==1 and self.l2 == 1):
            self.cost_min = self.solution[2][0]
        else:
            self.cost_min = self.solution[2][0]/self.n
        self.opt_perm = self.solution[0] ## \pi^*
        #self.a = self.solution[2][1]
        #self.b = self.solution[2][1]## First cavity field (on the blues)
        provv = int(np.sqrt(self.n))
        #self.a = np.reshape(self.solution[2][1],(provv,provv),order='F')
        #self.b = np.reshape(self.solution[2][2],(provv,provv),order='F') ## Second cavity field (on the reds)
        tf_naive = self.reds[self.opt_perm]-self.blues

        if (self.model == 'toro' or self.model=='klein' or self.model=='RP2'): ## Adjusted transport field on the torus
            agg_x = partial(aggiusta_v2, l=self.l1)
            agg_y = partial(aggiusta_v2, l=self.l2)
            dX = list(map(agg_x,tf_naive.T[0]))
            dY = list(map(agg_y,tf_naive.T[1]))

        if (self.model=='cilindro' or self.model=='moebius'):
            agg_x = partial(aggiusta_v2, l=self.l1)
            dX = list(map(agg_x,tf_naive.T[0]))
            dY = tf_naive.T[1]

        elif (self.model=='quadrato'or self.model=='Blue_square_Red_rectangle' or self.model=='Blue_disk_Red_disk' or self.model=='Blue_disk_Red_annuluse'):
            dX = list(tf_naive.T[0])
            dY = list(tf_naive.T[1])

        self.transp_field = np.array([dX,dY]).T

        self.angles1d = np.arctan2(self.transp_field.T[1],self.transp_field.T[0])


    def get_otf_components(self):
        '''
        Get the optimal transport field as two scalar fields (matrices)
        '''

        if (self.kind == 'gp' and (self.model=='toro' or self.model=='quadrato' or self.model=='cilindro')):
            if (not self.is_solved):
                self.solve()
            otf = self.transp_field
            if (self.l1==1 and self.l2 == 1):
                L = int(np.sqrt(self.n))
                mux = np.reshape(otf.T[0],(L,L),order='F');
                muy = np.reshape(otf.T[1],(L,L),order='F');
                return [mux,muy]
            else:
                mux = np.reshape(otf.T[0],(self.l1,self.l2),order='F');
                muy = np.reshape(otf.T[1],(self.l1,self.l2),order='F');
                return [mux,muy]

        else:
            raise ValueError('This has been implemented only for the Grid-Poisson problem with PBC...')

    #def div_minus_mu(self):
    #    '''Computes the lattice divergence (minus) of the transport field'''
    #    if (self.kind == 'gp' and self.model=='toro'):
    #        field = self.get_otf_components()
            #return div(field)
    #        return Div(field)
    #    else:
    #        raise ValueError('This has been implemented only for the Grid-Poisson problem with PBC...')

    #def curl_plus_mu(self):
    #    '''Computes the lattice curl (plus) of the transport field'''
    #    if (self.kind == 'gp' and self.model=='toro'):
    #        field = self.get_otf_components()
    #        #return curl(field)
    #        return Rot(field)
    #    else:
    #        raise ValueError('This has been implemented only for the Grid-Poisson problem with PBC...')


    def div(self):
        if (self.kind == 'gp' and (self.model=='toro' or self.model=='cilindro')):
            field = self.get_otf_components()
            return Div(field[0],field[1])
        else:
            raise ValueError('This has been implemented only for the Grid-Poisson problem with PBC...')

    def curl(self):
        if (self.kind == 'gp' and (self.model=='toro' or self.model=='cilindro')):
            field = self.get_otf_components()
            return Rot(field[0],field[1])
        else:
            raise ValueError('This has been implemented only for the Grid-Poisson problem with PBC...')





    def phi(self):
        phi = np.real(np.fft.ifft2(np.fft.fft2(self.divmu(),norm='ortho')/hatpmod(self.n),norm='ortho'))
        return phi
    def psi(self):
        psi = np.real(np.fft.ifft2(np.fft.fft2(self.curlmu(),norm='ortho')/hatpmod(self.n),norm='ortho'))
        return psi
    ## Energies of some other configuration
    def row_col_minima(self,retall=False):
        '''
        Recursive pruning of the cost matrix (row-columns minima)
        '''
        m = self.cmat
        row_col_mins = []

        for k in range(self.n):
            val = m.min()
            row_col_mins.append(val)
            i,j = np.unravel_index(m.argmin(), m.shape)
            m2 = np.delete(m,i,0)
            m = np.delete(m2,j,1)
        if (retall):
            return np.array(row_col_mins)
        else:
            return (np.array(row_col_mins)).sum() # i.e. the row column minima energy


    def row_minima(self,retall=False):
        '''
        Row minima of the cost matrix
        (Grid to Poisson < Poisson to Grid, and hence lower bound)
        retall: bool
            True: return whole list of minima
            False: return sum (i.e. the energy)
        '''
        if (retall):
            return self.cmat.min(axis=1)
        else:
            return (self.cmat.min(axis=1)).sum() # I.e. the row minima energy

    def H_minimax(self,retall=False):
        '''
        Minimax configuration
        '''

        mtilde = self.cmat

        minmaxmins = []

        for k in range(self.n):

            val = np.max(mtilde.min(axis=1))

            minmaxmins.append(val)

            posss = np.where(mtilde==val)
            i,j = posss[0].flatten(),posss[1].flatten()

            m2 = np.delete(mtilde,i,0)
            mtilde = np.delete(m2,j,1)

        if (retall):
            return minmaxmins
        else:
            return (np.array(minmaxmins)).sum() # I.e. the minimax energy




    ### ANALYSES

    def f2pr(self):
        '''Computes the two point correlation function, directly.'''
        if(not self.kind=='gp'):
            raise ValueError("This makes sense for Grid Poisson only!! (as of today...)")
            return np.nan
        else:
            tf = self.transp_field
            if (self.model=='toro'):
                dd = dist_toro(self.l1,self.l2,self.blues,self.blues)
            elif (self.model=='quadrato'):
                dd = distance.cdist(self.l1,self.l2,self.blues,self.blues)

            muxmuy = np.dot(tf,tf.T)

            last_digit = -int(math.log10(1/(3.*self.n)))+2 # Due blu non possono essere più vicini di 1/(3n), ma ci vado comunque largo

            d_F = np.round(dd.flatten(),last_digit) # Di fatto il binning avviene a questo livello !!
            #d_F = dd.flatten()
            mxy_f = muxmuy.flatten()
            cdmxy = np.array((d_F, mxy_f)).T
            tpf = pd.DataFrame(cdmxy).groupby(0, as_index=False)[1].mean().values # Statistical and average inside bins
            return tpf

    ### VISUALIZATION UTILITIES

    def plot_sol(self,plot_title=True):
        '''Plots the optimal solution on the torus (arrows go from blues to reds)'''
        if(self.is_solved):
            Hmin = round(self.cost_min,4)
        else:
            Hmin = np.nan
        if (self.l1==1 and self.l2 == 1):
            plt.figure(figsize=(8,8));
        else:
            plt.figure(figsize=(8*self.l1/np.sqrt(self.n),8*self.l2/np.sqrt(self.n)));
        if (plot_title):
            if (self.kind=='sierpinski' or self.kind=='kochcesaro'):
                plt.title("p="+str(self.p)+", n="+str(self.n)+r', $H_{\rm opt}$='+str(Hmin)+', '+str(self.kind)+', dH='+str(self.dH))
            else:
                plt.title("p="+str(self.p)+", n="+str(self.n)+r', $H_{\rm opt}$='+str(Hmin)+', '+str(self.model))
        plt.grid(linestyle='--');
        if (self.model=='toro' or self.model == 'cilindro' or self.model=='moebius'):
            plt.xlim(-.05,self.l1+.05);plt.ylim(-.05,self.l2+.05); ## In order to see links that go the other way round
        elif (self.model=='quadrato'):
            if (self.kind=='sierpinski'):
                plt.xlim(0,1);plt.ylim(-1,1);
            if (self.kind=='kochcesaro'):
                plt.xlim(0,2);plt.ylim(-1,1);
            else:
                plt.xlim(0,self.l1);plt.ylim(0,self.l2);
        plt.margins(y=0)
        plt.scatter(self.blues.T[0],self.blues.T[1],c="blue",s=5);plt.scatter(self.reds.T[0],self.reds.T[1],c="red",s=5);
        if(self.is_solved):
            for j,i in enumerate(self.opt_perm):
                coda_x,coda_y = self.blues[j][0],self.blues[j][1]
                if (self.model=='toro'):
                    agg_x = partial(aggiusta_v2, l=self.l1)
                    agg_y = partial(aggiusta_v2, l=self.l2)
                    dX = agg_x(self.reds[i][0]-self.blues[j][0])
                    dY = agg_y(self.reds[i][1]-self.blues[j][1])
                elif (self.model=='cilindro' or self.model=='moebius'):
                    agg_x = partial(aggiusta_v2, l=self.l1)
                    dX = agg_x(self.reds[i][0]-self.blues[j][0])
                    dY = self.reds[i][1]-self.blues[j][1]
                elif (self.model=='quadrato'):
                    dX = self.reds[i][0]-self.blues[j][0]
                    dY = self.reds[i][1]-self.blues[j][1]
                plt.arrow(coda_x,coda_y,dX,dY,lw=.5,head_width=0.005, head_length=None,length_includes_head=True)#,alpha=.4)


    def plot_muquadro(self):
        '''Plots \mu^2 (i.e. the local energy density)'''
        if (not self.is_solved):
            self.solve()
        else:
            muquadro_1d = (self.transp_field**2).sum(axis=1);
            muquadro_2d = np.flipud(np.reshape(muquadro_1d,(int(np.sqrt(self.n)),int(np.sqrt(self.n)))));
            plt.figure(figsize=(11,8));
            plt.title(r"$\mu^2(x)$, p="+str(self.p)+", n="+str(self.n)+r', $H_{\rm opt}$='+str(round(self.cost_min,4)))
            sns.heatmap(muquadro_2d,vmin=0);


    def plot_angles(self):
        '''Plots angles with respect to (arbitrary) x axis'''
        if (not self.is_solved):
            self.solve()
        else:
            angles_1d = self.angles1d
            angles_2d = np.flipud(np.reshape(angles_1d,(int(np.sqrt(self.n)),int(np.sqrt(self.n)))));
            plt.figure(figsize=(11,8));
            plt.title("Local angles, p="+str(self.p)+", n="+str(self.n)+r', $H_{\rm opt}$='+str(round(self.cost_min,4)))
            sns.heatmap(angles_2d,cmap='hsv',vmin=-np.pi,vmax=np.pi);

    def plot_ft(self):
        '''Displays absolute value of DFT of components in logarithmic scale'''
        if (not self.is_solved):
            self.solve()
        else:
            dFtx,dFty = self.dft_2d(components=True);
            plt.figure(figsize=(17,6));plt.grid();
            plt.subplot(1,2,1);
            M1 = np.log(np.abs(np.fft.fftshift(dFtx)));
            plt.title(r'$\log{<|(\mathcal{F}[\mu_1])(z)|>}$');
            sns.heatmap(M1,cmap='viridis');
            plt.subplot(1,2,2);
            M2 = np.log(np.abs(np.fft.fftshift(dFty)));
            plt.title(r'$\log{<|(\mathcal{F}[\mu_2])(z)|>}$');
            sns.heatmap(M2,cmap='gnuplot');
            #plt.show();

    def plot_cavity_fields(self):
        '''Displays the cavity fields associated to the optimal assignment'''
        if (not self.is_solved):
            self.solve()
        else:
            if (self.model=='toro'):
                plt.figure(figsize=(17,6));plt.grid();
                u = np.flipud(np.reshape(self.a,(int(np.sqrt(self.n)),int(np.sqrt(self.n)))));
                v = np.flipud(np.reshape(self.b,(int(np.sqrt(self.n)),int(np.sqrt(self.n)))));
                plt.subplot(1,2,1);
                plt.title('u cavity field (blues)');
                sns.heatmap(u,cmap='Blues');
                plt.subplot(1,2,2);
                plt.title('v cavity field (reds)');
                sns.heatmap(v,cmap='Reds');

            else:
                 raise ValueError("To be implemented ...")
