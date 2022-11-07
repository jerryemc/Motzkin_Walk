# pylint: disable=too-many-locals,C0103,R0913
import numpy as np

# import numpy.random as rnd

def make_op_index():
    '''Create a table of vertex operator index and vertex leg state.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    op_index : 2D array (9, 4)
        containing vertex state
    '''
    op_index = (np.zeros([9, 4])).astype(np.int64)
    # diagonal operator
    op_index[0,0]=-1
    op_index[0,1]=-1
    op_index[0,2]=-1
    op_index[0,3]=-1
    op_index[1,0]=-1
    op_index[1,1]=1
    op_index[1,2]=1
    op_index[1,3]=-1
    op_index[2,0]=1
    op_index[2,1]=1
    op_index[2,2]=1
    op_index[2,3]=1
    # off-diagonal operator
    op_index[3,0]=-1
    op_index[3,1]=0
    op_index[3,2]=-1
    op_index[3,3]=0
    op_index[4,0]=0
    op_index[4,1]=-1
    op_index[4,2]=0
    op_index[4,3]=-1
    op_index[5,0]=1
    op_index[5,1]=0
    op_index[5,2]=1
    op_index[5,3]=0
    op_index[6,0]=0
    op_index[6,1]=1
    op_index[6,2]=0
    op_index[6,3]=1
    op_index[7,0]=1
    op_index[7,1]=-1
    op_index[7,2]=0
    op_index[7,3]=0
    op_index[8,0]=0
    op_index[8,1]=0
    op_index[8,2]=-1
    op_index[8,3]=1
    
    return op_index

def op_update_table():
    '''Create a table of vertex operator update table.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    op_upd_tb : 1D array (9*4*2)
        containing vertex state
    '''
    
    x=-1
    op_upd_tb=[ 36, x, 30, x,  32, x,  26, x,  9, x, x, 10, x, 12, 15, x, x, 53, x,  47, x,  49, x,  43, 70,  x,  27, 7,  29, x,  64, 3, 62,  5,  35, x,  37, 1,  56, x, x, 40, 22, 61, x, 59, 18, 46, 20, 48, x, 69, 16, 67, x, 54, x,   39, 44, x,  42, 60, 63, 33, 65,  31, 52, 66, 50, x,  x,  25]
    return (np.array(op_upd_tb)).astype(np.int64)
    
def vtx2idx(vertex, eleg, schange):
    return vertex*8+eleg*2+(1-schange)//2

def idx2vtx(index):
    vertex = index//8
    xleg = (index%8)//2
    schange = 1-2*((index%8)%2)
    return vertex, xleg, schange

def vtxl2idx(vertex_leg,schange):
    return 2*vertex_leg+(1-schange)//2

def idx2vtxl(index):
    vertex_leg=index//2
    schange=1-2*(index%2)
    return vertex_leg, schange

def vtx_updt(vtx_tp,eleg,schange,op_upd_tb):
    idx=vtx2idx(vtx_tp,eleg,schange)
    idx=op_upd_tb[idx]
    vtx_tp,xleg,schange=idx2vtx(idx)
    return vtx_tp,xleg,schange


def update_opstring(self, maxsl, spin, opstring):
    '''
    When loop update create no-closed loop, update the operator string.
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    maxsl : Integer
        The cut-off maxsl of the expansion
    spin : 1D array (L)
        spin information
    opstring : 2D array (L,2M)
        operator string
    op_index : 2D array (9, 4)
        containing vertex state
    suba : Integer
        size of subsystem A; 0 if empty A.
    -------
    Returns
    -------
    opstring : 2D array (L,2)
        operator string
    '''
    alpha = self.renyi_alpha
    nb = self.l
    spin_temp=(np.zeros(nb)).astype(np.int64)
    for i in range(alpha):
        spin_temp[:]=spin[:,i]
        for j in range(maxsl):
            if opstring[j,2*i+1]>2:
                b=opstring[j,2*i]
                if (spin_temp[b]*spin_temp[b+1]==0):
                    if (spin_temp[b]==spin_temp[b+1]):
                        spin_temp[b]=1
                        spin_temp[b+1]=-1
                        opstring[j,2*i+1]=7
                    else:
                        x=spin_temp[b+1]
                        y=spin_temp[b]
                        spin_temp[b]=x
                        spin_temp[b+1]=y
                        opstring[j,2*i+1]=(9+(x+y)*(2+y-x))//2
                else:
                    spin_temp[b]=0
                    spin_temp[b+1]=0
                    opstring[j,2*i+1]=8
    return opstring
                    






def generate_spin_a_path_0(self, maxsl, spin, opstring, op_index):
    '''
    Generate the spin path in subsystem A when oslash.
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    maxsl : Integer
        The cut-off maxsl of the expansion
    spin : 1D array (L)
        spin information
    opstring : 2D array (L,2)
        operator string
    op_index : 2D array (9, 4)
        containing vertex state
    -------
    Returns
    -------
    spin_path : 2D array (sub_a, alpha*(maxsl+1))
        spin a path
    '''
    alpha=self.renyi_alpha
    # nb=self.l
    sub_a=self.suba
    spin_path=(np.zeros([sub_a,alpha*(maxsl+1)])).astype(np.int64)
    for i in range(alpha):
        spin_path[:,i*(maxsl+1)]=spin[0:sub_a,i]
        for j in range(maxsl):
            spin_path[:,i*(maxsl+1)+j+1]=spin_path[:,i*(maxsl+1)+j]
            vtxtp=opstring[j,2*i+1]
            b=opstring[j,2*i]
            if vtxtp>2 and b<sub_a:
                # b=opstring[j,2*i]
                spin_path[b,i*(maxsl+1)+j+1]=op_index[vtxtp,0]
                if b+1<sub_a:
                    spin_path[b+1,i*(maxsl+1)+j+1]=op_index[vtxtp,1]
    return spin_path



def generate_spin_a_path_a(self, maxsl, spin, opstring, op_index):
    '''
    Generate the spin path in subsystem A when attach.
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    maxsl : Integer
        The cut-off maxsl of the expansion
    spin : 1D array (L)
        spin information
    opstring : 2D array (L,2)
        operator string
    op_index : 2D array (9, 4)
        containing vertex state
    -------
    Returns
    -------
    spin_path : 2D array (sub_a, alpha*(maxsl+1))
        spin a path
    '''
    alpha=self.renyi_alpha
    # nb=self.l
    sub_a=self.suba
    spin_path=(np.zeros([sub_a,alpha*(maxsl+1)])).astype(np.int64)
    spin_path[:,0]=spin[0:sub_a,0]
    for i in range(alpha):
        if i>0:
            spin_path[:,i*(maxsl+1)]=spin_path[:,i*(maxsl+1)-1]
        for j in range(maxsl):
            spin_path[:,i*(maxsl+1)+j+1]=spin_path[:,i*(maxsl+1)+j]
            vtxtp=opstring[j,2*i+1]
            b=opstring[j,2*i]
            if vtxtp>2 and b<sub_a:
                # b=opstring[j,2*i]
                spin_path[b,i*(maxsl+1)+j+1]=op_index[vtxtp,0]
                if b+1<sub_a:
                    spin_path[b+1,i*(maxsl+1)+j+1]=op_index[vtxtp,1]
    return spin_path

# def makelattice(self):
#     '''Constructs the lattice, in the form of the list of sites connected by bonds 'bsites'.
    
#     Parameters
#     ----------
#     self : NameSpace
#         contains all the simulation parameters
#     Returns
#     -------
#     bsites : 2D array (nb, 2)
#         containing al the bonds
#     '''
    
#     l_chain = self.l
#     numb = 2 * l_chain
#     bsites = (np.zeros([l_chain, 2])).astype(np.int64)
#     for xi in range(l_chain):
#             bsites[xi-1,0] = xi
#             bsites[s-1,1] = 1+x2+y2*lx
#             x2=x1
#             y2=(y1+1)%ly
#             bsites[s+nn-1,0]=s
#             bsites[s+nn-1,1]=1+x2+y2*lx
#     return bsites.astype(np.int64)

def adjustcutoff(nh,mm):
    '''Carries out loop updates; each loop is constructed and flipped with probability 1/2. At the end, spins not connected to any operator (corresponding to purely time-like loops) are flipped with probability 1/2
    
    Parameters
    ----------
    nh : Integer
        Number of H-operators in string
    mm : Integer
        Maximum string length (self-determined)
    
        
    Returns
    ---------
    mm : Integer
        Maximum string length (self-determined)
        
    '''
    mmnew=nh+nh//3
    if (mmnew<=mm):
        pass
    else:
        mm=mmnew
    return mm

def trans_02a(self,spin_path,maxsl):
    '''Calculation the number of situation that empty set to A
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    spin_path :  2D array (L, 2)
        bonds
    maxsl : Integer
        The cut-off maxsl of the expansion
    sub_a : Integer
        number of H-operators in string
    Returns
    -------
    num_equal : Integer
        number of 0 to A
    '''
    alpha=self.renyi_alpha
    num_equal=0.0
    for i in range(maxsl+1):
        for j in range(maxsl+1):
            for k in range(alpha-1):
                if (spin_path[:,k*(maxsl+1)+i]==spin_path[:,(k+1)*(maxsl+1)+j]).all():
                    # print('num_equal=',num_equal)
                    num_equal+=1
    return num_equal

def trans_a20(self,spin_path,maxsl):
    '''Computes expectation values and accumulates them:
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    spin_path :  2D array (L, 2)
        bonds
    maxsl : Integer
        The cut-off maxsl of the expansion
    sub_a : Integer
        number of H-operators in string
    Returns
    -------
    num_equal : Integer
        Whether A to o is possible
    '''
    alpha=self.renyi_alpha
    num_equal=0.0
    re_a2o=0.0
    for i in range(alpha):
        if (spin_path[:,i*(maxsl+1)]==spin_path[:,(i+1)*(maxsl+1)-1]).all():
            num_equal+=1
    if num_equal==alpha:
        re_a2o=1.0
    else:
        re_a2o=0.0
    return re_a2o

# def stepmeasure(self,bsites,nh,mm,spin,opstring,smresult):
#     '''Computes expectation values and accumulates them:
#     enrg1 : Energy
#     enrg2 : Squared energy for specific heat calculation
#     amag2 : Staggered magnetization
#     ususc : Uniform susceptibility
    
#     Parameters
#     ----------
#     self : NameSpace
#         contains all the simulation parameters
#     bsites :  2D array (L, 2)
#         bonds
#     nh : Integer
#         number of H-operators in string
#     mm : Integer
#         The cut-off mm of the expansion
#     spin : 1D array (L)
#         spin information
#     opstring : 1D array (L)
#         operator string
#     smresult : 2D array (1,3)
    
#     Returns
#     -------
#     smresult : 1D array (L)
#         step result
#     '''
#     lx = self.lx
#     ly = self.ly
#     # nn is the number of sites
#     nn = lx * ly
#     i=j=b=s1=s2=am=op=0
#     for i in range(nn):
#         am=am+spin[i]*(-1)**((i%lx)+(i)//lx)
#     am=am//2
#     am2=0.0
#     for j in range(mm):
#         op=int(opstring[j])
#         if (op%2)==1:
#             b=op//2-1
#             s1=bsites[b,0]-1
#             s2=bsites[b,1]-1
#             spin[s1]=-spin[s1]
#             spin[s2]=-spin[s2]
#             am=am+2*spin[s1]*(-1)**(s1%lx+s1//lx)
#         am2=am2+am**2
#     am2=am2/mm
    
#     enrg1=smresult[0]+nh
#     enrg2=smresult[1]+nh**2
#     amag2=smresult[2]+am2
#     ususc=smresult[3]+(np.sum(spin)/2)**2
    
#     return [enrg1, enrg2, amag2, ususc]

# def binmeasure(self, msteps, smresult):
#     lx = self.lx
#     ly = self.ly
#     beta = self.beta
    
#     nn = lx*ly
#     nb = 2*nn

#     enrg1=smresult[0]/msteps
#     enrg2=smresult[1]/msteps
#     amag2=smresult[2]/msteps
#     ususc=smresult[3]/msteps

#     enrg2=(enrg2-enrg1*(enrg1+1.0))/nn
#     enrg1=-(enrg1/(beta*nn)-0.25*nb/nn)
#     amag2=3.0*amag2/nn**2
#     ususc=beta*ususc/nn
    
#     return [enrg1, enrg2, amag2, ususc]