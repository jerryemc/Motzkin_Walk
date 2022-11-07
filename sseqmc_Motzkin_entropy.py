# pylint: disable=C0103,R0914
# main function of sse qmc

import numpy as np
from random import random as ran
from update_algorithm import diagonalupdate,crt_vtxlist,loopupdate
from utility import make_op_index, op_update_table, adjustcutoff, trans_02a, trans_a20, generate_spin_a_path_a, generate_spin_a_path_0, update_opstring
from types import SimpleNamespace

def simulate_0(self):
    '''
    SSE QMC fucntion for calcualting prob empty to A
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    -------
    Returns
    -------
    prob_02A : float
        probability for A to empty
    '''
    nb = self.l
    isteps = self.isteps
    nbins = self.nbins
    # creat operator index table
    op_index = make_op_index()
    # creat operator update table
    op_upd_tb = op_update_table()
    # number of H-operators in string
    numh = 0
    # maximum string length (self-determined)
    maxsl=1000
    # sub_a=self.suba
    alpha=self.renyi_alpha
    opstring = -(np.ones([maxsl,2*alpha])).astype(np.int64)
    vertexlist = (np.zeros(4*maxsl)).astype(np.int64)

    # spin state initialize
    spin = (np.zeros([nb,alpha])).astype(np.int64)
    nb_1=nb//6
    # alpha_2=alpha%6
    for j in range(alpha):
        for i in range(nb_1):
            if ran()<0.5:
                spin[6*i,j]=1
            else:
                spin[6*i+1,j]=1
            if ran()<0.5:
                spin[6*i+2,j]=-1
            else:
                spin[6*i+3,j]=-1
            if ran()<0.5:
                spin[6*i+4,j]=1
                spin[6*i+5,j]=-1
    # spin[nb_1:nb,:]=0
    
    # loop_not_closed=0
    # thermolize
    for i in range(isteps):
        print('th_0_diagupdate', i)
        numh=diagonalupdate(self,op_index,numh,maxsl,spin,opstring,0)
        vertexlist=crt_vtxlist(self, maxsl, opstring, 0)
        print('th_0_loopupdate', i)
        loop_is_closed=loopupdate(self, maxsl,opstring,vertexlist,op_index,op_upd_tb)
        
        mmnew=adjustcutoff(numh,maxsl)
        # print(mmnew)
        if (mmnew-maxsl) > 0:
            print('new maxsl', mmnew-maxsl,i)
            opstring_temp=opstring
            opstring=-(np.ones([mmnew,2*alpha])).astype(np.int64)
            opstring[0:maxsl,:]=opstring_temp[:]
            vertexlist=(np.zeros(4*mmnew)).astype(np.int64)
        maxsl=mmnew
        if loop_is_closed==0:
            opstring=update_opstring(self, maxsl, spin, opstring)
            continue

    # Do nbins bins with msteps MC sweeps in each, measure after each
    # n0=0.0
    na=0.0
    prob_02a=0.0
    # prob_0=0.5
    # sub_a_tmp=0
    trash_step=0
    for j in range(nbins):
        print('nbin_0_diagupdate', j)
        numh=diagonalupdate(self,op_index,numh,maxsl,spin,opstring,0)
        vertexlist=crt_vtxlist(self, maxsl, opstring, 0)
        print('nbin_0_loopupdate', j)
        loop_is_closed=loopupdate(self,maxsl,opstring,vertexlist,op_index,op_upd_tb)
        print('loop_is_closed=',loop_is_closed)
        if loop_is_closed==0:
            trash_step+=1
            opstring=update_opstring(self, maxsl, spin, opstring)
            continue
        spin_path=generate_spin_a_path_0(self, maxsl, spin, opstring, op_index)
        na+=trans_02a(self,spin_path,maxsl)
        # print('na=',na)
    print('na=',na)
    print('nbins-trash_step=', nbins-trash_step)
    prob_02a=na/((nbins-trash_step)*((maxsl+1)**(alpha-1)))
    return prob_02a


def simulate_A(self):
    '''
    SSE QMC fucntion for calcualting prob A to empty
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    Returns
    -------
    prob_a2o : float
        probability for A to empty
    '''
    nb = self.l
    isteps = self.isteps
    nbins = self.nbins
    # creat operator index table
    op_index = make_op_index()
    # creat operator update table
    op_upd_tb = op_update_table()
    # number of H-operators in string
    numh = 0
    # maximum string length (self-determined)
    maxsl=1000
    sub_a=self.suba
    alpha=self.renyi_alpha
    opstring = ((-1)*np.ones([maxsl,2*alpha])).astype(np.int64)
    vertexlist=(np.zeros(4*maxsl)).astype(np.int64)

    # spin state initialize
    spin = (np.zeros([nb,alpha])).astype(np.int64)
    nb_1= nb//6
    # alpha_2=alpha%6
    for j in range(alpha):
        for i in range(nb_1):
            if ran()<0.5:
                spin[6*i,j]=1
            else:
                spin[6*i+1,j]=1
            if ran()<0.5:
                spin[6*i+2,j]=-1
            else:
                spin[6*i+3,j]=-1
            if ran()<0.5:
                spin[6*i+4,j]=1
                spin[6*i+5,j]=-1
    # spin[nb_1:nb,:]=0
    # thermolize
    for i in range(isteps):
        print('th_a_diagupdate', i)
        numh=diagonalupdate(self,op_index,numh,maxsl,spin,opstring,sub_a)
        vertexlist=crt_vtxlist(self, maxsl, opstring, sub_a)
        print('th_a_loopupdate', i)
        loop_is_closed=loopupdate(self,maxsl,opstring,vertexlist,op_index,op_upd_tb)
        mmnew=adjustcutoff(numh,maxsl)
        # print(mmnew)
        if (mmnew-maxsl) > 0:
            print('new maxsl', mmnew-maxsl,i)
            opstring_temp=opstring
            opstring=-(np.ones([mmnew,2*alpha])).astype(np.int64)
            opstring[0:maxsl,:]=opstring_temp[:]
            vertexlist=(np.zeros(4*mmnew)).astype(np.int64)
        maxsl=mmnew
        if loop_is_closed==0:
            opstring=update_opstring(self, maxsl, spin, opstring)
            continue
    # Do nbins bins with msteps MC sweeps in each, measure after each
    n0=0.0
    trash_step=0
    for j in range(nbins):
        print('nbin_a_diagupdate', j)
        numh=diagonalupdate(self,op_index,numh,maxsl,spin,opstring,sub_a)
        vertexlist=crt_vtxlist(self, maxsl, opstring, sub_a)
        print('nbin_a_loopupdate', j)
        loop_is_closed=loopupdate(self,maxsl,opstring,vertexlist,op_index,op_upd_tb)
        if loop_is_closed==0:
            trash_step+=1
            opstring=update_opstring(self, maxsl, spin, opstring)
            continue
        spin_path=generate_spin_a_path_a(self, maxsl, spin, opstring, op_index)
        n0+=trans_a20(self,spin_path,maxsl)
    print('n0=',n0)
    print('nbins-trash_step=',nbins-trash_step)
    prob_a20=n0/(nbins-trash_step)
    return prob_a20





sim = SimpleNamespace(isteps = 100,          # Number of MC sweeps for equilibration
                      l=30,                 # Grid size in y dimension
                      nbins=500,            # Number of bins
                      suba=2,               # size of sub a
                      beta = 1000,          # Initial inverse temperature
                      renyi_alpha = 2,      # Renyi entropy alpha
                      )

prob_0toa=simulate_0(sim)
# print(prob_0toa)
prob_ato0=simulate_A(sim)
# print(prob_ato0)
entropy=np.log(prob_0toa/prob_ato0)/(1-sim.renyi_alpha)
print(entropy)
