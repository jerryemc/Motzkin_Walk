# pylint: disable=too-many-locals,C0103
# functions for local update, vertex list and unlocal update

from random import random as ran
import numpy as np
from utility import vtx2idx, idx2vtx, vtx_updt




def updatespin(spin,loca_m,loca_l):
    '''
    Update spin state given an off-diagonal operator
    ----------
    Parameters
    ----------
    spin : 2D array (M, L)
        spin information of certain MC step 
    loca_m : int
        operator location in operator space
    loca_l : int
        operator location in chain space
    ------- 
    Returns
    -------
    None
    '''
    
    n=loca_m
    b=loca_l
    if (spin[n,b]*spin[n,b+1]==0):
        if (spin[n,b]==spin[n,b+1]):
            spin[n+1,b]=1
            spin[n+1,b+1]=-1
        else:
            # tempspin=spin[n,b+1]
            spin[n+1,b+1]=spin[n,b]
            spin[n+1,b]=spin[n,b+1]
    else:
        spin[n+1,b]=0
        spin[n+1,b+1]=0


def updatevertex(op_upd_tb,opstring,loca_m,eleg,schange):
    '''
    Directed Loop Update, given an off-diagonal operator, entrence leg and spin change
    ----------
    Parameters
    ----------
    op_upd_tb : 1D array (9*4*2)
        link operators to changed operators
    opstring : 2D array (M,2)
        operator string
    loca_m : int
        operator location in operator space
    eleg : int
        1, 2, 3, 4.
    schange : int
        1, -1.
    ------- 
    Returns
    -------
    xleg : int
        0, 1, 2, 3.
    change : int
        1, -1.
    '''
    
    n=loca_m
    avtxtp=opstring[n,1]
    aidx=vtx2idx(avtxtp,eleg,schange)
    bidx=op_upd_tb[aidx]
    bvtxtp,xleg,schange=idx2vtx(bidx)
    opstring[n,1]=bvtxtp
    return xleg,schange
    
    
    

    
    

def diagonalupdate(self,op_index,numh,maxsl,spin,opstring,sub_a):
    '''
    Carries out one sweep of diagonal updates
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    numh : int
        number of H-operators in string
    maxsl : Integer
        The cut-off maxsl of the expansion
    spin : 1D array (L)
        spin information
    opstring : 2D array (L,2)
        operator string
    a_or_0 : Integer
        a enjoined or oslash
    -------
    Returns
    -------
    numh : int
        number of H-operators in string
    '''

    nb = self.l
    beta = self.beta
    alpha = self.renyi_alpha
    # probabilities used in diagonal update
    aprob=0.5*beta*nb
    dprob=1.0/(0.5*beta*nb)
    # for j in range(alpha):
    #     for i in range(6*nb):
    #         spin_path[i,j*maxsl]=spin[i,j]
    # spin_path[]
    for j in range(alpha):
        for i in range(maxsl):
            vtxtp=opstring[i,2*j+1]
            if (vtxtp==-1):
                b=int(ran()*(nb-1))
                if (spin[b,j]*spin[b+1,j]!=0 and spin[b,j]<=spin[b+1,j]):
                    if (ran()*(maxsl-numh)<=aprob):
                        opstring[i,2*j]=b
                        if (spin[b,j]==spin[b+1,j]==-1):
                            opstring[i,2*j+1]=0
                        elif (spin[b,j]==spin[b+1,j]==1):
                            opstring[i,2*j+1]=2
                        else:
                            opstring[i,2*j+1]=1
                        numh=numh+1
            elif (vtxtp<3):
                if (ran()<=dprob*(maxsl-numh+1)):
                    opstring[i,2*j+1]=-1
                    opstring[i,2*j]=-1
                    numh=numh-1
            else:
                b=opstring[i,2*j]
                vtxtp=opstring[i,2*j+1]
                # updatespin(spin,b)
                spin[b,j]=op_index[vtxtp,0]
                spin[b+1,j]=op_index[vtxtp,1]
                # spin[bsites[b,0]-1]=-spin[bsites[b,0]-1]
                # spin[bsites[b,1]-1]=-spin[bsites[b,1]-1]
        if j<alpha-1:
            spin[0:sub_a,j+1]=spin[0:sub_a,j]
    return numh



def vtxlist_test(self,maxsl):
    alpha=self.renyi_alpha
    vtxlist=((-1)*np.ones(alpha*4*maxsl)).astype(np.int64)
    for i in range(1,alpha*4*maxsl-1):
        vtxlist[4*i]=4*(i+1)+3
        vtxlist[4*i+1]=4*(i+1)+2
        vtxlist[4*i+2]=4*(i-1)+1
        vtxlist[4*i+3]=4*(i-1)
    vtxlist[0]=7
    vtxlist[1]=6
    vtxlist[2]=4*(maxsl*alpha-1)+1
    vtxlist[3]=4*(maxsl*alpha-1)
    vtxlist[4*(maxsl*alpha-1)]=3
    vtxlist[4*(maxsl*alpha-1)+1]=2
    vtxlist[4*(maxsl*alpha-1)+2]=4*(maxsl*alpha-2)+1
    vtxlist[4*(maxsl*alpha-1)+3]=4*(maxsl*alpha-2)
    return vtxlist
    



def crt_vtxlist(self, maxsl, opstring, sub_a):
    '''
    Makes the linkes vertex list.
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    maxsl : Integer
        The cut-off maxsl of the expansion
    opstring : 1D array (L)
        operator string
    ---------
    Returns
    ---------
    vertexlist : 1D array (L)
    vertex information
    '''
    nb = self.l
    alpha=self.renyi_alpha
    # ----------------------------------------------------------------------
    # Makes the linkes vertex list. #-----------------------------------------------------------------------
    vertexlist=((-1)*np.ones(alpha*4*maxsl)).astype(np.int64)
    lk_chain=((-1)*np.ones((2,nb))).astype(np.int64)
    # start_leg=end_leg=0
    for j in range(alpha):
        base=j*maxsl
        for l in range(maxsl):
            op=opstring[l,2*j+1]
            # if there is operator in l, do vertex link
            if op>=0:
                b=opstring[l,2*j]
                if (lk_chain[0,b]==-1):
                    lk_chain[1,b]=base+4*l+3
                    lk_chain[0,b]=base+4*l
                else :
                    vertexlist[base+4*l+3]=lk_chain[0,b]
                    vertexlist[lk_chain[0,b]]=base+4*l+3
                    lk_chain[0,b]=base+4*l
                if (lk_chain[0,b+1]==-1):
                    lk_chain[1,b+1]=base+4*l+2
                    lk_chain[0,b+1]=base+4*l+1
                else :
                    vertexlist[base+4*l+2]=lk_chain[0,b+1]
                    vertexlist[lk_chain[0,b+1]]=base+4*l+2
                    lk_chain[0,b+1]=base+4*l+1
        if j==(alpha-1):
            sub_a=0
        for l in range(sub_a, nb):
            if lk_chain[0,l]!=-1:
                vertexlist[lk_chain[0,l]]=lk_chain[1,l]
                vertexlist[lk_chain[1,l]]=lk_chain[0,l]
                lk_chain[0,l]=lk_chain[1,l]=-1
    print('vertexlist = ', vertexlist)
    return vertexlist

def loopupdate(self,maxsl,opstring,vertexlist,op_index,op_upd_tb):
    '''
    Carries out loop updates.
    ----------
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    bsites :  2D array (L, 2)
        bonds
    maxsl : Integer
        The cut-off maxsl of the expansion
    spin : 1D array (L)
        spin information
    opstring : 1D array (L)
        operator string
    vertexlist : 1D array (L)
        vertex information
    ---------
    Returns
    ---------
    loop_is_closed : Integer
        If the loop travel is closed return 1; else return 0, and we need regenerate the opstring.
    '''
    
    #--------------------------------------------------------------------------
    # loop updates
    #--------------------------------------------------------------------------
    # number of loop - nl. nl~2M
    nl=0
    # length of loop - ll. ll~100M
    # ll=0
    alpha=self.renyi_alpha
    vtx_is_0=0
    loop_is_closed=0
    while nl<1:
        v0=int(ran()*4*alpha*maxsl)
        ## if no operator, randomly select next vertex leg
        # print('nl=', nl, 'vertexlist[v0] =',vertexlist[v0])
        if vertexlist[v0]<0:
            vtx_is_0+=1
            if vtx_is_0>4*alpha*(maxsl+1):
                print('no operator')
                break
            continue
        vtx_is_0=0
        # the loop start config
        v_j=v0//(4*maxsl)
        vn=(v0%(4*maxsl))//4
        vtx=opstring[vn,2*v_j+1]
        eleg=v0%4
        # print('nl =',nl)
        if op_index[vtx,eleg]<0:
            schg=1
        elif op_index[vtx,eleg]>0:
            schg=-1
        else:
            schg=2*int(2*ran())-1
        # travel the loop
        loop_is_closed=0
        v1=v0
        ll=0
        while ll<100*alpha*maxsl:
        # while 1:
            v_j=v1//(4*maxsl)
            vtx_n=(v1%(4*maxsl))//4
            eleg=v1%4
            # print('maxsl=',maxsl)
            vtx_tp=opstring[vtx_n,2*v_j+1]
            vtx_tp,xleg,schg=vtx_updt(vtx_tp,eleg,schg,op_upd_tb)
            opstring[vtx_n,2*v_j+1]=vtx_tp
            v1=v_j*(4*maxsl)+4*vtx_n+xleg
            if v1==v0:
                loop_is_closed=1
                print('ll=',ll)
                break
            v1=vertexlist[v1]
            if v1==v0:
                print('ll=',ll)
                loop_is_closed=1
                break
            ll+=1
        # print('loop_is_closed',loop_is_closed,'v1=',v1)
        if loop_is_closed==0:
            print('!!!_unclosed loop, break_!!!')
            break
        nl+=1
    return loop_is_closed
    
    # for i in range(1,nn+1):
    #     if (frstspinop[i]!=-1):
    #         if (vertexlist[frstspinop[i]]==-2):
    #             spin[i-1]=-spin[i-1]
    #     else:
    #         if (ran()<0.5): 
    #             spin[i-1]=-spin[i-1]
    # return None
                
        

        

    