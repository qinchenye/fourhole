import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util
import lanczos

def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    
    if orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        elif z2==z1 and orb1=='dx2y2' and orb2=='d3z2r2':
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];    
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
    tlabel = slabel[10:15] + [s4,orb4,x4,y4,z4]
    tmp34 = reorder_z(tlabel)
    
    if tmp34 == tlabel:
        slabel2 = slabel + [s4,orb4,x4,y4,z4]
    elif  tmp34 != tlabel:
        tlabel = slabel[5:10] + [s4,orb4,x4,y4,z4]
        tmp24 = reorder_z(tlabel)
        if tmp24 == tlabel:
            slabel2 = slabel[0:10]+ [s4,orb4,x4,y4,z4] + slabel[10:15]
        elif  tmp24 != tlabel:
            tlabel = slabel[0:5] + [s4,orb4,x4,y4,z4]   
            tmp14 = reorder_z(tlabel)
            if tmp14 == tlabel:
                slabel2 = slabel[0:5]+ [s4,orb4,x4,y4,z4] + slabel[5:15]
            elif  tmp24 != tlabel:
                slabel2 = [s4,orb4,x4,y4,z4] + slabel[0:15]     
                
    return slabel2


def get_ground_state(matrix, VS, S_Nid8_val, Sz_Nid8_val, S_Cud8_val, Sz_Cud8_val,\
                     S_other_Ni_val, Sz_other_Ni_val, S_other_Cu_val, Sz_other_Cu_val, tz):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     print ('H=')
#     print (M_dense)
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()                                                               #calculate atom limit
#     print ('lowest eigenvalue of H from np.linalg.eigh = ')
#     print (vals)
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    #get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,1):                                                                          #gai
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.01)

        wgt_LmLn = np.zeros(10)
        wgt_d9Ld10L2 = np.zeros(10)
        wgt_d9d10L3 = np.zeros(10)        
        wgt_d9L2d10L= np.zeros(10)
        wgt_d10Ld9L2= np.zeros(10)
        wgt_d10d9L3= np.zeros(10)   
        wgt_d10L2d9L= np.zeros(10)  
        wgt_d8Ld10L= np.zeros(10)
        wgt_d10Ld8L= np.zeros(10)        
        wgt_d8d10L2= np.zeros(10)
        wgt_d10d8L2 = np.zeros(10)
        wgt_d8L2d10 = np.zeros(10)        
        wgt_d10L2d8 = np.zeros(10) 
        wgt_d9L2d9 = np.zeros(10)
        wgt_d9d9L2 = np.zeros(10)    
        wgt_d9Ld9L = np.zeros(18)         
        wgt_d9d8L = np.zeros(10)     
        wgt_d8d9L = np.zeros(10) 
        wgt_d9Ld8 = np.zeros(10)  
        wgt_d8Ld9 = np.zeros(10)
        wgt_d8d8 = np.zeros(10) 
  
        #Sumweight refers to the general weight.Sumweight1 refers to the weight in indices.Sumweight_picture refers to the weight that is calculated.Sumweight2 refers to the weight that differs by orbits

        sumweight=0
        sumweight1=0
        synweight2=0

        print ("Compute the weights in GS (lowest Aw peak)")
        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            weight = abs(vecs[i,k])**2
            
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            s4 = state['hole4_spin']            
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb']            
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']            

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S_Nid8_12  = S_Nid8_val[i]
            Sz_Nid8_12 = Sz_Nid8_val[i]
            S_Cud8_12  = S_Cud8_val[i]
            Sz_Cud8_12 = Sz_Cud8_val[i]
            
            S_Niother_12  = S_other_Ni_val[i]
            Sz_Niother_12 = Sz_other_Ni_val[i]
            S_Cuother_12  = S_other_Cu_val[i]
            Sz_Cuother_12 = Sz_other_Cu_val[i]
                
            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
            s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];            
            if i in indices[0]: 
                sumweight1=sumweight1+abs(vecs[i,k])**2
#                 print (' state ', i, orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3,orb4,s4,x4,y4,z4,\
#                        'S_Nid8=', S_Nid8_12, 'Sz_Nid8=', Sz_Nid8_12, 'S_Cud8=', S_Cud8_12, 'Sz_Cud8=', Sz_Cud8_12, \
#                        'S_Ni_other=', S_Niother_12, 'Sz_Ni_other=', Sz_Niother_12, \
#                        'S_Cu_other=', S_Cuother_12, 'Sz_Cu_other=', Sz_Cuother_12, \
#                        ", weight = ", weight)   
                
                
            if (orb1 in pam.O_orbs) and  (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)  and  (orb4 in pam.O_orbs):
                wgt_LmLn[0]+=abs(vecs[i,k])**2 

                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)\
                      and (orb4 in pam.O_orbs)  and z1==z2==1 and z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d9Ld10L2[0]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2': 
                    wgt_d9Ld10L2[1]+=abs(vecs[i,k])**2   
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)\
                      and (orb4 in pam.O_orbs) and z1==1 and z2==z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d9d10L3[0]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d9d10L3[1]+=abs(vecs[i,k])**2          
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==z3==1 and z4==0:
                if orb1=='dx2y2':
                    wgt_d9L2d10L[0]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d9L2d10L[1]+=abs(vecs[i,k])**2                       
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z2==1 and z1==z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d10Ld9L2[0]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d10Ld9L2[1]+=abs(vecs[i,k])**2   
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d10d9L3[0]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d10d9L3[1]+=abs(vecs[i,k])**2          
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z2==z3==1 and z1==z4==0:
                if orb1=='dx2y2':
                    wgt_d10L2d9L[0]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d10L2d9L[1]+=abs(vecs[i,k])**2           
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==1 and z3==1 and z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Nid8_12==0:
                    wgt_d8Ld10L[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Nid8_12==1:
                    wgt_d8Ld10L[1]+=abs(vecs[i,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8Ld10L[2]+=abs(vecs[i,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Nid8_12==0:
                    wgt_d8Ld10L[3]+=abs(vecs[i,k])**2  
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==0 and z3==1 and z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Cud8_12==0:
                    wgt_d10Ld8L[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Cud8_12==1:
                    wgt_d10Ld8L[1]+=abs(vecs[i,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Cud8_12==0:
                    wgt_d10Ld8L[2]+=abs(vecs[i,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Cud8_12==0:
                    wgt_d10Ld8L[3]+=abs(vecs[i,k])**2                      
                                          
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==1 and z3==z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d10L2[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Nid8_12==1:
                    wgt_d8d10L2[1]+=abs(vecs[i,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8d10L2[2]+=abs(vecs[i,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d10L2[3]+=abs(vecs[i,k])**2                     
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==0 and z3==z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Cud8_12==0:
                    wgt_d10d8L2[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Cud8_12==1:
                    wgt_d10d8L2[1]+=abs(vecs[i,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Cud8_12==0:
                    wgt_d10d8L2[2]+=abs(vecs[i,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Cud8_12==0:
                    wgt_d10d8L2[3]+=abs(vecs[i,k])**2                        
                                     
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==1 and z3==z4==1 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Nid8_12==0:
                    wgt_d8L2d10[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Nid8_12==1:
                    wgt_d8L2d10[1]+=abs(vecs[i,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8L2d10[2]+=abs(vecs[i,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Nid8_12==0:
                    wgt_d8L2d10[3]+=abs(vecs[i,k])**2                        
               
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==0 and z3==z4==1 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Cud8_12==0:
                    wgt_d10L2d8[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Cud8_12==1:
                    wgt_d10L2d8[1]+=abs(vecs[i,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Cud8_12==0:
                    wgt_d10L2d8[2]+=abs(vecs[i,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Cud8_12==0:
                    wgt_d10L2d8[3]+=abs(vecs[i,k])**2                         

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==1 and z2==0 and z3==z4==1 :                               
                if orb1=='d3z2r2' and orb2=='dx2y2':
                    wgt_d9L2d9[0]+=abs(vecs[i,k])**2    
                elif orb1=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9L2d9[1]+=abs(vecs[i,k])**2            
                elif orb1=='d3z2r2' and orb2=='d3z2r2':
                    wgt_d9L2d9[2]+=abs(vecs[i,k])**2    
                elif orb1=='dx2y2' and orb2=='dx2y2':
                    wgt_d9L2d9[3]+=abs(vecs[i,k])**2   
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==1 and z2==0 and z3==z4==0 :                               
                if orb1=='d3z2r2' and orb2=='dx2y2':
                    wgt_d9d9L2[0]+=abs(vecs[i,k])**2    
                elif orb1=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9d9L2[1]+=abs(vecs[i,k])**2            
                elif orb1=='d3z2r2' and orb2=='d3z2r2':
                    wgt_d9d9L2[2]+=abs(vecs[i,k])**2    
                elif orb1=='dx2y2' and orb2=='dx2y2':
                    wgt_d9d9L2[3]+=abs(vecs[i,k])**2                       
                    
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==1 and z2==0 and z3==1 and z4==0 :                  
                if orb1=='d3z2r2' and orb2=='dx2y2' and s1==s2 and S_Niother_12==0:
                    wgt_d9Ld9L[0]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='d3z2r2' and s1==s2 and S_Niother_12==0:
                    wgt_d9Ld9L[1]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and s1==s2 and S_Niother_12==0:
                    wgt_d9Ld9L[2]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='dx2y2'and s1==s2 and S_Niother_12==0:
                    wgt_d9Ld9L[3]+=abs(vecs[i,k])**2  
                elif orb1=='d3z2r2' and orb2=='dx2y2' and s1!=s2 and S_Niother_12==0:
                    wgt_d9Ld9L[4]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='d3z2r2' and s1!=s2 and S_Niother_12==0:
                    wgt_d9Ld9L[5]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and s1!=s2 and S_Niother_12==0:
                    wgt_d9Ld9L[6]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='dx2y2' and s1!=s2 and S_Niother_12==0:
                    wgt_d9Ld9L[7]+=abs(vecs[i,k])**2      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and s1!=s2 and S_Niother_12==1:
                    wgt_d9Ld9L[8]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='d3z2r2' and s1!=s2 and S_Niother_12==1:
                    wgt_d9Ld9L[9]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and s1!=s2 and S_Niother_12==1:
                    wgt_d9Ld9L[10]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='dx2y2' and s1!=s2 and S_Niother_12==1:
                    wgt_d9Ld9L[11]+=abs(vecs[i,k])**2                   
                if orb1=='d3z2r2' and orb2=='dx2y2' and s1==s2 and S_Niother_12==1:
                    wgt_d9Ld9L[12]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='d3z2r2' and s1==s2 and S_Niother_12==1:
                    wgt_d9Ld9L[13]+=abs(vecs[i,k])**2   
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and s1==s2 and S_Niother_12==1:
                    wgt_d9Ld9L[14]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='dx2y2'and s1==s2 and S_Niother_12==1:
                    wgt_d9Ld9L[15]+=abs(vecs[i,k])**2  
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==1 and z2==z3==z4==0:                     
                if orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='d3z2r2' and S_Cud8_12==0:
                    wgt_d9d8L[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9d8L[1]+=abs(vecs[i,k])**2                       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9d8L[2]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==1:
                    wgt_d9d8L[3]+=abs(vecs[i,k])**2       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==1:
                    wgt_d9d8L[4]+=abs(vecs[i,k])**2                     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9d8L[5]+=abs(vecs[i,k])**2 
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9d8L[6]+=abs(vecs[i,k])**2                 
                  
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==1 and z3==z4==0:                 
                if orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8d9L[0]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Nid8_12==1:
                    wgt_d8d9L[1]+=abs(vecs[i,k])**2                       
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8d9L[2]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d9L[3]+=abs(vecs[i,k])**2     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Nid8_12==1:
                    wgt_d8d9L[4]+=abs(vecs[i,k])**2       
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d9L[5]+=abs(vecs[i,k])**2       
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d9L[6]+=abs(vecs[i,k])**2                       
                 
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z4==1 and z2==z3==0:                 
                if orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='d3z2r2' and S_Cud8_12==0:
                    wgt_d9Ld8[0]+=abs(vecs[i,k])**2                      
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9Ld8[1]+=abs(vecs[i,k])**2                       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9Ld8[2]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==1:
                    wgt_d9Ld8[3]+=abs(vecs[i,k])**2       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cud8_12==1:
                    wgt_d9Ld8[4]+=abs(vecs[i,k])**2                     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9Ld8[5]+=abs(vecs[i,k])**2 
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cud8_12==0:
                    wgt_d9Ld8[6]+=abs(vecs[i,k])**2                 
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==z4==1 and z3==0:                      
                if orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8Ld9[0]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Nid8_12==1:
                    wgt_d8Ld9[1]+=abs(vecs[i,k])**2                       
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8Ld9[2]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Nid8_12==0:
                    wgt_d8Ld9[3]+=abs(vecs[i,k])**2     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Nid8_12==1:
                    wgt_d8Ld9[4]+=abs(vecs[i,k])**2       
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Nid8_12==0:
                    wgt_d8Ld9[5]+=abs(vecs[i,k])**2   
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Nid8_12==0:
                    wgt_d8Ld9[6]+=abs(vecs[i,k])**2                   
   

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.Ni_Cu_orbs) and z1==z2==1 and z3==z4==0:                      
                if orb1=='dx2y2' and orb2=='dx2y2'  and orb3=='d3z2r2' and orb4=='d3z2r2'  and S_Nid8_12==0:
                    wgt_d8d8[0]+=abs(vecs[i,k])**2                       
                elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='d3z2r2' and orb4=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d8[1]+=abs(vecs[i,k])**2     
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='dx2y2' and orb4=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d8[2]+=abs(vecs[i,k])**2                          
                elif orb1=='dx2y2' and orb2=='dx2y2'  and orb3=='dx2y2' and orb4=='dx2y2' and S_Nid8_12==0:
                    wgt_d8d8[3]+=abs(vecs[i,k])**2  
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and orb3=='dx2y2' and orb4=='dx2y2' and S_Nid8_12==1:
                    wgt_d8d8[4]+=abs(vecs[i,k])**2                  
                elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='d3z2r2' and orb4=='dx2y2' and S_Nid8_12==1:
                    wgt_d8d8[5]+=abs(vecs[i,k])**2 
                elif orb1=='dx2y2' and orb2=='dx2y2' and orb3=='d3z2r2' and orb4=='dx2y2' and S_Cud8_12==1:
                    wgt_d8d8[6]+=abs(vecs[i,k])**2  
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='d3z2r2' and orb4=='d3z2r2' and S_Nid8_12==0:
                    wgt_d8d8[7]+=abs(vecs[i,k])**2  
                elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='d3z2r2' and orb4=='dx2y2' and S_Nid8_12==1:
                    wgt_d8d8[8]+=abs(vecs[i,k])**2                  
                    
                    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs):
                wgt_d8d8[9]+=abs(vecs[i,k])**2 
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==1 and z3==z4==0:
                wgt_d9Ld10L2[8]+=abs(vecs[i,k])**2
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==1 and z2==z3==z4==0:
                wgt_d9d10L3[8]+=abs(vecs[i,k])**2    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==1 and z4==0:
                wgt_d9L2d10L[8]+=abs(vecs[i,k])**2
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z2==1 and z1==z3==z4==0:
                wgt_d10Ld9L2[8]+=abs(vecs[i,k])**2      
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==z4==0:
                wgt_d10d9L3[8]+=abs(vecs[i,k])**2                     
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z2==z3==1 and z1==z4==0:
                wgt_d10L2d9L[8]+=abs(vecs[i,k])**2  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==1 and z4==0:
                wgt_d8Ld10L[8]+=abs(vecs[i,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==1 and z3==z4==0:
                wgt_d8d10L2[8]+=abs(vecs[i,k])**2  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z3==1 and z1==z2==z4==0:
                wgt_d10Ld8L[8]+=abs(vecs[i,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==z4==0:
                wgt_d10d8L2[8]+=abs(vecs[i,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==z4==1:
                wgt_d8L2d10[8]+=abs(vecs[i,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z3==z4==1 and z1==z2==0:
                wgt_d10L2d8[8]+=abs(vecs[i,k])**2                    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z3==z4==1 and z2==0:
                wgt_d9L2d9[8]+=abs(vecs[i,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==1 and z2==z3==z4==0:
                wgt_d9d9L2[8]+=abs(vecs[i,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z3==1 and z2==z4==0:
                wgt_d9Ld9L[16]+=abs(vecs[i,k])**2   
                if i in indices[0]:
                    print (' state ', i, orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3,orb4,s4,x4,y4,z4,\
                           'S_Nid8=', S_Nid8_12, 'Sz_Nid8=', Sz_Nid8_12, 'S_Cud8=', S_Cud8_12, 'Sz_Cud8=', Sz_Cud8_12, \
                           'S_Ni_other=', S_Niother_12, 'Sz_Ni_other=', Sz_Niother_12, \
                           'S_Cu_other=', S_Cuother_12, 'Sz_Cu_other=', Sz_Cuother_12, \
                           ", weight = ", weight)                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==1 and z2==z3==z4==0:
                wgt_d9d8L[8]+=abs(vecs[i,k])**2   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==1 and z3==z4==0:
                wgt_d8d9L[8]+=abs(vecs[i,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==z4==1 and z2==z3==0:
                wgt_d9Ld8[8]+=abs(vecs[i,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z4==1 and z3==0:
                wgt_d8Ld9[8]+=abs(vecs[i,k])**2                   

            sumweight=sumweight+abs(vecs[i,k])**2
        print ('sumweight=',sumweight)
        print ('sumweight1=',sumweight1)
        
     
        
        wgt_d9Ld10L2[2]=wgt_d9Ld10L2[0]+wgt_d9Ld10L2[1]        
        wgt_d9d10L3[2]=wgt_d9d10L3[0]+wgt_d9d10L3[1]          
        wgt_d9L2d10L[2]=wgt_d9L2d10L[0]+wgt_d9L2d10L[1]        
        wgt_d10Ld9L2[2]=wgt_d10Ld9L2[0]+wgt_d10Ld9L2[1]  
        wgt_d10d9L3[2]=wgt_d10d9L3[0]+wgt_d10d9L3[1]        
        wgt_d10L2d9L[2]=wgt_d10L2d9L[0]+wgt_d10L2d9L[1]          
        wgt_d8Ld10L[4]=wgt_d8Ld10L[0]+wgt_d8Ld10L[1]+wgt_d8Ld10L[2]+wgt_d8Ld10L[3]        
        wgt_d10Ld8L[4]=wgt_d10Ld8L[0]+wgt_d10Ld8L[1]+wgt_d10Ld8L[2]+wgt_d10Ld8L[3]    
        wgt_d8d10L2[4]=wgt_d8d10L2[0]+wgt_d8d10L2[1]+wgt_d8d10L2[2]+wgt_d8d10L2[3]          
        wgt_d10d8L2[4]=wgt_d10d8L2[0]+wgt_d10d8L2[1]+wgt_d10d8L2[2]+wgt_d10d8L2[3]            
        wgt_d8L2d10[4]=wgt_d8L2d10[0]+wgt_d8L2d10[1]+wgt_d8L2d10[2]+wgt_d8L2d10[3]          
        wgt_d10L2d8[4]=wgt_d10L2d8[0]+wgt_d10L2d8[1]+wgt_d10L2d8[2]+wgt_d10L2d8[3]            
        wgt_d9L2d9[4]=wgt_d9L2d9[0]+wgt_d9L2d9[1]+wgt_d9L2d9[2]+wgt_d9L2d9[3]
        wgt_d9d9L2[4]=wgt_d9d9L2[0]+wgt_d9d9L2[1]+wgt_d9d9L2[2]+wgt_d9d9L2[3]
        wgt_d9Ld9L[17]=wgt_d9Ld9L[0]+wgt_d9Ld9L[1]+wgt_d9Ld9L[2]+wgt_d9Ld9L[3]+wgt_d9Ld9L[4]+ wgt_d9Ld9L[5]\
                  +wgt_d9Ld9L[6]+wgt_d9Ld9L[7]+wgt_d9Ld9L[8]+wgt_d9Ld9L[9]+wgt_d9Ld9L[10]+wgt_d9Ld9L[11]\
                  +wgt_d9Ld9L[12]+wgt_d9Ld9L[13]+wgt_d9Ld9L[14]+wgt_d9Ld9L[15]
        
        
        wgt_d9d8L[7]=wgt_d9d8L[0]+wgt_d9d8L[1]+wgt_d9d8L[2]+wgt_d9d8L[3]+wgt_d9d8L[4]+wgt_d9d8L[5] +wgt_d9d8L[6]        
        wgt_d8d9L[7]=wgt_d8d9L[0]+wgt_d8d9L[1]+wgt_d8d9L[2]+wgt_d8d9L[3]+wgt_d8d9L[4]+wgt_d8d9L[5] +wgt_d8d9L[6]  
        wgt_d9Ld8[7]=wgt_d9Ld8[0]+wgt_d9Ld8[1]+wgt_d9Ld8[2]+wgt_d9Ld8[3]+wgt_d9Ld8[4]+wgt_d9Ld8[5] +wgt_d9Ld8[6]     
        wgt_d8Ld9[7]=wgt_d8Ld9[0]+wgt_d8Ld9[1]+wgt_d8Ld9[2]+wgt_d8Ld9[3]+wgt_d8Ld9[4]+wgt_d8Ld9[5] +wgt_d8Ld9[6] 
        
        
        path = './data'		# create file

        if os.path.isdir(path) == False:
            os.mkdir(path) 
        
        txt=open('./data/LmLn','a')                                  
        txt.write(str(wgt_LmLn[0])+'\n')
        txt.close()     
               
        
        txt=open('./data/d9Ld10L2_dx2y2','a')                                  
        txt.write(str(wgt_d9Ld10L2[0])+'\n')
        txt.close()  
        txt=open('./data/d9Ld10L2_d3z2r2','a')                                  
        txt.write(str(wgt_d9Ld10L2[1])+'\n')
        txt.close() 
        txt=open('./data/d9Ld10L2','a')                                  
        txt.write(str(wgt_d9Ld10L2[2])+'\n')
        txt.close()         
        
        txt=open('./data/d9d10L3_dx2y2','a')                                  
        txt.write(str(wgt_d9d10L3[0])+'\n')
        txt.close()  
        txt=open('./data/d9d10L3_d3z2r2','a')                                  
        txt.write(str(wgt_d9d10L3[1])+'\n')
        txt.close()
        txt=open('./data/d9d10L3','a')                                  
        txt.write(str(wgt_d9d10L3[2])+'\n')
        txt.close()        
        
        txt=open('./data/d9L2d10L_dx2y2','a')                                  
        txt.write(str(wgt_d9L2d10L[0])+'\n')
        txt.close()  
        txt=open('./data/d9L2d10L_d3z2r2','a')                                  
        txt.write(str(wgt_d9L2d10L[1])+'\n')
        txt.close()
        txt=open('./data/d9L2d10L','a')                                  
        txt.write(str(wgt_d9L2d10L[2])+'\n')
        txt.close()        
        
        txt=open('./data/d10Ld9L2_dx2y2','a')                                  
        txt.write(str(wgt_d10Ld9L2[0])+'\n')
        txt.close()  
        txt=open('./data/d10Ld9L2_d3z2r2','a')                                  
        txt.write(str(wgt_d10Ld9L2[1])+'\n')
        txt.close()
        txt=open('./data/d10Ld9L2','a')                                  
        txt.write(str(wgt_d10Ld9L2[2])+'\n')
        txt.close()        
        
        txt=open('./data/d10d9L3_dx2y2','a')                                  
        txt.write(str(wgt_d10d9L3[0])+'\n')
        txt.close()  
        txt=open('./data/d10d9L3_d3z2r2','a')                                  
        txt.write(str(wgt_d10d9L3[1])+'\n')
        txt.close()    
        txt=open('./data/d10d9L3','a')                                  
        txt.write(str(wgt_d10d9L3[2])+'\n')
        txt.close()         
        
        txt=open('./data/d10L2d9L_dx2y2','a')                                  
        txt.write(str(wgt_d10L2d9L[0])+'\n')
        txt.close()  
        txt=open('./data/d10L2d9L_d3z2r2','a')                                  
        txt.write(str(wgt_d10L2d9L[1])+'\n')
        txt.close()   
        txt=open('./data/d10L2d9L','a')                                  
        txt.write(str(wgt_d10L2d9L[2])+'\n')
        txt.close()          
        
        txt=open('./data/d8Ld10L_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d8Ld10L[0])+'\n')
        txt.close()  
        txt=open('./data/d8Ld10L_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8Ld10L[1])+'\n')
        txt.close() 
        txt=open('./data/d8Ld10L_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d8Ld10L[2])+'\n')
        txt.close()        
        txt=open('./data/d8Ld10L_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8Ld10L[3])+'\n')
        txt.close()        
        txt=open('./data/d8Ld10L','a')                                  
        txt.write(str(wgt_d8Ld10L[4])+'\n')
        txt.close()              
        
        txt=open('./data/d10Ld8L_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d10Ld8L[0])+'\n')
        txt.close()  
        txt=open('./data/d10Ld8L_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d10Ld8L[1])+'\n')
        txt.close() 
        txt=open('./data/d10Ld8L_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d10Ld8L[2])+'\n')
        txt.close()          
        txt=open('./data/d10Ld8L_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d10Ld8L[3])+'\n')
        txt.close()          
        txt=open('./data/d10Ld8L','a')                                  
        txt.write(str(wgt_d10Ld8L[4])+'\n')
        txt.close()               

        txt=open('./data/d8d10L2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d8d10L2[0])+'\n')
        txt.close()  
        txt=open('./data/d8d10L2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8d10L2[1])+'\n')
        txt.close() 
        txt=open('./data/d8d10L2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d8d10L2[2])+'\n')
        txt.close()          
        txt=open('./data/d8d10L2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8d10L2[3])+'\n')
        txt.close()          
        txt=open('./data/d8d10L2','a')                                  
        txt.write(str(wgt_d8d10L2[4])+'\n')
        txt.close()         

        txt=open('./data/d10d8L2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d10d8L2[0])+'\n')
        txt.close()  
        txt=open('./data/d10d8L2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d10d8L2[1])+'\n')
        txt.close()  
        txt=open('./data/d10d8L2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d10d8L2[2])+'\n')
        txt.close()          
        txt=open('./data/d10d8L2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d10d8L2[3])+'\n')
        txt.close()          
        txt=open('./data/d10d8L2','a')                                  
        txt.write(str(wgt_d10d8L2[4])+'\n')
        txt.close()         

        txt=open('./data/d8L2d10_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d8L2d10[0])+'\n')
        txt.close()  
        txt=open('./data/d8L2d10_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8L2d10[1])+'\n')
        txt.close()
        txt=open('./data/d8L2d10_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d8L2d10[2])+'\n')
        txt.close()         
        txt=open('./data/d8L2d10_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8L2d10[3])+'\n')
        txt.close()         
        txt=open('./data/d8L2d10','a')                                  
        txt.write(str(wgt_d8L2d10[4])+'\n')
        txt.close()         
        
        txt=open('./data/d10L2d8_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d10L2d8[0])+'\n')
        txt.close()  
        txt=open('./data/d10L2d8_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d10L2d8[1])+'\n')
        txt.close()
        txt=open('./data/d10L2d8_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d10L2d8[2])+'\n')
        txt.close()         
        txt=open('./data/d10L2d8_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d10L2d8[3])+'\n')
        txt.close()         
        txt=open('./data/d10L2d8','a')                                  
        txt.write(str(wgt_d10L2d8[4])+'\n')
        txt.close()         
        
        txt=open('./data/d9L2d9_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d9L2d9[0])+'\n')
        txt.close()  
        txt=open('./data/d9L2d9_dx2y2_d3z2r2','a')                                  
        txt.write(str(wgt_d9L2d9[1])+'\n')
        txt.close()   
        txt=open('./data/d9L2d9_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d9L2d9[2])+'\n')
        txt.close()  
        txt=open('./data/d9L2d9_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d9L2d9[3])+'\n')
        txt.close()           
        txt=open('./data/d9L2d9','a')                                  
        txt.write(str(wgt_d9L2d9[4])+'\n')
        txt.close()           
        
        txt=open('./data/d9d9L2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d9d9L2[0])+'\n')
        txt.close()        
        txt=open('./data/d9d9L2_dx2y2_d3z2r2','a')                                  
        txt.write(str(wgt_d9d9L2[1])+'\n')
        txt.close()       
        txt=open('./data/d9d9L2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d9d9L2[2])+'\n')
        txt.close()        
        txt=open('./data/d9d9L2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d9d9L2[3])+'\n')
        txt.close()              
        txt=open('./data/d9d9L2','a')                                  
        txt.write(str(wgt_d9d9L2[4])+'\n')
        txt.close()           
        
        txt=open('./data/d9Ld9L_d3z2r2_dx2y2_ss_0','a')                                  
        txt.write(str(wgt_d9Ld9L[0])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_d3z2r2_ss_0','a')                                  
        txt.write(str(wgt_d9Ld9L[1])+'\n')
        txt.close()    
        txt=open('./data/d9Ld9L_d3z2r2_d3z2r2_ss_0','a')                                  
        txt.write(str(wgt_d9Ld9L[2])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_dx2y2_ss_0','a')                                  
        txt.write(str(wgt_d9Ld9L[3])+'\n')
        txt.close()         
        txt=open('./data/d9Ld9L_d3z2r2_dx2y2_sd_0','a')                                  
        txt.write(str(wgt_d9Ld9L[4])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_d3z2r2_sd_0','a')                                  
        txt.write(str(wgt_d9Ld9L[5])+'\n')
        txt.close()    
        txt=open('./data/d9Ld9L_d3z2r2_d3z2r2_sd_0','a')                                  
        txt.write(str(wgt_d9Ld9L[6])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_dx2y2_sd_0','a')                                  
        txt.write(str(wgt_d9Ld9L[7])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_d3z2r2_dx2y2_sd_1','a')                                  
        txt.write(str(wgt_d9Ld9L[8])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_d3z2r2_sd_1','a')                                  
        txt.write(str(wgt_d9Ld9L[9])+'\n')
        txt.close()    
        txt=open('./data/d9Ld9L_d3z2r2_d3z2r2_sd_1','a')                                  
        txt.write(str(wgt_d9Ld9L[10])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_dx2y2_sd_1','a')                                  
        txt.write(str(wgt_d9Ld9L[11])+'\n')
        txt.close()              
        txt=open('./data/d9Ld9L_d3z2r2_dx2y2_ss_1','a')                                  
        txt.write(str(wgt_d9Ld9L[12])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_d3z2r2_ss_1','a')                                  
        txt.write(str(wgt_d9Ld9L[13])+'\n')
        txt.close()    
        txt=open('./data/d9Ld9L_d3z2r2_d3z2r2_ss_1','a')                                  
        txt.write(str(wgt_d9Ld9L[14])+'\n')
        txt.close()      
        txt=open('./data/d9Ld9L_dx2y2_dx2y2_ss_1','a')                                  
        txt.write(str(wgt_d9Ld9L[15])+'\n')
        txt.close()              
        txt=open('./data/d9Ld9L','a')                                  
        txt.write(str(wgt_d9Ld9L[17])+'\n')
        txt.close()      


        txt=open('./data/d9d8L_dx2y2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d9d8L[0])+'\n')
        txt.close()          
        txt=open('./data/d9d8L_d3z2r2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d9d8L[1])+'\n')
        txt.close()              
        txt=open('./data/d9d8L_dx2y2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d9d8L[2])+'\n')
        txt.close()          
        txt=open('./data/d9d8L_d3z2r2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d9d8L[3])+'\n')
        txt.close()        
        txt=open('./data/d9d8L_dx2y2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d9d8L[4])+'\n')
        txt.close()          
        txt=open('./data/d9d8L_d3z2r2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d9d8L[5])+'\n')
        txt.close()  
        txt=open('./data/d9d8L_dx2y2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d9d8L[6])+'\n')
        txt.close()  
        txt=open('./data/d9d8L','a')                                  
        txt.write(str(wgt_d9d8L[7])+'\n')
        txt.close()   
        txt=open('./data/d9d8L_singlet','a')                                  
        txt.write(str(wgt_d9d8L[7]-wgt_d9d8L[3]-wgt_d9d8L[4])+'\n')
        txt.close()    
        txt=open('./data/d9d8L_triplet','a')                                  
        txt.write(str(wgt_d9d8L[3]+wgt_d9d8L[4])+'\n')
        txt.close()            
        
        txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2','a')                                  
        txt.write(str(wgt_d8d9L[0])+'\n')
        txt.close()          
        txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_S1','a')                                  
        txt.write(str(wgt_d8d9L[1])+'\n')
        txt.close()              
        txt=open('./data/d8d9L_dx2y2_dx2y2_d3z2r2','a')                                  
        txt.write(str(wgt_d8d9L[2])+'\n')
        txt.close()          
        txt=open('./data/d8d9L_d3z2r2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d8d9L[3])+'\n')
        txt.close()        
        txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8d9L[4])+'\n')
        txt.close()          
        txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8d9L[5])+'\n')
        txt.close()  
        txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8d9L[6])+'\n')
        txt.close()  
        txt=open('./data/d8d9L','a')                                  
        txt.write(str(wgt_d8d9L[7])+'\n')
        txt.close()    
        txt=open('./data/d8d9L_singlet','a')                                  
        txt.write(str(wgt_d8d9L[7]-wgt_d8d9L[1]-wgt_d8d9L[4])+'\n')
        txt.close()            
        txt=open('./data/d8d9L_triplet','a')                                  
        txt.write(str(wgt_d8d9L[1]+wgt_d8d9L[4])+'\n')
        txt.close()    
        
        
        
        txt=open('./data/d9Ld8_dx2y2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d9Ld8[0])+'\n')
        txt.close()          
        txt=open('./data/d9Ld8_d3z2r2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d9Ld8[1])+'\n')
        txt.close()              
        txt=open('./data/d9Ld8_dx2y2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d9Ld8[2])+'\n')
        txt.close()          
        txt=open('./data/d9Ld8_d3z2r2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d9Ld8[3])+'\n')
        txt.close()        
        txt=open('./data/d9Ld8_dx2y2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d9Ld8[4])+'\n')
        txt.close()          
        txt=open('./data/d9Ld8_d3z2r2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d9Ld8[5])+'\n')
        txt.close() 
        txt=open('./data/d9Ld8_dx2y2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d9Ld8[6])+'\n')
        txt.close() 
        txt=open('./data/d9Ld8','a')                                  
        txt.write(str(wgt_d9Ld8[7])+'\n')
        txt.close()         
        txt=open('./data/d9Ld8_singlet','a')                                  
        txt.write(str(wgt_d9Ld8[7]-wgt_d9Ld8[3]-wgt_d9Ld8[4])+'\n')
        txt.close()    
        txt=open('./data/d9Ld8_triplet','a')                                  
        txt.write(str(wgt_d9Ld8[3]+wgt_d9Ld8[4])+'\n')
        txt.close()         
        
        
        
        txt=open('./data/d8Ld9_d3z2r2_dx2y2_d3z2r2','a')                                  
        txt.write(str(wgt_d8Ld9[0])+'\n')
        txt.close()          
        txt=open('./data/d8Ld9_d3z2r2_dx2y2_d3z2r2_S1','a')                                  
        txt.write(str(wgt_d8Ld9[1])+'\n')
        txt.close()              
        txt=open('./data/d8Ld9_dx2y2_dx2y2_d3z2r2','a')                                  
        txt.write(str(wgt_d8Ld9[2])+'\n')
        txt.close()          
        txt=open('./data/d8Ld9_d3z2r2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d8Ld9[3])+'\n')
        txt.close()        
        txt=open('./data/d8Ld9_d3z2r2_dx2y2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8Ld9[4])+'\n')
        txt.close()          
        txt=open('./data/d8Ld9_d3z2r2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8Ld9[5])+'\n')
        txt.close() 
        txt=open('./data/d8Ld9_dx2y2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8Ld9[6])+'\n')
        txt.close() 
        txt=open('./data/d8Ld9','a')                                  
        txt.write(str(wgt_d8Ld9[7])+'\n')
        txt.close()    
        txt=open('./data/d8Ld9_singlet','a')                                  
        txt.write(str(wgt_d8Ld9[7]-wgt_d8Ld9[1]-wgt_d8Ld9[4])+'\n')
        txt.close()            
        txt=open('./data/d8Ld9_triplet','a')                                  
        txt.write(str(wgt_d8Ld9[1]+wgt_d8Ld9[4])+'\n')
        txt.close()            
        

        
        txt=open('./data/d8d8_dx2y2_dx2y2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d8d8[0])+'\n')
        txt.close()          
        txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2','a')                                  
        txt.write(str(wgt_d8d8[1])+'\n')
        txt.close()              
        txt=open('./data/d8d8_d3z2r2_d3z2r2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8d8[2])+'\n')
        txt.close()          
        txt=open('./data/d8d8_dx2y2_dx2y2_dx2y2_dx2y2','a')                                  
        txt.write(str(wgt_d8d8[3])+'\n')
        txt.close()      
        txt=open('./data/d8d8_d3z2r2_dx2y2_dx2y2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8d8[4])+'\n')
        txt.close()        
        txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8d8[5])+'\n')
        txt.close()    
        txt=open('./data/d8d8_dx2y2_dx2y2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8d8[6])+'\n')
        txt.close()  
        txt=open('./data/d8d8_d3z2r2_d3z2r2_d3z2r2_d3z2r2','a')                                  
        txt.write(str(wgt_d8d8[7])+'\n')
        txt.close() 
        txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_S1','a')                                  
        txt.write(str(wgt_d8d8[8])+'\n')
        txt.close()      
        txt=open('./data/d8d8','a')                                  
        txt.write(str(wgt_d8d8[9])+'\n')
        txt.close()     
        txt=open('./data/d8d8_singlet','a')                                  
        txt.write(str(wgt_d8d8[9]-wgt_d8d8[4]-wgt_d8d8[5]-wgt_d8d8[6]-wgt_d8d8[8])+'\n')
        txt.close()            
        txt=open('./data/d8d8_triplet','a')                                  
        txt.write(str(wgt_d8d8[4]+wgt_d8d8[5]+wgt_d8d8[6]+wgt_d8d8[8])+'\n')
        txt.close()         
        
        sumweight_picture=wgt_LmLn[0]+wgt_d9Ld10L2[2]+wgt_d9d10L3[2]+wgt_d9L2d10L[2]+wgt_d10Ld9L2[2]+wgt_d10d9L3[2]\
                    +wgt_d10L2d9L[2]+wgt_d8Ld10L[4]+wgt_d10Ld8L[4]+wgt_d8d10L2[4]+wgt_d10d8L2[4]+wgt_d8L2d10[4]\
                    +wgt_d10L2d8[4]+wgt_d9L2d9[4]+wgt_d9d9L2[4]+wgt_d9Ld9L[17]+wgt_d9d8L[7]+wgt_d8d9L[7]\
                    +wgt_d9Ld8[7]+wgt_d8Ld9[7]+ wgt_d8d8[9] 
        sumweight2=wgt_LmLn[0]+wgt_d9Ld10L2[8]+wgt_d9d10L3[8]+wgt_d9L2d10L[8]+wgt_d10Ld9L2[8]+wgt_d10d9L3[8]\
                          +wgt_d10L2d9L[8]+wgt_d8Ld10L[8]+wgt_d10Ld8L[8]+wgt_d8d10L2[8]+wgt_d10d8L2[8]\
                          +wgt_d8L2d10[8]+ wgt_d10L2d8[8]+wgt_d9L2d9[8]+wgt_d9d9L2[8]+wgt_d9Ld9L[16]\
                          +wgt_d9d8L[8]+wgt_d8d9L[8]+wgt_d9Ld8[8]+wgt_d8Ld9[8]+ wgt_d8d8[9] 
        
        print ('sumweight2=',sumweight2)        
        print ('sumweight_picture=',sumweight_picture)
        print ('LmLn=',wgt_LmLn[0])
        print ('d9Ld10L2=',wgt_d9Ld10L2[2])
        print ('d9d10L3=',wgt_d9d10L3[2])        
        print ('d9L2d10L=',wgt_d9L2d10L[2])
        print ('d10Ld9L2=',wgt_d10Ld9L2[2])        
        print ('d10d9L3=',wgt_d10d9L3[2])
        print ('d10L2d9L=',wgt_d10L2d9L[2])
        print ('d8Ld10L=',wgt_d8Ld10L[4])
        print ('d10Ld8L=',wgt_d10Ld8L[4])        
        print ('d8d10L2=',wgt_d8d10L2[4])
        print ('d10d8L2=',wgt_d10d8L2[4]) 
        print ('d8L2d10=',wgt_d8L2d10[4])
        print ('d10L2d8=',wgt_d10L2d8[4])
        print ('d9L2d9=',wgt_d9L2d9[4])
        print ('d9d9L2=',wgt_d9d9L2[4])        
        print ('d9Ld9L=',wgt_d9Ld9L[17])        
        print ('d9d8L=',wgt_d9d8L[7])        
        print ('d8d9L=',wgt_d8d9L[7])
        print ('d9Ld8=',wgt_d9Ld8[7])  
        print ('d8Ld9=',wgt_d8Ld9[7])        
        print ('d8d8=',wgt_d8d8[9])

        print ('LmLn=',wgt_LmLn)
        print ('d9Ld10L2=',wgt_d9Ld10L2)
        print ('d9d10L3=',wgt_d9d10L3)        
        print ('d9L2d10L=',wgt_d9L2d10L)
        print ('d10Ld9L2=',wgt_d10Ld9L2)        
        print ('d10d9L3=',wgt_d10d9L3)
        print ('d10L2d9L=',wgt_d10L2d9L)
        print ('d8Ld10L=',wgt_d8Ld10L)
        print ('d10Ld8L=',wgt_d10Ld8L)        
        print ('d8d10L2=',wgt_d8d10L2)
        print ('d10d8L2=',wgt_d10d8L2) 
        print ('d8L2d10=',wgt_d8L2d10)
        print ('d10L2d8=',wgt_d10L2d8)
        print ('d9L2d9=',wgt_d9L2d9)
        print ('d9d9L2=',wgt_d9d9L2)        
        print ('d9Ld9L=',wgt_d9Ld9L)  
        print ('d9d8L=',wgt_d9d8L)        
        print ('d8d9L=',wgt_d8d9L)
        print ('d9Ld8=',wgt_d9Ld8)  
        print ('d8Ld9=',wgt_d8Ld9)        
        print ('d8d8=',wgt_d8d8)  
        
    print("--- get_ground_state %s seconds ---" % (time.time() - t1))
                
    return vals, vecs 

#########################################################################
    # set up Lanczos solver
#     dim  = VS.dim
#     scratch = np.empty(dim, dtype = complex)
    
#     #`x0`: Starting vector. Use something randomly initialized
#     Phi0 = np.zeros(dim, dtype = complex)
#     Phi0[10] = 1.0
    
#     vecs = np.zeros(dim, dtype = complex)
#     solver = lanczos.LanczosSolver(maxiter = 200, 
#                                    precision = 1e-12, 
#                                    cond = 'UPTOMAX', 
#                                    eps = 1e-8)
#     vals = solver.lanczos(x0=Phi0, scratch=scratch, y=vecs, H=matrix)
#     print ('GS energy = ', vals)
    
#     # get state components in GS; note that indices is a tuple
#     indices = np.nonzero(abs(vecs)>0.01)
#     wgt_d8 = np.zeros(6)
#     wgt_d9L = np.zeros(4)
#     wgt_d10L2 = np.zeros(1)

#     print ("Compute the weights in GS (lowest Aw peak)")
#     #for i in indices[0]:
#     for i in range(0,len(vecs)):
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
 
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         orb1 = state['hole1_orb']
#         orb2 = state['hole2_orb']
#         orb3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']

#         #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
#         #    continue
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         o12 = sorted([orb1,orb2,orb3])
#         o12 = tuple(o12)

#         if i in indices[0]:
#             print (' state ', orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3 ,'S=',S12,'Sz=',Sz12,", weight = ", abs(vecs[i,k])**2)
#     return vals, vecs, wgt_d8, wgt_d9L, wgt_d10L2
