'''
pyALMA3 (Python plAnetary Love nuMbers cALculator)
pythonized version of ALMA 3
------------------------------------------------------------------------------

Author: Saikiran Tharimena, Daniele Melini, Giorgio Spada, Steve Vance, Marshall Styczinski
Copyright (C) 2022 Saikiran Tharimena
'''

'''
VERSION CONTROL

v0.1: This is the initial conversion of FORTRAN to Python
'''

#%% Import Libraries
import os
from time import time
import numpy as np
import scipy as sp
from toml import load as tomlload
from itertools import islice, product
from scipy.linalg import lu as ludecomp
from scipy.linalg import solve as solveequ
from scipy.special import binom


#%% Main file
if __name__ == "__main__":
    
    # Read parameters from PARAMS.TOML file
    params = tomlload(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.toml'))

    # Read input model
    fname = os.path.abspath(params['mode']['filename'])
    if fname.split('.', -1)[1] == 'bm':
        # TODO: check this part. need to add parameters to get velocities from rigidity
        COLUMNS, UNITS, MODEL, RAD, RHO, RIG, VIS = read_model_bm(fname)
    else:
        COLUMNS, UNITS, MODEL, LAMBDA, MU, K, SIGMA, Y, RIG, VIS = read_model_pp(fname)


    
 call build_model
 call write_log(1)
!
 call normalization
 call write_log(2)
!
!
!---------------------------------------------- Build the time steps
 write(*,*) ' - Building the time steps'
!
 call time_steps
 call write_log(3)
!
!
!---------------------------------------------- Compute the LNs
!
 call salzer_weights
!
 call cpu_time(time_1)
!
 ndeg = 0
 do n=lmin,lmax,lstep
    ndeg = ndeg+1
 end do
!
 allocate( h_love(ndeg,p+1) )
 allocate( l_love(ndeg,p+1) )
 allocate( k_love(ndeg,p+1) )
!
 h_love = to_fm('0.0')
 l_love = to_fm('0.0')
 k_love = to_fm('0.0')
!
 idx = 0
!
!============================================== Real LNs
 if ( (itype==0 ) .or. (itype==2 ) ) then        
!
    do n=lmin,lmax,lstep
!
       idx = idx+1
!
       do it=1,p+1
!
          f = log( to_fm('2') ) / t(it) 
! 
          do ik=1,2*order
!
            s = f * to_fm(ik)
!
            call love_numbers(n,s,hh,ll,kk)
!
            if( ihist.eq.1 ) then
               fh = to_fm('1')/s
            elseif( ihist.eq.2 ) then
               fh = (to_fm('1') - exp(-s*to_fm(tau))) / (to_fm(tau) * s**2)
            end if
!
            if( flag_rate ) then
               h_love(idx,it) = h_love(idx,it) + fh * (s * hh) * zeta(ik) * f 
               l_love(idx,it) = l_love(idx,it) + fh * (s * ll) * zeta(ik) * f 
               k_love(idx,it) = k_love(idx,it) + fh * (s * kk) * zeta(ik) * f 
            else
               h_love(idx,it) = h_love(idx,it) + fh * hh * zeta(ik) * f 
               l_love(idx,it) = l_love(idx,it) + fh * ll * zeta(ik) * f 
               k_love(idx,it) = k_love(idx,it) + fh * kk * zeta(ik) * f 
            end if
! 
          end do
! 
       end do
!
       call cpu_time(time_2)
!
       write(*,*) ' - Harmonic degree n = ',n,'(',time_2-time_1,'s)' 
       time_1 = time_2
!
    end do
!
!============================================== Complex LNs
 elseif (itype==1) then
!  
    do n=lmin,lmax,lstep
!
       idx = idx+1
!
       do it=1,p+1
!
          omega = 2 * pi / t(it)
          s     = iota * omega
! 
          call love_numbers(n,s,hh,ll,kk)
!
          h_love(idx,it) = hh
          l_love(idx,it) = ll
		  k_love(idx,it) = kk
! 
       end do
!
       call cpu_time(time_2)
!
       write(*,*) ' - Harmonic degree n = ',n,'(',time_2-time_1,'s)' 
       time_1 = time_2	   
!
    end do
!     
 end if
!
!
!
!---------------------------------------------- Write outputs
!
 write(*,*) " - Writing output files '"//trim(file_h)//"', '"// & 
           trim(file_l)//"', '"//trim(file_k)//"'"
!
 call date_and_time(values=idate)
 write(timestamp,'(i4,2(a1,i2.2),1x,i2.2,2(a1,i2.2))') &
    idate(1),'-',idate(2),'-',idate(3), &
    idate(5),':',idate(6),':',idate(7)
!
 open(71,file=trim(file_h),status='unknown')
 open(72,file=trim(file_l),status='unknown')
 open(73,file=trim(file_k),status='unknown')
!
 do iu=71,73 
    write(iu,'(a)') '# ----------------------------------------------------------' 
    if( flag_rate ) then
       if(iload==0) write(iu,'(a)') '# '//ch(iu-70)//'-dot tidal Love number '
       if(iload==1) write(iu,'(a)') '# '//ch(iu-70)//'-dot load Love number '
    else
       if(iload==0) write(iu,'(a)') '# '//ch(iu-70)//' tidal Love number '
       if(iload==1) write(iu,'(a)') '# '//ch(iu-70)//' load Love number '
    end if
    write(iu,'(a)') '# Created by ALMA on '//trim(timestamp)
    write(iu,'(a)') '# ----------------------------------------------------------' 
    write(iu,'(a)') '# '  
 end do
!
!
 if( ifmt==1 ) then
!
   idx=0
! 
   do n=lmin,lmax,lstep
!
      idx = idx + 1
!
      if ((itype==0).or.(itype==2)) then
         write(71,'(i4,1x,3048(e19.8))') n,(to_dp(h_love(idx,it)),it=1,p+1) 
         write(72,'(i4,1x,3048(e19.8))') n,(to_dp(l_love(idx,it)),it=1,p+1) 
         write(73,'(i4,1x,3048(e19.8))') n,(to_dp(k_love(idx,it)),it=1,p+1) 
      elseif (itype==1) then
         write(71,'(i4,1x,3048(e19.8))') n,(to_dpz(h_love(idx,it)),it=1,p+1)
         write(72,'(i4,1x,3048(e19.8))') n,(to_dpz(l_love(idx,it)),it=1,p+1)
         write(73,'(i4,1x,3048(e19.8))') n,(to_dpz(k_love(idx,it)),it=1,p+1)
	  end if
!
   end do
!
 elseif( ifmt==2 ) then
!
   do it=1,p+1
!  
      if ((itype==0).or.(itype==2)) then
         write(71,'(3048(e19.8))') to_dp(t(it)),(to_dp(h_love(idx,it)),idx=1,ndeg) 
         write(72,'(3048(e19.8))') to_dp(t(it)),(to_dp(l_love(idx,it)),idx=1,ndeg) 
         write(73,'(3048(e19.8))') to_dp(t(it)),(to_dp(k_love(idx,it)),idx=1,ndeg) 
      elseif (itype==1) then
         write(71,'(3048(e19.8))') to_dp(t(it)),(to_dpz(h_love(idx,it)),idx=1,ndeg)
         write(72,'(3048(e19.8))') to_dp(t(it)),(to_dpz(l_love(idx,it)),idx=1,ndeg) 
         write(73,'(3048(e19.8))') to_dp(t(it)),(to_dpz(k_love(idx,it)),idx=1,ndeg)	   
      end if
!
   end do
!
 end if
! 
 close(71)
 close(72)
 close(73) 
!
!
!---------------------------------------------- Close the log file
!
 write(*,*) " - Closing the log file '"//trim(file_log)//"'"
 close(99)
! 
!
!
!---------------------------------------------- Release dynamic arrays
!
 deallocate(t)
 deallocate(h_love)
 deallocate(l_love)
 deallocate(k_love)
 deallocate(r)
 deallocate(rho)
 deallocate(mu)
 deallocate(eta)
 deallocate(irheol)
! 
!
!
!---------------------------------------------- All done
!
 call cpu_time(time_1)
 write(*,*) ' - ALMA job completed. Time elapsed: ',time_1-time_0,'s'
 write(*,*) ''
!
!
!
end
!
