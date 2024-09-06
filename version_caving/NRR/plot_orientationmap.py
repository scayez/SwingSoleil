import os 
import numpy as np
import glob
import math
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def custom_function(x,y0,slope,I1, x0, gamma,eta):
    pi=np.pi
    ln2=np.log(2)
    a=(2/gamma)*(ln2/pi)**(1/2)
    b=(4*ln2/(gamma**2))
    return y0+slope*x+I1*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))#+I2*(eta*(a*np.exp(-b*(x-(x0+180))**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))


def azim_profile_fit(chi,I):
    # perform data smoothing
    I=savgol_filter(I,9,2)
    
    
    # define fitting window and extract corresponding arrays
    #we seek the maximum of intensity (peaks can be around -90 or 0!)
    width=180
    valid_indices=np.where((-135<chi)&(chi<45))[0]
    index_in_subset=np.argmax(I[valid_indices])
    index=valid_indices[index_in_subset]
    a=chi[index]-width/2
    b=chi[index]+width/2
    
    # define fitting region
    test=np.where((a<chi)&(chi<b))[0]
    min=test[0]
    max=test[-1]
    #extract arrays corresponding to the fitting region
    x2fit=chi[test]
    xmin=np.min(x2fit);xmax=np.max(x2fit)
    y2fit=I[test]

    # define starting values for refinement
    y0_guess=np.mean(y2fit[0:5])
    slope_guess=0
    I1_guess=np.max(y2fit)
    x0_guess=chi[index]
    gamma_guess=5
    eta_guess=0.5
    init_params=[y0_guess,slope_guess,I1_guess,x0_guess,gamma_guess,eta_guess]

    # define bounds
    lb_G=[0,-np.inf,0,xmin,0,0] # sigma low bounds can be negative in the formula 
    ub_G=[np.inf,np.inf,np.inf,xmax,np.inf,1]
    bounds_G=(lb_G,ub_G) 
    
    # fit the parameters and extract sigmas
    params_PV, _ =curve_fit(custom_function,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf',nan_policy='omit')
    rmse=np.sqrt(np.mean((y2fit-custom_function(x2fit,params_PV[0],params_PV[1],params_PV[2],params_PV[3],params_PV[4],params_PV[5]))**2))
    return params_PV,rmse

def make_plot(dir,figdir, q1=0.008,q2=0.01):
    """
    dir is the path to the directroy containing the data
    q1 and q2 are the q limits where the peak is observed
    """
    path=dir+'integration/'

        # Retrieve the map
    chimap_file=glob.glob(path+'*_chimap.npy')[0]
    qmap_file=glob.glob(path+('*_qmap.npy'))[0]
    imap_file_list=glob.glob(path+'*_imap.npy')



    # Retrieve position
    path=dir+'positions/'
    positions_file_list=glob.glob(path+'*_positions.npy')



    # number_of_files=number of lines
    number_of_files=len(imap_file_list)
    #print('nb files',number_of_files)
    # number of points in file= number of columns
    number_of_points_in_file=np.load(positions_file_list[0]).shape[0]

    q=np.load(qmap_file)
    chi=np.load(chimap_file)
   


    # Step 1: Filter the indices for the given q range
    indices = np.where((q>= q1) & (q <= q2))[0]  # Get the indices of q within the range

    #intialize arrays (note that number_of_files=number_of_lines, number_of_points_in_file=number_of_columns)
    azim_profile_map=np.zeros([number_of_files,number_of_points_in_file,chi.shape[1]])
    
    for i in range(number_of_files):
        I=np.load(imap_file_list[i])
        for j in range(number_of_points_in_file):
            I_filtered=I[j,:,indices]
            I_averaged=np.mean(I_filtered,axis=0)
            azim_profile_map[i,j,:]=I_averaged
    orientation_map=np.zeros([number_of_files,number_of_points_in_file])
    fwhm_map=np.zeros_like(orientation_map)
    eta_map=np.zeros_like(fwhm_map)
    rmse_map=np.zeros_like(eta_map)
    coordinates=np.zeros([number_of_files,number_of_points_in_file,2])
    fig,ax=plt.subplots(number_of_files,number_of_points_in_file,figsize=(20,25))
    for i in range(number_of_files):
        position_file=positions_file_list[i]
        position_array=np.load(position_file)
        #print('pos_array',position_array)
        for j in range(number_of_points_in_file):
            chi_test=chi[j]
            azim_profile_test=azim_profile_map[i,j]
            coordinates[i,j]=position_array[j]
            width=180
            valid_indices=np.where((-135<chi_test)&(chi_test<45))[0]
            index_in_subset=np.argmax(azim_profile_test[valid_indices])
            index=valid_indices[index_in_subset]
        
            a=chi_test[index]-width/2
            b=chi_test[index]+width/2
            try:
                p,rmse=azim_profile_fit(chi_test,azim_profile_test)
                y0=p[0]; slope=p[1]; I1=p[2]; gamma=p[4];eta=p[5]
                x0=p[3]
                fit=custom_function(chi_test,y0,slope,I1,x0,gamma,eta)
                    
                if number_of_points_in_file!=1:
                    ax[i,j].plot(chi_test,azim_profile_test,'-b')
                    ax[i,j].plot(chi_test,fit,'--r')
                    ax[i,j].set_xlim(a,b)
                else:
                    ax[i].plot(chi_test,azim_profile_test,'-b')
                    ax[i].plot(chi_test,fit,'--r')
                    ax[i].set_xlim(a,b)
            
            except:
                rmse=np.nan
                x0=np.nan
                gamma=np.nan
                eta=np.nan
                fit=np.zeros_like(chi_test)
            
            orientation_map[i,j]=x0
            fwhm_map[i,j]=gamma
            eta_map[i,j]=eta
            rmse_map[i,j]=rmse
    # Plot Azimuth profiles and savfig        
    samplename=os.path.basename(os.path.normpath(dir))
    figname=figdir+samplename+'_azim_profile_fittings.png'
    fig.savefig(figname)
    figname=dir+'azim_profile_fittings.png'
    plt.savefig(figname)
    fig.clf()
    # Extract x and y arrays from the coordinates
    x_arrays = coordinates[:, :, 0]  # Extracting x values
    y_arrays = coordinates[:, :, 1]  # Extracting y values
    X_grid = x_arrays  # This will be a 2D array
    Y_grid = y_arrays  # This will be a 2D array

    # Plot FWHM map
    
    plt.figure()
    if number_of_points_in_file!=1:
        contour=plt.contourf(X_grid,Y_grid,fwhm_map,levels=20,cmap='jet')
        #plt.grid(color='black')
        plt.colorbar(contour)
        plt.title('FWHM of azimuth profile peak (°)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        figname=figdir+samplename+'_fwhm_map.png'
        plt.savefig(figname)
        figname=dir+'FWHM_map.png'
        plt.savefig(figname)
        plt.clf()        
    else:
        
        fig,ax=plt.subplots()
        ax.plot(Y_grid,fwhm_map,'.-')
        #ax.set_title('Nanorods orientation (°)')
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('FWHM of azimuth profile peak (°)')
        figname=figdir+samplename+'_fwhm_map.png'
        fig.savefig(figname)
        figname=dir+'FWHM_map.png'
        fig.savefig(figname)
        plt.clf() 

    # Plot eta map
    
    plt.figure()
    if number_of_points_in_file!=1:
        contour=plt.contourf(X_grid,Y_grid,eta_map,levels=20,cmap='jet')
        #plt.grid(color='black')
        plt.colorbar(contour)
        plt.title('Gaussian ratio')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        figname=figdir+samplename+'_eta_map.png'
        plt.savefig(figname)
        figname=dir+'eta_map.png'
        plt.savefig(figname)
        plt.clf()        
    else:
        
        fig,ax=plt.subplots()
        ax.plot(Y_grid,eta_map,'.-')
        #ax.set_title('Nanorods orientation (°)')
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('Gaussian ratio')
        figname=figdir+samplename+'_eta_map.png'
        fig.savefig(figname)
        figname=dir+'eta_map.png'
        fig.savefig(figname)
        fig.clf()

    # Plot rmse map
    
    plt.figure()
    if number_of_points_in_file!=1:
        contour=plt.contourf(X_grid,Y_grid,rmse_map,levels=20,cmap='jet')
        #plt.grid(color='black')
        plt.colorbar(contour)
        plt.title('Root Mean Square Error')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        figname=figdir+samplename+'_rmse_map.png'
        plt.savefig(figname)
        figname=dir+'RMSE_map.png'
        plt.savefig(figname)
        plt.clf()        
    else:
        
        fig,ax=plt.subplots()
        ax.plot(Y_grid,rmse_map,'.-')
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('Root Mean Square Error')
        figname=figdir+samplename+'_rmse_map.png'
        fig.savefig(figname)
        figname=dir+'rmse_map.png'
        fig.savefig(figname)
        fig.clf()



    # Plot orientation map
    

    U=np.zeros([number_of_files,number_of_points_in_file])
    V=np.zeros_like(U)
    # Calculate vectors coordinates at each position
    plt.figure()
    for i in range(number_of_files):
        for j in range(number_of_points_in_file):
            angle=math.radians(orientation_map[i,j]+90)
            U[i,j]=np.cos(angle)
            V[i,j]=np.sin(angle)
    if number_of_points_in_file!=1:
        contour=plt.contourf(X_grid,Y_grid,orientation_map+90,levels=20,cmap='jet')
        #plt.grid(color='black')
        plt.colorbar(contour)
        plt.title('Nanorods orientation (°)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.quiver(X_grid,Y_grid,U,V,scale_units='xy')
        figname=figdir+samplename+'_orientation_map.png'
        plt.savefig(figname)
        figname=dir+'orientation_map.png'
        plt.savefig(figname)
        plt.clf()
    else:
        
        fig,ax=plt.subplots()
        ax.plot(Y_grid,orientation_map+90,'.-')
        #ax.set_title('Nanorods orientation (°)')
        ax.set_xlabel('y (µm)')
        ax.set_ylabel('Nanorod orientation (°)')
        #ax.quiver(X_grid,U,V,scale_units='x')
        figname=figdir+samplename+'_orientation_map.png'
        fig.savefig(figname)
        figname=dir+'orientation_map.png'
        fig.savefig(figname)
        fig.clf()
    