import numpy as np
import matplotlib.pyplot as plt
import h5py
import fabio
import SwingTools as st
from scipy.interpolate import interp1d

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors


def extract_from_h5( filename,
                    print_output: bool = False,             
                    ):
    """Variante de extract from h5 ne renvoyant qu'un dictionnaire contenant toutes les données"""
    with h5py.File(filename, "r") as f:
        #Retrieve Sample Name
        group = list(f.keys())[0]
        sample_name = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')
        #Retrieve Eiger SAXS Data
        target = group + '/scan_data/eiger_image'
        eiger = np.array(f[target])
        if eiger.shape[1]==1: #in some case the format is (number image, 1, pixel height, pixel width), switch to (number image,pixel height, pixel width)
            eiger = eiger.squeeze(axis=1)

        #Calculate eiger mean to mean on all frames
        eiger_mean = np.mean(eiger,axis=0) 
            
        # Retrieve Basler image (microscope)
        basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]

        # Retrieve positions start ,end, number of positions= frames???
        position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
        position_x_end=f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
        position_z_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]
        position_z_end = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position_post'][()]
        position = {'X_start':position_x_start,'X_end': position_x_end, 'Z_start':position_z_start,'Z_end': position_z_end}

         # Retrieve experimental parameters
        target = group + '/SWING/EIGER-4M'
        distance_m = f[target + '/distance'][0] / 1000  # Convert to meters
        pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert to meters
        pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6  # Convert to meters
        x_center = f[target + '/dir_beam_x'][0]
        z_center = f[target + '/dir_beam_z'][0]
        nb_frames=f[target+'/nb_frames'][0]
        bin_x=f[target+'/binning_x'][0]
        bin_y=f[target+'/binning_y'][0]
        exposure_time = f[target+'/exposure_time'][0]
        target = group + '/SWING/i11-c-c03__op__mono'
        wl = f[target + '/wavelength'][0]

        #Retrieve Transmission       
        target = group + '/sample_info/sample_transmission'
        transmission = np.array(f[target])


        target = group + '/scan_data/averagemi8b'
        averagemi8b = np.array(f[target])

                # Retrieve Time Stamps
        target = group + '/scan_data/eiger_timestamp'
        time_stamps = np.array(f[target])

        params={"Sample_Name":sample_name,
                "eiger":eiger,
                "eiger_mean":eiger_mean,
                "basler_image":basler_image,
                "WaveLength":wl,
                "Center_1":x_center,"Center_2":z_center,
                "PixSize_x":pixel_size_x,"PixSize_z":pixel_size_z,
                "SampleDistance":distance_m,
                "Dim_1":eiger.shape[1],"Dim_2":eiger.shape[2],
                'Binning_1':bin_x,'Binning_2':int(bin_y),
                'nb_frames':nb_frames,
                "exposure_time":exposure_time,
                "averagemi8b":averagemi8b,
                "transmission":transmission, 
                "time_stamps":time_stamps,
                "position":position}
            
    if print_output:
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, dict):
                    print(f"{key}: {len(value)} keys")  # Just showing number of keys in the nested dict
                else:
                    print(f"{key}: {value} (type: {type(value).__name__})")
            print('------------------------------------------------------')


    return params

def integrate(params,maskfile,mean=False) :
    """Variante de integrate permettant d'integrer eiger mean ou l'ensemble des eiger"""

    #Retrieve parameters
    pixel_size_x = params['PixSize_x']
    pixel_size_z = params['PixSize_z']
    bin_x = params['Binning_1']
    bin_y = params['Binning_2']
    distance_m = params['SampleDistance']
    x_center = params['Center_1']
    z_center = params['Center_2']
    wl = params['WaveLength']
    nb_frames = params["nb_frames"]

    nbins = 1000            #nb  de points
    unit_type = "q_A^-1"    #unité q


    if mean:
        image = params['eiger_mean']
        detector = pyFAI.detectors.Detector(pixel1=pixel_size_x*bin_x, pixel2=pixel_size_z*bin_y)
        # Create an AzimuthalIntegrator object with the specified distance and detector
        ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
        # Set the fit parameters for the integrator (distance in mm, x and z centers, and wavelength)
        ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance in mm
        mask=fabio.open(maskfile)
        maskdata=mask.data
        # Perform the integration with the mask applied
        q,i=ai.integrate1d(image,nbins,unit=unit_type,normalization_factor=1,mask=maskdata)

        q_array = np.array(q)
        i_array = np.array(i)

     
    else:
        image = params['eiger']

        q_list = []
        i_list = []
        for i in range (nb_frames):
            # Create a pyFAI detector object with the specified pixel sizes and binning factors
            detector = pyFAI.detectors.Detector(pixel1=pixel_size_x*bin_x, pixel2=pixel_size_z*bin_y)
            # Create an AzimuthalIntegrator object with the specified distance and detector
            ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
            # Set the fit parameters for the integrator (distance in mm, x and z centers, and wavelength)
            ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance in mm
            mask=fabio.open(maskfile)
            maskdata=mask.data
            # Perform the integration with the mask applied
            q,i=ai.integrate1d(image[i],nbins,unit=unit_type,normalization_factor=1,mask=maskdata)
       

            q_list.append(q)
            i_list.append(i)
        q_array = np.array(q_list)
        i_array = np.array(i_list)
        
    return q_array, i_array

def correction_mi_time(i,eiger,params):

    "corrige les intensités avec average mi8 et exposure time et le coef 30700 (lié à l'angle solide)"

    for i in range (eiger.shape):
        mi8 = params['averagemi8b']
        time = params['exposure_time']
        i_corr = 1/(time*mi8*30700)

    return i_corr

def ratio(q_0,i_0,q_1,i_1):

    """calcule le facteur multiplicatif ratio entre 2 tableaux numpy """

    # Create an interpolation function
    f = interp1d(q_1, i_1, bounds_error=False, fill_value='extrapolate')
    # Interpolate i_Foxtrot to match the shape of i_pyFAI_raw
    i_1_interpolated = f(q_0)
    ratio = i_0 / i_1_interpolated 
    
    if len( q_1)>len(q_0):
        # Create an interpolation function
        f = interp1d(q_1, i_1, bounds_error=False, fill_value='extrapolate')
        # Interpolate i_Foxtrot to match the shape of i_pyFAI_raw
        i_1_interpolated = f(q_0)
        ratio = i_0 / i_1_interpolated 

    else: 
        # Create an interpolation function
        f = interp1d(q_0, i_0, bounds_error=False, fill_value='extrapolate')
        # Interpolate i_Foxtrot to match the shape of i_pyFAI_raw
        i_0_interpolated = f(q_1)
        ratio = i_1 / i_0_interpolated 

    mean_ratio = np.mean(ratio[1:-1])
    sigma_ratio = np.std(ratio[1:-1]) 
    
    print(f'ratio= {mean_ratio:.3f}, sigma= {sigma_ratio:.3f}')
    return mean_ratio, sigma_ratio


def load_foxtrot(foxtrot_path):
    """ charger q et i d'un fichier foxtrot format txt"""
    foxtrot_sample = np.loadtxt(foxtrot_path, skiprows=21)
    q_foxtrot = foxtrot_sample[:, 0] 
    i_foxtrot = foxtrot_sample[:, 1]   
    return(q_foxtrot,i_foxtrot)    