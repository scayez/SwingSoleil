import h5py
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import os
import fabio
from IPython.display import clear_output

def extract_from_h5( filename,
                    print_output: bool = False,             
                    ):
    """
    Extracts data from an HDF5 file Beamline SWING at Synchrotron Soleil.
    This function reads various datasets from the specified HDF5 file, including sample information,
    Eiger SAXS data, Basler image data, position information, experimental parameters, time stamps, 
    and transmission data. It can optionally print the extracted data to the console for verification.
    Parameters:
    ----------
    filename : str
        The path to the HDF5 file to extract data from. This file should contain the necessary datasets
        structured in a specific format.
    print_output : bool, optional
        Whether to print the extracted data to the console. Defaults to False. If set to True, the function
        will print the sample name, dimensions of the Eiger and Basler data, position information, 
        experimental parameters, time stamps, and transmission data.
    Returns:
    -------
    tuple
        A tuple containing the following extracted data:
        - sample_name (str): The name of the sample extracted from the HDF5 file.
        - eiger (numpy array): The Eiger SAXS data, structured as a 3D array (number of images, pixel height, pixel width).
        - basler_image (numpy array): The Basler image data, structured as an array.
        - position (dict): A dictionary containing the start and end positions for X and Z axes:
            - 'X_start': Start position for X-axis.
            - 'X_end': End position for X-axis.
            - 'Z_start': Start position for Z-axis.
            - 'Z_end': End position for Z-axis.
        - params (dict): A dictionary containing experimental parameters such as:
            - 'Sample_Name': The name of the sample.
            - 'WaveLength': The wavelength of the radiation used, in meters.
            - 'Center_1': The X-coordinate of the beam center.
            - 'Center_2': The Z-coordinate of the beam center.
            - 'PixSize_x': The pixel size in the X direction, in meters.
            - 'PixSize_z': The pixel size in the Z direction, in meters.
            - 'SampleDistance': The distance from the sample to the detector, in meters.
            - 'Dim_1': The first dimension size of the Eiger data (number of pixels in height).
            - 'Dim_2': The second dimension size of the Eiger data (number of pixels in width).
            - 'ExposureTime': The exposure time for the data collection.
            - 'Binning_1': The binning factor in the X direction.
            - 'Binning_2': The binning factor in the Z direction.
            - 'nb_frames': The number of frames in the Eiger data.
        - time_stamps (numpy array): The time stamps corresponding to each Eiger data frame.
        - transmission (numpy array): The transmission data for the sample.
    Example:
    --------
    sample_name, eiger_data, basler_data, position_info, experimental_params, timestamps, transmission_data = extract_from_h5('data_file.h5', print_output=True)
    """  
    with h5py.File(filename, "r") as f:
        #Retrieve Sample Name
        group = list(f.keys())[0]
        sample_name = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')

        #RetrieveEiger SAXS Data
        target = group + '/scan_data/eiger_image'
        eiger = np.array(f[target])
        if eiger.shape[1]==1: #in some case the format is (number image, 1, pixel height, pixel width), switch to (number image,pixel height, pixel width)
            eiger = eiger.squeeze(axis=1)
            
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
        target = group + '/SWING/i11-c-c03__op__mono'
        wl = f[target + '/wavelength'][0]

        params={"Sample_Name":str(sample_name),"WaveLength":float(wl),"Center_1":float(x_center),"Center_2":float(z_center),"PixSize_x":float(pixel_size_x),"PixSize_z":float(pixel_size_z),
            "SampleDistance":float(distance_m),"Dim_1":int(eiger.shape[1]),"Dim_2":int(eiger.shape[2]),
            "ExposureTime":1,'Binning_1':int(bin_x),'Binning_2':int(bin_y),'nb_frames':int(nb_frames)}
               
        # Retrieve Time Stamps
        target = group + '/scan_data/eiger_timestamp'
        time_stamps = np.array(f[target])

        #Retrieve Transmission       
        target = group + '/sample_info/sample_transmission'
        transmission = np.array(f[target])
    
    if print_output:
        print('Sample Name -->',sample_name)
        print('Taille Eiger -->',eiger.shape)
        print('Taille Basler -->',basler_image.shape)
        print('Positions -->',position)
        print('Params -->',params)
        print('Time_stamps -->',time_stamps)
        print('Transmission -->',transmission)

    return sample_name, eiger, basler_image, position, params, time_stamps, transmission

def save_exported_data(file,sample_name=None, eiger=None, basler_image=None, position=None, params=None, time_stamps=None, transmission=None, integration=None, maskfile=None ):
    """
    Saves exported data to various files.
    This function saves the provided data to different files in a structured directory.
    If no data is provided, it will attempt to extract the necessary data from the specified HDF5 file.
    Parameters:
    ----------
    file : str
        The path to the HDF5 file to extract data from. This file should contain the necessary datasets
        structured in a specific format.
    sample_name : str, optional
        The name of the sample. If not provided, it will be extracted from the HDF5 file.
    eiger : numpy array, optional
        The Eiger SAXS data. If not provided, it will be extracted from the HDF5 file.
    basler_image : numpy array, optional
        The Basler image data. If not provided, it will be extracted from the HDF5 file.
    position : dict, optional
        A dictionary containing the start and end positions for X and Z axes. If not provided, it will be extracted from the HDF5 file.
    params : dict, optional
        A dictionary containing experimental parameters. If not provided, it will be extracted from the HDF5 file.
    time_stamps : numpy array, optional
        The time stamps corresponding to each Eiger data frame. If not provided, it will be extracted from the HDF5 file.
    transmission : numpy array, optional
        The transmission data for the sample. If not provided, it will be extracted from the HDF5 file.
    integration : numpy array, optional
        The integration data. If not provided, it will be calculated using the provided Eiger data and other parameters.
    maskfile : str, optional
        The path to the mask file used for integration. If not provided, it will be assumed to be None.
    Returns:
    -------
    None
    """
    #Retrieve all data from file
    sample_name, eiger, basler_image, position, params, time_stamps, transmission = extract_from_h5(file,print_output=False)
    nb_frames = params["nb_frames"]
    position_x_start = position['X_start']
    position_x_end = position['X_end']
    position_z = position['Z_start']
    pixel_size_x = params["PixSize_x"]
    pixel_size_z = params["PixSize_z"]
    bin_x = params['Binning_1']
    bin_y = params['Binning_2']
    distance_m = params["SampleDistance"]
    x_center = params["Center_1"]
    z_center = params["Center_2"]
    wl = params["WaveLength"]
    nbins = 1000            #??????????????
    unit_type = "q_A^-1"     #??????????????
   
    #Create folder for all data with sample name
    output_base = '/'+os.path.basename(file).replace('.h5', '')
    data_dir = os.path.dirname(file) 
    folder_name = '/'+sample_name
    output_dir = data_dir + folder_name
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories if necessary
    eiger_dir = os.path.join(output_dir, 'Eiger_image')
    basler_dir = os.path.join(output_dir, 'basler_image')
    positions_dir = os.path.join(output_dir, 'positions')
    params_dir = os.path.join(output_dir, 'params')
    time_stamp_dir = os.path.join(output_dir, 'time_stamp')
    transmission_dir  = os.path.join(output_dir, 'transmission')
    integration_dir = os.path.join(output_dir, 'integration')

    os.makedirs(eiger_dir, exist_ok=True)
    os.makedirs(positions_dir, exist_ok=True) 
    os.makedirs(basler_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(time_stamp_dir , exist_ok=True)
    os.makedirs(transmission_dir , exist_ok=True)
    os.makedirs(integration_dir, exist_ok=True)

    #Save files
    if eiger is not None:
        np.save(os.path.join(eiger_dir+  output_base + '_eiger.npy'), eiger)
        print(os.path.join(eiger_dir+  output_base + '_eiger.npy'))

    if basler_image is not None:
         np.save(os.path.join(basler_dir+ output_base + '_basler_image.npy'), basler_image)
         print(os.path.join(basler_dir+ output_base + '_basler_image.npy'))

    if position is not None:

        position_array = np.zeros([nb_frames,4])
        for i in range(nb_frames):
            position_array[i][0] = position['X_start']
            position_array[i][1] = position['X_end']
            position_array[i][2] = position['Z_start']
            position_array[i][3] = position['Z_end']
             
        np.save(os.path.join(positions_dir+ output_base + '_positions.npy'), position_array)
        header = "X_start   X_end  Z_start    Z_end"
        np.savetxt(os.path.join(positions_dir+output_base + '_positions.txt'), position_array, header=header, fmt='%.6f')
        print(os.path.join(positions_dir+ output_base + '_positions.npy'))
        
    if params is not None:
        params_array = np.array([params],dtype=object)
        np.save(os.path.join(params_dir+output_base + '_params.npy'), params_array)
        #save in text file
        with open(os.path.join(params_dir+ output_base + '_params.txt'), 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        print(os.path.join(params_dir+ output_base + '_params.npy'))
    
    if time_stamps is not None:
        time_stamp_dir = os.path.join(output_dir, 'time_stamp')
        np.save(os.path.join(time_stamp_dir+ output_base + '_time_stamps.npy'), time_stamps)
        np.savetxt(os.path.join(time_stamp_dir+ output_base + '_time_stamps.txt'), time_stamps)
        print(os.path.join(time_stamp_dir+ output_base + '_time_stamps.npy'))

    if transmission is not None:
        np.save(os.path.join(transmission_dir+ output_base + '_transmission.npy'), transmission)
        np.savetxt(os.path.join(transmission_dir+ output_base + '_transmission.txt'), transmission)
        print(os.path.join(transmission_dir+ output_base + '_transmission.npy'))

    if integration is not None:
        print(os.path.join(integration_dir+ output_base + '_integration.npy'))

        for i in range(nb_frames):
            q_iso,i_iso = integrate(eiger[i],pixel_size_x,pixel_size_z,bin_x,bin_y,distance_m,x_center, z_center,unit_type,nbins,wl,maskfile)
            integration = np.column_stack((q_iso, i_iso))
            np.save(os.path.join(integration_dir+ output_base + '_integration'+str(i)+'.npy'), integration)
            header = "q_iso   i_iso"
            np.savetxt(os.path.join(integration_dir+ output_base + '_integration'+str(i)+'.txt'), integration,header = header , fmt='%.6f')

        #calculer integration sur la moyenne des images
        eiger_mean = np.mean(eiger,axis=0) 
        q_iso,i_iso = integrate(eiger_mean,pixel_size_x,pixel_size_z,bin_x,bin_y,distance_m,x_center, z_center,unit_type,nbins,wl,maskfile)
        integration = np.column_stack((q_iso, i_iso))
        np.save(os.path.join(integration_dir+ output_base + '_integration_mean.npy'), integration)
        header = "q_iso   i_iso"
        np.savetxt(os.path.join(integration_dir+ output_base + '_integration_mean.txt'), integration,header = header , fmt='%.6f')

def integrate(image,pixel_size_x,pixel_size_z,bin_x,bin_y,distance_m,x_center, z_center,unit_type,nbins, wl, maskfile=None ):
    """
    Performs azimuthal integration on an image.
    This function integrates an image using the pyFAI library, taking into account the detector geometry,
    distance, and wavelength. It can also apply a mask to the image if provided.
    Parameters:
    ----------
    image : numpy array
        The 2D image to be integrated.
    pixel_size_x : float
        The size of a pixel in the x-direction.
    pixel_size_z : float
        The size of a pixel in the z-direction.
    bin_x : int
        The binning factor in the x-direction.
    bin_y : int
        The binning factor in the y-direction.
    distance_m : float
        The distance from the sample to the detector in meters.
    x_center : float
        The x-coordinate of the beam center.
    z_center : float
        The z-coordinate of the beam center.
    unit_type : str
        The unit type for the integration (e.g., 'q_A^-1').
    nbins : int
        The number of bins for the integration.
    wl : float
        The wavelength of the radiation used.
    maskfile : str, optional
        The path to a mask file to be applied to the image. If not provided, no mask will be applied.
    Returns:
    -------
    tuple
        A tuple containing the integrated q-values (q_iso) and intensities (i_iso).
    """
    # Create a pyFAI detector object with the specified pixel sizes and binning factors
    detector = pyFAI.detectors.Detector(pixel1=pixel_size_x*bin_x, pixel2=pixel_size_z*bin_y)
    # Create an AzimuthalIntegrator object with the specified distance and detector
    ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
    # Set the fit parameters for the integrator (distance in mm, x and z centers, and wavelength)
    ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance in mm
    # If no mask file is provided,Perform the integration without a mask
    if maskfile is None:
        q_iso,i_iso=ai.integrate1d(image,nbins,unit_type,normalization_factor=1) 
    else: 
        #open maskfile it and extract the mask data
        mask=fabio.open(maskfile)
        maskdata=mask.data
        # Perform the integration with the mask applied
        q_iso,i_iso=ai.integrate1d(image,nbins,unit=unit_type,normalization_factor=1,mask=maskdata)

    return q_iso, i_iso




def extract_sample_names(directory):
    # Create a list to store the data
    sample_names_list = []

    file_count = 0
    total_files = sum(len(files) for _, _, files in os.walk(directory))
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                file_count += 1
                try:
                    file_path = os.path.join(root, file)
                    sample_name, _, _, _, _, _, _ = extract_from_h5(file_path, print_output=False)
                    # Append a new element to the list
                    sample_names_list.append((file_path, sample_name))
                    print(f'Processing file n° {file_count} on {total_files}')
                    clear_output(wait=True)
                except:
                    print('error on ', file_path)

    # Convert the list to a structured NumPy array
    sample_names = np.array(sample_names_list, dtype=[('file_path', 'U200'), ('sample_name', 'U200')])

    return sample_names

