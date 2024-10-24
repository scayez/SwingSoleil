import h5py
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import os
import fabio
from IPython.display import clear_output
from matplotlib import pyplot as plt
import pandas as pd
import shutil
import logging

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
        - eiger_mean (numpy array): The mean Eiger SAXS data across all frames, structured as a 2D array (pixel height, pixel width).
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
        print('Taille Eiger Mean -->',eiger_mean.shape)
        print('Taille Basler -->',basler_image.shape)
        print('Positions -->',position)
        print('Params -->',params)
        print('Time_stamps -->',time_stamps)
        print('Transmission -->',transmission)

    return sample_name, eiger,eiger_mean, basler_image, position, params, time_stamps, transmission

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
    sample_name, eiger,eiger_mean, basler_image, position, params, time_stamps, transmission = extract_from_h5(file,print_output=False)
   
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

   
    #Create folder for all data with sample name and the n°
    output_base = '/'+os.path.basename(file).replace('.h5', '')
    data_dir = os.path.dirname(file) 
    folder_name = '/'+sample_name
    output_dir = data_dir + folder_name
    file_num = file.split('rodriguez_')[1]
    file_num = file_num.split('_2024')[0]
    output_dir = output_dir + '_'+ file_num
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
       # print(output_base+'--> Eiger')

    if basler_image is not None:
         np.save(os.path.join(basler_dir+ output_base + '_basler_image.npy'), basler_image)
        # print(output_base+'--> Basler')

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
      #  print(output_base+'--> Position')
        
    if params is not None:
        params_array = np.array([params],dtype=object)
        np.save(os.path.join(params_dir+output_base + '_params.npy'), params_array)
        #save in text file
        with open(os.path.join(params_dir+ output_base + '_params.txt'), 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
 
    if time_stamps is not None:
        time_stamp_dir = os.path.join(output_dir, 'time_stamp')
        np.save(os.path.join(time_stamp_dir+ output_base + '_time_stamps.npy'), time_stamps)
        np.savetxt(os.path.join(time_stamp_dir+ output_base + '_time_stamps.txt'), time_stamps)
     #   print(output_base+'--> Params')

    if transmission is not None:
        np.save(os.path.join(transmission_dir+ output_base + '_transmission.npy'), transmission)
        np.savetxt(os.path.join(transmission_dir+ output_base + '_transmission.txt'), transmission)
      #  print(output_base+'--> Transmission')
        
    if integration is not None:

        for i in range(nb_frames):
            #q_iso,i_iso = integrate(eiger[i],pixel_size_x,pixel_size_z,bin_x,bin_y,distance_m,x_center, z_center,unit_type,nbins,wl,maskfile)
            integration = integrate(eiger[i],params,maskfile) ##!!!! Le nom integration est deja utilisé ailleurs!!!!!
            q_iso = integration[0]
            i_iso = integration[1]
            integration_col = np.column_stack((q_iso, i_iso))
            np.save(os.path.join(integration_dir+ output_base + '_integration'+str(i)+'.npy'), integration_col)
            header = "q_iso   i_iso"
            np.savetxt(os.path.join(integration_dir+ output_base + '_integration'+str(i)+'.txt'), integration_col,header = header , fmt='%.6f')
        #calculer integration sur la moyenne des images
        eiger_mean = np.mean(eiger,axis=0) 
        integration = integrate(eiger_mean,params,maskfile)
        q_iso = integration[0]
        i_iso = integration[1]
        integration_col = np.column_stack((q_iso, i_iso))
        np.save(os.path.join(integration_dir+ output_base + '_integration_mean.npy'), integration_col)
        header = "q_iso   i_iso"
        np.savetxt(os.path.join(integration_dir+ output_base + '_integration_mean.txt'), integration_col,header = header , fmt='%.6f')
        #plot_integration(integration,eiger_mean,sample_name ,data_folder=data_dir , plot=False , save_path = os.path.join(integration_dir))
        save_images(data_dir,file,integration,eiger_mean, basler_image,integration_dir, eiger_dir,basler_dir, sample_name)
    clear_output(wait=True)

def integrate(image,params, maskfile=None ):
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
        Returns:
    -------
    numpy array
        An array containing:
        - [0] (numpy array): The integrated q-values (q_iso).
        - [1] (numpy array): The integrated intensities (i_iso) corresponding to the q-values.

    Example:
    --------
    integration = integrate(image_data, params, maskfile='mask.tif')
    q_values = integration[0]
    intensities = integration[1]
    """
    #Retrieve parameters
    pixel_size_x = params['PixSize_x']
    pixel_size_z = params['PixSize_z']
    bin_x = params['Binning_1']
    bin_y = params['Binning_2']
    distance_m = params['SampleDistance']
    x_center = params['Center_1']
    z_center = params['Center_2']
    wl = params['WaveLength']

    nbins = 1000            #??????????????
    unit_type = "q_A^-1"    #??????????????

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
        mask=fabio.open(maskfile)
        maskdata=mask.data
        # Perform the integration with the mask applied
        q_iso,i_iso=ai.integrate1d(image,nbins,unit=unit_type,normalization_factor=1,mask=maskdata)

    #return integration containing q_iso, i_iso
    integration = np.array([q_iso,i_iso])
    return integration

def plot_integration(integration,image, sample_name=None, time_stamps=None, data_folder=None, plot=False,save_path=None):
    """
    Plots the integration data along with the provided image.
    Parameters:
    integration (numpy array): A numpy array containing q_iso and i_iso values.
    image (numpy array): The image to be plotted alongside the integration data.
    sample_name (str, optional): The name of the sample. Defaults to None.
    time_stamps (numpy array, optional): Time stamps for the data. Defaults to None.
    data_folder (str, optional): The folder containing the data. Defaults to None.
    plot (bool, optional): Whether to display the plot. Defaults to False.
    save_path (str, optional): The path to save the plot. Defaults to None.
    Returns:
    None
    Example:
    >>> plot_integration(integration, image, sample_name='Sample1', data_folder='Data', plot=True, save_path='c:/.../Plots')
    """
    q_iso = integration[0]
    i_iso = integration[1]
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].imshow(np.log1p(image),cmap='jet')
    ax[1].loglog(q_iso, i_iso)
    ax[1].set_xlabel('q_iso')  
    ax[1].set_ylabel('i_iso') 
    # Add title
    if sample_name is not None:
        if data_folder is not None:
            if data_folder[-1] == '/':
                title = data_folder.split('/')[-2] + '_' + sample_name
            else:
                title = data_folder.split('/')[-1] + '_' + sample_name        
        else:
            title = sample_name
    else:
        title = 'SAXS'
    fig.suptitle(title)
    # Show plot if requested
    if plot:
        plt.show()  # Only show if plot is True
    # Save the figure if a save path is provided
    if save_path is not None:
        fig.savefig(os.path.join(save_path, title + '.png'))  # Ensure the title has a file extension
        plt.close(fig)  # Close the figure after saving
    else:
        plt.close(fig)  # Close if not saving



def save_images(data_dir,file, integration,eiger_mean, basler_image,integration_dir, eiger_dir,basler_dir, sample_name=None):
    
    file_num = file.split('rodriguez_')[1]
    file_num = file_num.split('_2024')[0]

    basler_X0 = 70
    basler_Z0 = 309
    figsize = (8,8)
    fig1, ax1 = plt.subplots( figsize=figsize)
    ax1.imshow(np.log1p(eiger_mean),cmap='jet')

    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.imshow(basler_image,cmap='gray')
    ax2.scatter(basler_X0, basler_Z0, s = 2000, marker = '+', color = 'r', linewidths=2)

    

    if sample_name is not None:
        if data_dir is not None:
            if data_dir[-1] == '/':
                title = data_dir.split('/')[-2] + '_' + sample_name
            else:
                title = data_dir.split('/')[-1] + '_' + sample_name        
        else:
            title = sample_name
    else:
        title = 'SAXS'

    fig1.suptitle(title)
    fig1.savefig(eiger_dir+'/'+'@eiger@'+sample_name+'@'+file_num+'.png' )  
    plt.close(fig1)
    
    fig2.suptitle(title)
    fig2.savefig(basler_dir+'/'+'@basler@'+sample_name+'@'+file_num+'.png' )  
    plt.close(fig2)






def create_info_file(directory):

    log_file_path = os.path.join(directory, 'error_log.txt')
                            
    # Create a list to store the data
    data_list = []
    file_count = 0
    list_error = []
    # Check if directory exists
    if not os.path.isdir(directory):
        with open(log_file_path, 'w') as f:
            f.write(f"Directory not found: {directory}\n")
        return None, None

    files = os.listdir(directory)
    total_files = len([file for file in os.listdir(directory) if file.endswith('.h5')])
    for file in files:
        if file.endswith('.h5'):
            file_count += 1
            try:
                file_path = os.path.join(directory, file)
                sample_name, _, _, _, _, _, _, transmission = extract_from_h5(file_path, print_output=False)
                transmission = np.mean(transmission)
                # Get only the file name
                file_name = os.path.basename(file_path)
                # Append a new element to the list
                data_list.append((file_name, sample_name, transmission))
                print(f'Processing file n° {file_count} on {total_files}')
                print(sample_name)
                clear_output(wait=True)
            except FileNotFoundError as e:
                list_error.append(f"File not found: {file} - {str(e)}")
            except ValueError as e:
                list_error.append(f"Value error in file {file}: {str(e)}")
            except Exception as e:
                list_error.append(f"Unexpected error with file {file}: {str(e)}")

               # Create a DataFrame from the list
    df = pd.DataFrame(data_list, columns=['File', 'Sample name', 'Mean Transmission'])
    #Add columns to dataframe 
    # Adding the 'Type' column using if-else statements
    df['Type'] = 'unknown'  # Set default value
    for index, row in df.iterrows():
        if row['Sample name'].startswith('R'):
            df.at[index, 'Type'] = 'ref'
        elif row['Sample name'].startswith('S'):
            df.at[index, 'Type'] = 'sample'

    # Adding the 'Channel' column
    df['Channel'] = 'unknown'  # Set default value
    for index, row in df.iterrows():
        try:
            channel_value = row['Sample name'].split('_C_')[1][0]
            if channel_value in ['1', '2']:
                df.at[index, 'Channel'] = int(channel_value)
        except (IndexError, ValueError):
            pass  # Default value is already 'unknown'

    # Adding the 'Position' column
    df['Position'] = 'unknown'  # Set default value
    for index, row in df.iterrows():
        try:
            position_value = row['Sample name'].split('_P_')[1].split('_')[0]
            df.at[index, 'Position'] = int(position_value)
        except (IndexError, ValueError):
            pass  # Default value is already 'unknown'

    # Adding the 'Time' column
    df['Time'] = 'unknown'  # Set default value
    for index, row in df.iterrows():
        try:
            time_value = row['File'].split('.h5')[0].split('_')[-2:]  # Get the last two parts (date and time)
            full_time_string = '-'.join(time_value)  # Combine them with a dash
            df.at[index, 'Time'] = pd.to_datetime(full_time_string, format='%Y-%m-%d-%H-%M-%S')  # Convert to datetime
        except (IndexError, ValueError):
            pass  # Default value is already 'unknown'

    # Adding the 'Flow' column
    df['Flow'] = 'unknown'  # Set default value
    for index, row in df.iterrows():
        try:
            if 'SF' in row['Sample name']:
                df.at[index, 'Flow'] = 'SF'
            elif 'LF' in row['Sample name']:
                df.at[index, 'Flow'] = 'LF'
        except Exception as e:
            pass  # Default value is already 'unknown'


    # Separate references and samples
    references = df[df['Type'] == 'ref']
    samples = df[df['Type'] == 'sample']
    # Merge on Channel and Position
    merged = pd.merge(samples, references, on=['Channel', 'Position'], suffixes=('_sample', '_ref'))


    # Save DataFrame to CSV file
    df.to_csv(directory+'/_samples_ref.csv', index=False)
    merged.to_csv(directory+'/_samples_ref_associated.csv', index=False)
    # Print errors at the end
   # Save errors to the text file

    with open(log_file_path, 'a') as f:
        for error in list_error:
            f.write(error + '\n')
    return df, merged

def batch_convert(data_folder,maskfile):
    """
    Processes all .h5 files in the specified data folder, performing data extraction,
    integration, and saving of the results. Additionally, it copies all .png files from
    the data folder and its subfolders to a 'plots' directory.
    Parameters:
    - data_folder (str): The path to the folder containing the .h5 files and subfolders.
    - maskfile (str): The path to the mask file used for integration.
    Returns:
    None
    Example:
    >>> data_folder = 'C:/path/to/data_folder/'
    >>> maskfile = 'C:/path/to/mask.edf'
    >>> batch_convert(data_folder, maskfile)
    This will process all .h5 files in the specified folder and copy all .png files to a 'plots' subfolder.
    """
    # Get a list of all files in the directory
    files = [f for f in os.listdir(data_folder) if f.endswith('.h5')]
    # Loop over the files
    for index, file in enumerate(files):
        try:
            print(f'Processing file {index + 1}/{len(files)}: {file}')
            clear_output(wait=True)
            file_path = os.path.join(data_folder, file)  # Construct the full path for each file
            # Extract data from the file
            sample_name, eiger, eiger_mean, basler_image, position, params, time_stamps, transmission = extract_from_h5(file_path, print_output=False)
            # Perform integration
            integration_mean = integrate(eiger_mean, params, maskfile)
            # Save the exported data
            save_exported_data(file_path, sample_name, eiger, basler_image, position, params, time_stamps, transmission, integration_mean, maskfile)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    #copy images of plots to have together
    plot_folder = data_folder+'plots'
    os.makedirs(plot_folder, exist_ok=True)
        # Walk through the source folder and its subfolders
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Skip the plot folder
            if root == plot_folder:
                continue
            if file.endswith('.png'):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Copy the file to the destination folder
                shutil.copy(file_path, plot_folder)

def  integration_correction(folder,asso,maskfile,coef):
    """
    Perform integration correction for sample and reference data based on provided coefficients.
    This function processes each row in the provided association DataFrame (`asso`), extracting sample 
    and reference data, performing integration, applying corrections, and saving the results to specified 
    directories. It generates plots for the integrated data and positions.
    Parameters:
    - asso (DataFrame): A pandas DataFrame containing association information with columns:
        - "File_sample": The filename of the sample data.
        - "File_ref": The filename of the reference data.
        - "Position": The position of the sample.
        - "Channel": The channel number associated with the sample.
    - maskfile (str): The path to the mask file used for integration.
    - coef (float): The coefficient used to apply correction to the integrated intensity.
    Returns:
    - None: The function saves the processed data and plots to the filesystem and does not return any values.
    Process:
    1. Iterates over each row in the `asso` DataFrame.
    2. Extracts data for the sample and its associated reference using the `extract` function.
    3. Performs integration on the extracted data using the `integ` function.
    4. Applies a correction to the integrated intensity using the provided coefficient.
    5. Creates directories for saving results if they do not exist.
    6. Saves the integrated data and corrected data as both NumPy arrays and text files.
    7. Generates and saves plots for the integrated data and positions.
    Example:
    maskfile = 'mask.edf'
    coef = 0.0181 
    integration_correction(asso,maskfile,coef)
    """
    for index, row in asso.iterrows(): 
        sample_file = str(row["File_sample"])
        ref_file = str(row["File_ref"])
        sample_path = folder+'/'+sample_file
        ref_path = folder+'/'+ref_file
        output_base = '/'+os.path.basename(sample_file).replace('.h5', '')

        #extract data for sample and associated reference
        sample_name, sample_eiger, sample_eiger_mean, sample_basler_image, sample_position, sample_params, sample_time_stamps, sample_transmission = extract_from_h5(sample_path,print_output=False)
        ref_name, ref_eiger, ref_eiger_mean, ref_basler_image, ref_position, ref_params, ref_time_stamps, ref_transmission = extract_from_h5(ref_path,print_output=False)
        #Do integration
        sample_integration_mean = integrate(sample_eiger_mean,sample_params,maskfile)
        ref_integration_mean = integrate(ref_eiger_mean,ref_params,maskfile)

        sample_q_iso = sample_integration_mean[0]
        sample_i_iso = sample_integration_mean[1]

        ref_q_iso = ref_integration_mean[0]
        ref_i_iso = ref_integration_mean[1]

        #apply correction
        corr_i_iso = coef*((sample_i_iso/np.mean(sample_transmission))-(ref_i_iso/np.mean(ref_transmission)))

        # Create a new directory with the sample name
        file_num = sample_file.split('rodriguez_')[1].split('_')[0]
        sample_dir = os.path.join(folder, sample_name+'_'+file_num)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Create a new directory for integration correction inside sample_dir
        integration_corr_dir = os.path.join(sample_dir, 'integration_corr')
        if not os.path.exists(integration_corr_dir):
            os.makedirs(integration_corr_dir)

         # define directory for position plot
        position_dir = os.path.join(sample_dir, 'positions')
        if not os.path.exists(position_dir):
            os.makedirs(position_dir)


        #save to files
        sample_integration_col = np.column_stack(( sample_q_iso,  sample_i_iso))
        ref_integration_col = np.column_stack((ref_q_iso, ref_i_iso))
        corr_integration_col = np.column_stack((sample_q_iso, corr_i_iso))

        # np.save(os.path.join(integration_corr_dir, f'sample_integration_{sample_name}.npy'), sample_integration_col)
        # np.save(os.path.join(integration_corr_dir, f'ref_integration_{sample_name}.npy'), ref_integration_col)
        # np.save(os.path.join(integration_corr_dir, f'corr_integration_{sample_name}.npy'), corr_integration_col)
        np.save(integration_corr_dir+output_base+'_sample_integration_'+sample_name+'.npy', sample_integration_col)
        np.save(integration_corr_dir+output_base+ '_ref_integration_'+sample_name+'.npy', ref_integration_col)
        np.save(integration_corr_dir+output_base+ '_corr_integration_'+sample_name+'.npy', corr_integration_col)
        
        #plt.savefig(integration_corr_dir+output_base+'_integr_corr_'+sample_name+'.png')


        header = "sample_q_iso   sample_i_iso"
        np.savetxt(integration_corr_dir+output_base+'_sample_integration_'+sample_name+'.txt', sample_integration_col,header = header , fmt='%.6f')
        #np.savetxt(os.path.join(integration_corr_dir, f'sample_integration_{sample_name}.txt'), sample_integration_col,header = header , fmt='%.6f')
        header = "ref_q_iso   ref_i_iso"
        np.savetxt(integration_corr_dir+output_base+ '_ref_integration_'+sample_name+'.txt', ref_integration_col,header = header , fmt='%.6f')
        #np.savetxt(os.path.join(integration_corr_dir, f'ref_integration_{sample_name}.txt'), ref_integration_col,header = header , fmt='%.6f')
        header = "corr_q_iso   corr_i_iso"
        np.savetxt(integration_corr_dir+output_base+ '_corr_integration_'+sample_name+'.txt', corr_integration_col,header = header , fmt='%.6f')
        #np.savetxt(os.path.join(integration_corr_dir, f'corr_integration_{sample_name}.txt'), corr_integration_col,header = header , fmt='%.6f')

        ###############SAVE INTEGRATION COR PLOT###############
        figsize = (8,8)
        fig,ax = plt.subplots(figsize=figsize )
        ax.loglog(sample_q_iso, sample_i_iso, label='Sample I_iso', color='lightblue', linestyle=':')
        ax.loglog(ref_q_iso, ref_i_iso, label='Ref I_iso', color='lightgreen', linestyle=':')
        ax.loglog(sample_q_iso, corr_i_iso, label='Corr I_iso', color='red', linestyle='-')
        # Add labels and title
        ax.set_xlabel('q_iso')
        ax.set_ylabel('I_iso')
        ax.set_title(f' {sample_name}')
        ax.legend()
        # Save the figure
        # plt.savefig(os.path.join(integration_corr_dir, f'plot_iteration_{sample_name}.png'))
        plt.savefig(integration_corr_dir+'/'+'@corr_integration_@'+sample_name+'@'+file_num+'.png')
        print(integration_corr_dir+'/'+'@corr_integration_@'+sample_name+'@'+file_num+'.png')
        plt.close(fig)  # Close the figure to free up memory

         #save all other files
        save_exported_data(sample_path,sample_name, sample_eiger, sample_basler_image, sample_position, sample_params, 
                       sample_time_stamps, sample_transmission, sample_integration_mean ,maskfile)
        ###############SAVE POSITIONS PLOT###############
        fig,ax = plt.subplots(figsize=figsize )
        # Clear the plot to start fresh in each iteration
        ax.clear()   
        #fig.patch.set_facecolor('black')  # Black background for figure
        ax.set_facecolor('lightgrey')  # Black background for axes
        # Set labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Channel')
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2])
        x_min = int(asso['Position'].min())
        x_max = int(asso['Position'].max())
        ax.set_xticks(np.arange(x_min, x_max + 1, 1))  # Set step of 1 for x-axis ticks
        ax.axhline(y=1, color='black',  linewidth=50,zorder=1)
        ax.axhline(y=2, color='black',  linewidth=50,zorder=1)
        # Plot all points in blue (ensures all data is always visible)
        ax.scatter(asso['Position'], asso['Channel'],marker='+', color='yellow',s = 100, zorder=2)
        # Plot the current point in red and bigger
        ax.scatter(row['Position'], row['Channel'], marker='*',color='red', s=1400, zorder=2)
        # Save the current plot as an image
        #plt.savefig(os.path.join(position_dir, f'plot_position_{sample_name}.png'))
        plt.savefig(position_dir+'/'+'@pos_@'+sample_name+'@'+file_num+'.png')
        print(position_dir+'/'+'@pos_@'+sample_name+'@'+file_num+'.png')
        plt.close()
    
        print(f'File: {index+1} on {len(asso)} ')