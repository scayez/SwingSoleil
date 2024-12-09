import numpy as np
import matplotlib.pyplot as plt
import h5py
import fabio
import SwingTools as st
from scipy.interpolate import interp1d

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import os
import pandas as pd
from IPython.display import clear_output


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
                "Binning_1":bin_x,'Binning_2':int(bin_y),
                "nb_frames":nb_frames,
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

    # correction par la transmission
    # transmission = np.mean(params["transmission"])
    # mi8 = np.mean(params["averagemi8b"])
    # exposure_time = params["exposure_time"]

    #i_array_corr = i_array/transmission

    return q_array , i_array






def load_txt(path, skiprows=1):
    """ charger q et i d'un fichier foxtrot format txt"""
    sample = np.loadtxt(path, skiprows=skiprows)
    q = sample[:, 0] 
    i = sample[:, 1]   
    return q, i    


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
                params = extract_from_h5(file_path, print_output=False)
                transmission = params["transmission"]
                transmission= np.mean(transmission)
                # Get only the file name
                file_name = os.path.basename(file_path)
                sample_name = params["Sample_Name"]
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
    print(data_list)       

               # Create a DataFrame from the list
    df = pd.DataFrame(data_list, columns=['File', 'Sample name', 'Mean Transmission'])
    #Add columns to dataframe 
    # Adding the 'Type' column using if-else statements
    df['Type'] = 'unknown'  # Set default value
    
    for index, row in df.iterrows():

        if row['Sample name'].startswith('E'):
            df.at[index, 'Type'] = 'empty'

        elif row['Sample name'].startswith('R'):
            df.at[index, 'Type'] = 'ref'
        # elif row['Sample name'].startswith('E'):
        #     df.at[index, 'Type'] = 'empty'
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

    #Add corrercted time column
    merged['Time_ref'] = pd.to_datetime(merged['Time_ref'])
    min_time_ref = merged['Time_ref'].min()
    
    merged['Time_ref_adjusted'] = merged['Time_ref'] - min_time_ref
    #depasser 24h mais ne pas compter les jours
    merged['Time_ref_adjusted'] =merged['Time_ref_adjusted'].apply(lambda x: str(x).split(' ')[2] if pd.notnull(x) else '00:00:00')

    #add file number column
    merged['File num.'] = merged['File_ref'].apply(lambda x: x.split('rodriguez_')[1].split('_2024')[0])

    # Save DataFrame to CSV file
    df.to_csv(directory+'/_samples_ref.csv', index=False)
    merged.to_csv(directory+'/_samples_ref_associated.csv', index=False)
    
   # Save errors to the text file

    with open(log_file_path, 'a') as f:
        for error in list_error:
            f.write(error + '\n')
    return df, merged

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
