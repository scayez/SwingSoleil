import h5py
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import os

def extract_from_h5( filename,
                    print_output: bool = False,             
                    ):
    """
    Extracts data from an HDF5 file.
    Parameters:
    filename (str): The path to the HDF5 file to extract data from.
    print_output (bool, optional): Whether to print the extracted data to the console. Defaults to False.
    plot_output (bool, optional): Whether to plot the extracted data. Defaults to False.
    Returns:
    tuple: A tuple containing the following extracted data:
        - sample_name (str): The name of the sample.
        - eiger (numpy array): The Eiger SAXS data.
        - basler_image (numpy array): The Basler image data.
        - position (dict): A dictionary containing the start and end positions for X and Z axes.
        - params (dict): A dictionary containing experimental parameters such as wavelength, center, pixel size, sample distance, and binning.
        - time_stamps (numpy array): The time stamps for the Eiger data.

    """
     
     #Sample Name
    with h5py.File(filename, "r") as f:

        #Retrieve Sample Name
        group = list(f.keys())[0]
        sample_name = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')

        #RetrieveEiger SAXS Data
        target = group + '/scan_data/eiger_image'
        eiger = np.array(f[target])

        if eiger.shape[1]==1: #in some case the format is (number image, 1, pixel height, pixel width), switch to (number image,pixel height, pixel width)
            eiger = eiger.squeeze(axis=1)
            
        # Retrieve Basler image
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

        params={"WaveLength":float(wl),"Center_1":float(x_center),"Center_2":float(z_center),"PixSize_x":float(pixel_size_x),"PixSize_z":float(pixel_size_z),
            "SampleDistance":float(distance_m),"Dim_1":int(eiger.shape[1]),"Dim_2":int(eiger.shape[2]),
            "ExposureTime":1,'Binning_1':int(bin_x),'Binning_2':int(bin_y)}
               
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

def save_exported_data(file,data_dir,sample_name, eiger, basler_image, position, params, time_stamps, transmission, integration ):

    output_base = '/'+os.path.basename(file).replace('.h5', '')

 

    #Create folder for all data with sample name
    folder_name = sample_name
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
    os.makedirs(integration_dir, exist_ok=True)
    os.makedirs(positions_dir, exist_ok=True) 
    os.makedirs(basler_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(time_stamp_dir , exist_ok=True)
    os.makedirs(transmission_dir , exist_ok=True)
    os.makedirs(integration_dir, exist_ok=True)

    #Save files
    if eiger is not None:
        np.save(os.path.join(eiger_dir+  output_base + '_eiger.npy'), eiger)
        #np.save(os.path.join(data_dir, output_base + '_images.npy'), data)
        # print(eiger_dir)
        # print(os.path.join(eiger_dir+ '/eiger.npy'))
        

    # if basler_image is not None:
    #     np.save(os.path.join(basler_dir, output_base + '_basler_image.npy'), basler_image)

    # if position is not None:
    #     # Save the (x,z) coordinates associated with each diffusion image
    #     positions=np.zeros([nb_frames,2])
    #     step=(position_x_end-position_x_start)/(nb_frames-1)
    #     for i in range(nb_frames):
    #         x=position_x_start+i*step
    #         positions[i][0]=x; positions[i][1]=position_z
     
    #     np.save(os.path.join(positions_dir, output_base + '_positions.npy'), positions)
        
    
    # if params is not None:
    #     params_array = np.array([params],dtype=object)
    #     np.save(os.path.join(params_dir, output_base + '_params.npy'), params_array)
    #     #save in text file
    #     with open(os.path.join(params_dir, output_base + '_params.txt'), 'w') as f:
    #         for key, value in params.items():
    #             f.write(f"{key}: {value}\n")
    
    # if time_stamps is not None:
    #     time_stamp_dir = os.path.join(output_dir, 'time_stamp')
    #     np.save(os.path.join(time_stamp_dir, output_base + '_time_stamps.npy'), time_stamps)
    #     np.savetxt(os.path.join(time_stamp_dir, output_base + '_time_stamps.txt'), time_stamps)

    # if transmission is not None:
    #     np.save(os.path.join(transmission, output_base + '_images.npy'), transmission)





        #







def integrate(image,pixel_size_x,pixel_size_z,bin_x,bin_y,distance_m,x_center, z_center,unit_type,nbins, wl, maskdata=None ):
    detector = pyFAI.detectors.Detector(pixel1=pixel_size_x*bin_x, pixel2=pixel_size_z*bin_y)
    ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
    ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance in mm

    if maskdata is None:
        q_iso,i_iso=ai.integrate1d(image,nbins,unit_type,normalization_factor=1)
    else: 
        q_iso,i_iso=ai.integrate1d(image,nbins,unit=unit_type,normalization_factor=1,mask=maskdata)

    return q_iso, i_iso