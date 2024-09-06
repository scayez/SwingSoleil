import h5py
import os
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import fabio
from matplotlib import pyplot as plt
import glob
from IPython.display import clear_output

def nxs_swing_HandV_integration(
        nxsfile: str, 
        output_dir :str,
        offset:float = 0,
        sector_angle: float = 10,
        maskfile: str = None,
        save_data: bool = True, 
        save_positions: bool = True, 
        save_basler_image: bool = True     
    ) -> None:
    """
    Perform horizontal and vertical integrations on a SWING .nxs file.
    
    Args:
        nxsfile (str): Path to the .nxs file.
        output_dir (str): Directory to save the output data.
        offset (float): Offset angle for the azimuthal integration.
        sector_angle (float): Sector opening angle for the integration.
        maskfile (str, optional): Path to the mask file (optional).
        save_data (bool): Whether to save the normalized averaged images.
        save_positions (bool): Whether to save the x and z positions.
        save_basler_image (bool): Whether to save the Basler image.
    
    Returns:
        None
    """
    output_base = os.path.basename(nxsfile).replace('.h5', '')
    
    # Create subdirectories if necessary
    data_dir = os.path.join(output_dir, 'image')
    integration_dir = os.path.join(output_dir, 'integration')
    positions_dir = os.path.join(output_dir, 'positions')
    basler_dir = os.path.join(output_dir, 'basler_image')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(integration_dir, exist_ok=True)
    os.makedirs(positions_dir, exist_ok=True)
    os.makedirs(basler_dir, exist_ok=True)

    with h5py.File(nxsfile, "r") as f:
        group = list(f.keys())[0]
        nb_frames = f[group + '/SWING/EIGER-4M/nb_frames'][0]
        
        # Retrieve image data
        target = group + '/scan_data/eiger_image'
        data = np.array(f[target])
        
        # Retrieve experimental parameters
        target = group + '/SWING/EIGER-4M'
        distance_m = f[target + '/distance'][0] / 1000  # Convert to meters
        pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert to meters
        pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6  # Convert to meters
        x_center = f[target + '/dir_beam_x'][0]
        z_center = f[target + '/dir_beam_z'][0]
        nb_frames=f[target+'/nb_frames'][0]
        
        # Retrieve wavelength
        target = group + '/SWING/i11-c-c03__op__mono'
        wl = f[target + '/wavelength'][0]
        
        # Retrieve positions: our h5 file contains all the images of a single line
        # In the h5 we retrieve x_start, x_end, and constant z
        position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
        position_x_end=f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
        position_z = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]

        # Retrieve Basler image
        basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]

    # Save data according to the chosen options
    if save_data:
        np.save(os.path.join(data_dir, output_base + '_images.npy'), np.log1p(data))    
    
    if save_positions:
        # Save the (x,z) coordinates associated with each diffusion image
        positions=np.zeros([nb_frames,2])
        step=(position_x_end-position_x_start)/(nb_frames-1)
        for i in range(nb_frames):
            x=position_x_start+i*step
            positions[i][0]=x; positions[i][1]=position_z
     
        np.save(os.path.join(positions_dir, output_base + '_positions.npy'), positions)
        
    if save_basler_image:
        np.save(os.path.join(basler_dir, output_base + '_basler_image.npy'), basler_image)
    
    # Define the detector and azimuthal integrator
    detector = pyFAI.detectors.Detector(pixel1=pixel_size_x, pixel2=pixel_size_z)
    ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
    ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance in mm

    nbins = 1000
    unit_type = "q_A^-1"

    # Perform azimuthal integration with or without a mask for each image in the nxs
    intensities=np.zeros([nb_frames,3,nbins])
    qmap_array=np.zeros([nb_frames,300])
    chimap_array=np.zeros([nb_frames,360])
    imap_array=np.zeros([nb_frames,360,300])
    
    for i in range(nb_frames):
        if maskfile is None:
            qh, ih = ai.integrate1d(data[i], nbins, azimuth_range=(offset+180 - sector_angle, offset+180 + sector_angle), unit=unit_type)  
            qv, iv = ai.integrate1d(data[i], nbins, azimuth_range=(offset+270 - sector_angle, offset+270 + sector_angle), unit=unit_type)
            q_iso,i_iso=ai.integrate1d(data[i],nbins,unit=unit_type,normalization_factor=1)
            res2d = ai.integrate2d(data[i],300,360,unit=unit_type)
            intensities[i][0]=ih; intensities[i][1]=iv; intensities[i][2]=i_iso
        else:
            mask_img = fabio.open(maskfile)
            maskdata = mask_img.data
            qh, ih = ai.integrate1d(data[i], nbins, azimuth_range=(offset + 180 - sector_angle, offset + 180 + sector_angle), mask=maskdata, unit=unit_type)  
            qv, iv = ai.integrate1d(data[i], nbins, azimuth_range=(offset + 270 - sector_angle, offset + 270 + sector_angle), mask=maskdata, unit=unit_type)
            q_iso,i_iso=ai.integrate1d(data[i],nbins,unit=unit_type,normalization_factor=1,mask=maskdata)
            
            # Extract and save 2D map
            res2d = ai.integrate2d(data[i],300,360,unit=unit_type,mask=maskdata)
            Imap,qmap,chimap=res2d
            qmap=np.squeeze(qmap)
            chimap=np.squeeze(chimap)
            qmap_array[i] = qmap; chimap_array[i] = chimap ; imap_array[i,:,:] = Imap
            intensities[i][0]=ih; intensities[i][1]=iv; intensities[i][2]=i_iso
    q=[qh,qv,q_iso]  
    np.save(os.path.join(integration_dir,output_base+'_integrations.npy'),intensities)
    np.save(os.path.join(integration_dir,output_base+'_q.npy'),q)
    np.save(os.path.join(integration_dir,output_base+'_qmap.npy'),qmap_array)
    np.save(os.path.join(integration_dir,output_base+'_chimap.npy'),chimap_array)
    np.save(os.path.join(integration_dir,output_base+'_imap.npy'),imap_array)
    return

def extract_number(file_path: str) -> int:
    """
    Extract the number from the filename.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        int: Extracted number from the filename.
    """
    # Split the filename to get the number
    filename = file_path.split('\\')[-1]  # Get the last part of the path
    number = filename.split('_')[1]       # Get the number after 'lacroix_'
    return int(number)

def extract_h5_from_dir(directory: str,
        offset: float = 0,
        sector_angle: float = 10,
        maskfile: str = None,
        save_data: bool = True, 
        save_positions: bool = True, 
        save_basler_image: bool = True) -> str:
    """
    Extract and process .h5 files from a directory.
    
    Args:
        directory (str): Directory containing the .h5 files.
        offset (float): Offset angle for the azimuthal integration.
        sector_angle (float): Sector opening angle for the integration.
        maskfile (str, optional): Path to the mask file (optional).
        save_data (bool): Whether to save the normalized averaged images.
        save_positions (bool): Whether to save the x and z positions.
        save_basler_image (bool): Whether to save the Basler image.
    
    Returns:
        str: Output directory where the processed data is saved.
    """
    file_list = glob.glob(directory+'/*.h5')
    file_list = sorted(file_list, key=extract_number)
    print(file_list)
    
    with h5py.File(file_list[0], "r") as f:
        group = list(f.keys())[0]
        samplename = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')
    output_dir = os.path.dirname(file_list[0])+'/'+samplename

    for i, file in enumerate(file_list):
        nxs_swing_HandV_integration(file, output_dir, offset, sector_angle, maskfile, save_data, save_positions, save_basler_image)
        print('File %d out of %d files in the directory %s' % (i+1, len(file_list), directory))
        print(os.path.basename(file))
        clear_output(wait=True)
    
    return output_dir

def read_numpy_from_list(file_list: list) -> np.ndarray:
    """
    Read a list of numpy files and return a numpy array.
    
    Args:
        file_list (list): List of numpy files.
    
    Returns:
        np.ndarray: Combined numpy array from the list of files.
    """
    # Dimension of a single numpy file corresponding to a file
    size = np.shape(np.load(file_list[0]))
    shape = np.concatenate((np.array([len(file_list)]), size))
    array = np.zeros(shape)

    for i, file in enumerate(file_list):
        line = np.load(file)
        array[i] = line
    
    return array





# import h5py
# import os
# import numpy as np
# from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
# import pyFAI.detectors
# import fabio
# from matplotlib import pyplot as plt
# import glob
# from IPython.display import clear_output

# def nxs_swing_HandV_integration(
#         nxsfile: str, 
#         output_dir :str,
#         offset:float = 0,
#         sector_angle: float = 10,
#         maskfile: str = None,
#         save_data: bool = True, 
#         save_positions: bool = True, 
#         save_basler_image: bool = True     
#     ) -> None:
#     """
#     Effectue des intégrations horizontale et verticale sur un fichier .nxs de SWING.
    
#     Args:
#         nxsfile (str): Chemin du fichier .nxs.
#         sector_angle (float): Angle d'ouverture du secteur pour l'intégration.
#         save_data (bool): Sauvegarder ou non les images normalisées moyennées.
#         save_params (bool): Sauvegarder ou non les paramètres expérimentaux.
#         save_positions (bool): Sauvegarder ou non les positions x et z.
#         save_basler_image (bool): Sauvegarder ou non l'image Basler.
#         maskfile (Optional[str]): Chemin du fichier de masque (optionnel).
    
#     Returns:
#         None
#     """
#     output_base = os.path.basename(nxsfile).replace('.h5', '')
    

#     # Création des sous-dossiers si nécessaire
#     data_dir = os.path.join(output_dir, 'image')
#     integration_dir = os.path.join(output_dir, 'integration')
#     positions_dir = os.path.join(output_dir, 'positions')
#     basler_dir = os.path.join(output_dir, 'basler_image')

#     os.makedirs(data_dir, exist_ok=True)
#     os.makedirs(integration_dir, exist_ok=True)
#     os.makedirs(positions_dir, exist_ok=True)
#     os.makedirs(basler_dir, exist_ok=True)

#     with h5py.File(nxsfile, "r") as f:
#         group = list(f.keys())[0]
#         nb_frames = f[group + '/SWING/EIGER-4M/nb_frames'][0]
        
#         # Récupérer les données d'image
#         target = group + '/scan_data/eiger_image'
#         data = np.array(f[target])
        
#         # Récupérer les paramètres expérimentaux
#         target = group + '/SWING/EIGER-4M'
#         distance_m = f[target + '/distance'][0] / 1000  # Convertir en mètres
#         pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convertir en mètres
#         pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6  # Convertir en mètres
#         x_center = f[target + '/dir_beam_x'][0]
#         z_center = f[target + '/dir_beam_z'][0]
#         nb_frames=f[target+'/nb_frames'][0]
#         # Récupérer la longueur d'onde
#         target = group + '/SWING/i11-c-c03__op__mono'
#         wl = f[target + '/wavelength'][0]
        
#         # Récupérer les positions: notre fichier h5 continet toutes les images d'une même ligne
#         # dans le h5 on récupère x_start, x_end, et z constant
#         position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
#         position_x_end=f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
#         position_z = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]

#         # Récupérer l'image Basler
#         basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]

#     # Sauvegarde des données selon les options choisies
#     if save_data:
#         np.save(os.path.join(data_dir, output_base + '_images.npy'), np.log1p(data))    
    
#     if save_positions:
#         # on sauvegarde les coordonnées (x,z) associées à chaque image de diffusion
#         positions=np.zeros([nb_frames,2])
#         step=(position_x_end-position_x_start)/(nb_frames-1)
#         for i in range(nb_frames):
#             x=position_x_start+i*step
#             positions[i][0]=x; positions[i][1]=position_z
     
#         np.save(os.path.join(positions_dir, output_base + '_positions.npy'), positions)
        
    
#     if save_basler_image:
#         np.save(os.path.join(basler_dir, output_base + '_basler_image.npy'), basler_image)
    
#     # Définir le détecteur et l'intégrateur azimutal
#     detector = pyFAI.detectors.Detector(pixel1=pixel_size_x, pixel2=pixel_size_z)
#     ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
#     ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance en mm

#     nbins = 1000
#     unit_type = "q_A^-1"

#     # Effectuer l'intégration azimutale avec ou sans masque pour chaque image du nxs
    
#     intensities=np.zeros([nb_frames,3,nbins])
#     qmap_array=np.zeros([nb_frames,300])
#     chimap_array=np.zeros([nb_frames,360])
#     imap_array=np.zeros([nb_frames,360,300])
    
#     for i in range(nb_frames):
#         if maskfile is None:
#             qh, ih = ai.integrate1d(data[i], nbins, azimuth_range=(offset+180 - sector_angle, offset+180 + sector_angle), unit=unit_type)  
#             qv, iv = ai.integrate1d(data[i], nbins, azimuth_range=(offset+270 - sector_angle, offset+270 + sector_angle), unit=unit_type)
#             q_iso,i_iso=ai.integrate1d(data[i],nbins,unit=unit_type,normalization_factor=1)
#             res2d = ai.integrate2d(data[i],300,360,unit=unit_type)
#             intensities[i][0]=ih; intensities[i][1]=iv; intensities[i][2]=i_iso
#         else:
#             mask_img = fabio.open(maskfile)
#             maskdata = mask_img.data
#             qh, ih = ai.integrate1d(data[i], nbins, azimuth_range=(offset + 180 - sector_angle, offset + 180 + sector_angle), mask=maskdata, unit=unit_type)  
#             qv, iv = ai.integrate1d(data[i], nbins, azimuth_range=(offset + 270 - sector_angle, offset + 270 + sector_angle), mask=maskdata, unit=unit_type)
#             q_iso,i_iso=ai.integrate1d(data[i],nbins,unit=unit_type,normalization_factor=1,mask=maskdata)
            
#             # Extract and save 2D map
#             res2d = ai.integrate2d(data[i],300,360,unit=unit_type,mask=maskdata)
#             Imap,qmap,chimap=res2d
#             qmap=np.squeeze(qmap)
#             chimap=np.squeeze(chimap)
#             qmap_array[i] = qmap; chimap_array[i] = chimap ; imap_array[i,:,:] = Imap
#             intensities[i][0]=ih; intensities[i][1]=iv; intensities[i][2]=i_iso
#     q=[qh,qv,q_iso]  
#     np.save(os.path.join(integration_dir,output_base+'_integrations.npy'),intensities)
#     np.save(os.path.join(integration_dir,output_base+'_q.npy'),q)
#     np.save(os.path.join(integration_dir,output_base+'_qmap.npy'),qmap_array)
#     np.save(os.path.join(integration_dir,output_base+'_chimap.npy'),chimap_array)
#     np.save(os.path.join(integration_dir,output_base+'_imap.npy'),imap_array)
#     return

# def extract_number(file_path):
#     # Split the filename to get the number
#     filename = file_path.split('\\')[-1]  # Get the last part of the path
#     number = filename.split('_')[1]       # Get the number after 'lacroix_'
#     return int(number)

# def extract_h5_from_dir(directory,
#         offset: float = 0,
#         sector_angle: float = 10,
#         maskfile: str = None,
#         save_data: bool = True, 
#         save_positions: bool = True, 
#         save_basler_image: bool = True):
    
    
#     file_list=glob.glob(directory+'/*.h5')
#     file_list=sorted(file_list,key=extract_number)
#     print(file_list)
#     with h5py.File(file_list[0], "r") as f:
#         group = list(f.keys())[0]
#         samplename = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8)')
#     output_dir = os.path.dirname(file_list[0])+'/'+samplename

#     for i,file in enumerate(file_list):
#         nxs_swing_HandV_integration(file,output_dir,offset,sector_angle,maskfile,save_data, save_positions, save_basler_image)
#         print('File %d'%(i+1)+' out  of %d '%(len(file_list)) +' files in the directory %s'%directory)
#         print(os.path.basename(file))
#         clear_output(wait=True)
    
#     return output_dir

# def read_numpy_from_list(file_list):
#     """
#     Read  list of numpy files
#     return: numpy array
#     """
#     #dimension of a single numpy file corresponding to a file
#     size=np.shape(np.load(file_list[0]))
#     shape=np.concatenate((np.array([len(file_list)]),size))
#     array = np.zeros(shape)

#     for i, file in  enumerate(file_list):
#         line = np.load(file)
#         array[i] = line
#     return array


