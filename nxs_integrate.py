import h5py
import os
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import fabio

def nxs_swing_HandV_integration(nxsfile, sector_angle, save_data_ok=True, save_params=True, save_positions=True, save_basler_image=True, maskfile=None):
    """
    Cette fonction prend un fichier .nxs de SWING comme entrée et effectue des intégrations horizontale et verticale.
    L'ouverture du secteur est définie par la variable sector_angle.
    Un fichier de masque peut être fourni en option.
    
    Les paramètres de sauvegarde indiquent quelles données doivent être sauvegardées.
    """
    output_base = os.path.basename(nxsfile).replace('.nxs', '')
    output_dir = os.path.dirname(nxsfile)

    # Création des sous-dossiers si nécessaire
    data_ok_dir = os.path.join(output_dir, 'image')
    params_dir = os.path.join(output_dir, 'params')
    integration_dir = os.path.join(output_dir, 'integration')
    positions_dir = os.path.join(output_dir, 'positions')
    basler_dir = os.path.join(output_dir, 'basler_image')

    os.makedirs(data_ok_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(integration_dir, exist_ok=True)
    os.makedirs(positions_dir, exist_ok=True)
    os.makedirs(basler_dir, exist_ok=True)

    with h5py.File(nxsfile, "r") as f:
        group = list(f.keys())[0]
        nb_frames = f[group + '/SWING/EIGER-4M/nb_frames'][0]
        
        # Récupérer les données d'image
        target = group + '/scan_data/eiger_image'
        data = np.array(f[target])
        
        # Normaliser les données par le courant de la source
        target = group + '/SWING/ans__ca__machinestatus/current'
        current = f[target][0]
        
        target = group + '/SWING/EIGER-4M'
        distance_m = f[target + '/distance'][0] / 1000
        pixel_size_x = f[target + '/pixel_size_x'][0] * 0.000001
        pixel_size_z = f[target + '/pixel_size_z'][0] * 0.000001
        x_center = f[target + '/dir_beam_x'][0]
        z_center = f[target + '/dir_beam_z'][0]
        
        # Récupérer la longueur d'onde
        target = group + '/SWING/i11-c-c03__op__mono'
        wl = f[target + '/wavelength'][0]
        
        # Récupérer les positions
        position_x = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
        position_z = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]
        
        # Récupérer l'image Basler
        basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]

    # Calculer la moyenne des images
    data_ok = np.mean(data, axis=0) / current
    data_ok = np.log1p(data_ok)
    
    # Sauvegarde des données selon les options choisies
    if save_data_ok:
        np.save(os.path.join(data_ok_dir, output_base + '_image.npy'), data_ok)
    
    if save_params:
        with open(os.path.join(params_dir, output_base + '_params.npz'), 'wb') as f_params:
            np.savez(f_params, 
                     distance_m=distance_m, pixel_size_x=pixel_size_x, pixel_size_z=pixel_size_z, 
                     x_center=x_center, z_center=z_center, wl=wl)
    
    if save_positions:
        np.save(os.path.join(positions_dir, output_base + '_position_x.npy'), position_x)
        np.save(os.path.join(positions_dir, output_base + '_position_z.npy'), position_z)
    
    if save_basler_image:
        np.save(os.path.join(basler_dir, output_base + '_basler_image.npy'), basler_image)
    
    # Définir le détecteur
    detector = pyFAI.detectors.Detector(pixel1=pixel_size_x, pixel2=pixel_size_z)
    ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
    ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)
    
    nbins = 1000
    unit_type = "q_nm^-1"

    # Effectuer l'intégration
    if maskfile is None:
        qh, ih = ai.integrate1d(data_ok, nbins, azimuth_range=(0 - sector_angle, 0 + sector_angle), unit=unit_type)  
        qv, iv = ai.integrate1d(data_ok, nbins, azimuth_range=(90 - sector_angle, 90 + sector_angle), unit=unit_type)
    else:
        mask_img = fabio.open(maskfile)
        maskdata = mask_img.data
        qh, ih = ai.integrate1d(data_ok, nbins, azimuth_range=(0 - sector_angle, 0 + sector_angle), mask=maskdata, unit=unit_type)  
        qv, iv = ai.integrate1d(data_ok, nbins, azimuth_range=(90 - sector_angle, 90 + sector_angle), mask=maskdata, unit=unit_type)
    
    # Sauvegarder les résultats d'intégration dans des fichiers numpy
    with open(os.path.join(integration_dir, output_base + '_integration.npz'), 'wb') as f_integration:
        np.savez(f_integration, qh=qh, ih=ih, qv=qv, iv=iv)
    
    return
