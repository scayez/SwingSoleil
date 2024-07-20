import h5py
import os
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
import fabio
from typing import Optional

def nxs_swing_HandV_integration(
        nxsfile: str, 
        sector_angle: float, 
        save_data_ok: bool = True, 
        save_params: bool = True, 
        save_positions: bool = True, 
        save_basler_image: bool = True, 
        maskfile: Optional[str] = None
    ) -> None:
    """
    Effectue des intégrations horizontale et verticale sur un fichier .nxs de SWING.
    
    Args:
        nxsfile (str): Chemin du fichier .nxs.
        sector_angle (float): Angle d'ouverture du secteur pour l'intégration.
        save_data_ok (bool): Sauvegarder ou non les images normalisées moyennées.
        save_params (bool): Sauvegarder ou non les paramètres expérimentaux.
        save_positions (bool): Sauvegarder ou non les positions x et z.
        save_basler_image (bool): Sauvegarder ou non l'image Basler.
        maskfile (Optional[str]): Chemin du fichier de masque (optionnel).
    
    Returns:
        None
    """
    # Détermination des chemins de sauvegarde
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
        
        # Récupérer les paramètres expérimentaux
        target = group + '/SWING/EIGER-4M'
        distance_m = f[target + '/distance'][0] / 1000  # Convertir en mètres
        pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convertir en mètres
        pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6  # Convertir en mètres
        x_center = f[target + '/dir_beam_x'][0]
        z_center = f[target + '/dir_beam_z'][0]
        nb_frames=f[target+'/nb_frames'][0]
        # Récupérer la longueur d'onde
        target = group + '/SWING/i11-c-c03__op__mono'
        wl = f[target + '/wavelength'][0]
        
        # Récupérer les positions
        position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
        position_x_end=f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
        position_z = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]
        # position_y = f[group + '/SWING/i11-c-c08__ex__tab-mt_ty.4/position'][()]  # Récupérer la position y

        # Récupérer l'image Basler
        basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]

    # Calculer la moyenne des images et normaliser par le courant
    #data_ok = np.mean(data, axis=0) / current
    #data_ok = np.log1p(data_ok)  # Logarithme naturel pour la dynamique

    # Sauvegarde des données selon les options choisies
    if save_data_ok:
        np.save(os.path.join(data_ok_dir, output_base + '_images.npy'), np.log1p(data/current))
    
    if save_params:
        with open(os.path.join(params_dir, output_base + '_params.npz'), 'wb') as f_params:
            np.savez(f_params, 
                     distance_m=distance_m, pixel_size_x=pixel_size_x, pixel_size_z=pixel_size_z, 
                     x_center=x_center, z_center=z_center, wl=wl)
    
    if save_positions:
        np.save(os.path.join(positions_dir, output_base + '_position_x.npy'), np.linspace(position_x_start,position_x_end,num=nb_frames))
        np.save(os.path.join(positions_dir, output_base + '_position_z.npy'), position_z)
        # np.save(os.path.join(positions_dir, output_base + '_position_y.npy'), position_y)  # Sauvegarder la position y
    
    if save_basler_image:
        np.save(os.path.join(basler_dir, output_base + '_basler_image.npy'), basler_image)
    
    # Définir le détecteur et l'intégrateur azimutal
    detector = pyFAI.detectors.Detector(pixel1=pixel_size_x, pixel2=pixel_size_z)
    ai = AzimuthalIntegrator(dist=distance_m, detector=detector)
    ai.setFit2D(distance_m * 1000, x_center, z_center, wavelength=wl)  # Distance en mm

    nbins = 1000
    unit_type = "q_nm^-1"

    # Effectuer l'intégration azimutale avec ou sans masque pour chaque image du nxs
    i=np.zeros([2,nb_frames])
    for i in range(nb_frames):
        if maskfile is None:
            qh, ih = ai.integrate1d(data[i], nbins, azimuth_range=(0 - sector_angle, 0 + sector_angle), unit=unit_type)  
            qv, iv = ai.integrate1d(data[i], nbins, azimuth_range=(90 - sector_angle, 90 + sector_angle), unit=unit_type)
        else:
            mask_img = fabio.open(maskfile)
            maskdata = mask_img.data
            qh, ih = ai.integrate1d(data[i], nbins, azimuth_range=(0 - sector_angle, 0 + sector_angle), mask=maskdata, unit=unit_type)  
            qv, iv = ai.integrate1d(data[i], nbins, azimuth_range=(90 - sector_angle, 90 + sector_angle), mask=maskdata, unit=unit_type)
    
    # Sauvegarder les résultats d'intégration dans des fichiers numpy
    with open(os.path.join(integration_dir, output_base + '_integration.npz'), 'wb') as f_integration:
        np.savez(f_integration, qh=qh, ih=ih, qv=qv, iv=iv)
    
    return
