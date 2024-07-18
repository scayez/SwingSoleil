import glob
import os
from IPython.display import clear_output
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def load_numpy_files_with_names(folder):
    """Charge tous les fichiers numpy d'un dossier spécifique avec leurs noms."""
    file_list = glob.glob(folder + '/*.np*')
    loaded_files = [(np.load(file), os.path.basename(file)) for file in file_list]
    return loaded_files

def extract_integration_data(integration_files):
    """Extrait les données d'intégration à partir d'une liste de fichiers .npz."""
    qh_list, ih_list, qv_list, iv_list = [], [], [], []
    for file, name in integration_files:
        qh_list.append(file['qh'])
        ih_list.append(file['ih'])
        qv_list.append(file['qv'])
        iv_list.append(file['iv'])
    return qh_list, ih_list, qv_list, iv_list

def load_positions(folder):
    """Charge les fichiers de positions et retourne une liste de tuples (x, z)."""
    position_x_files = glob.glob(os.path.join(folder, '*_position_x.npy'))
    position_z_files = glob.glob(os.path.join(folder, '*_position_z.npy'))
    # position_y_files = glob.glob(os.path.join(folder, '*_position_y.npy'))  # Ajouter cette ligne pour les fichiers y

    positions = []
    max_x = -np.inf
    max_z = -np.inf
    min_x = np.inf
    min_z = np.inf

    for px_file, pz_file in zip(position_x_files, position_z_files):
        x = np.load(px_file)
        z = np.load(pz_file)
        # y = np.load(py_file)  # Décommenter cette ligne pour charger y
        
        positions.append((x, z))
        # positions.append((x, y, z))  # Utiliser cette ligne pour ajouter y aux positions

        max_x = max(max_x, np.max(x))
        max_z = max(max_z, np.max(z))
        min_x = min(min_x, np.min(x))
        min_z = min(min_z, np.min(z))
    
    return positions, max_x, max_z, min_x, min_z

def visualize_images_and_integration_animation(images_with_names, qh_list, ih_list, qv_list, iv_list, basler_images_with_names, positions, max_x, max_z, min_x, min_z):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot (0, 0): Image Eiger
    im = axs[0, 0].imshow(images_with_names[0][0], cmap='viridis')
    axs[0, 0].axis('off')

    # Subplot (0, 1): Courbes d'intégration
    qh, ih, qv, iv = qh_list[0], ih_list[0], qv_list[0], iv_list[0]
    line1, = axs[0, 1].loglog(qh, ih, label='horizontal')
    line2, = axs[0, 1].loglog(qv, iv, label='vertical')
    axs[0, 1].set_ylabel('I (a.u.)')
    axs[0, 1].set_xlabel('q (1/nm)')
    axs[0, 1].legend()

    # Subplot (1, 0): Image Basler
    im_basler = axs[1, 0].imshow(basler_images_with_names[0][0], cmap='gray')
    axs[1, 0].axis('off')

    # Subplot (1, 1): Coordonnées (x, z) avec point rouge
    point, = axs[1, 1].plot(positions[0][0], positions[0][1], 'ro', markersize=10)
    axs[1, 1].set_xlim(min_x, max_x)
    axs[1, 1].set_ylim(min_z, max_z)
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('z')

    # Positionner le titre au-dessus de la figure et ajuster les marges
    fig.subplots_adjust(top=0.92, bottom=0.08)

    fig.suptitle(f"File: {images_with_names[0][1]}", fontsize=12)

    def update(frame):
        im.set_data(images_with_names[frame][0])
        line1.set_data(qh_list[frame], ih_list[frame])
        line2.set_data(qv_list[frame], iv_list[frame])
        fig.suptitle(f"File: {images_with_names[frame][1]}", fontsize=12)
        im_basler.set_data(basler_images_with_names[frame][0])
        point.set_data(positions[frame][0], positions[frame][1])
        # point.set_data(positions[frame][0], positions[frame][2])  # Ajouter y pour le point (x, y, z)
        print(f"File {frame} on {len(images_with_names)} : {images_with_names[frame][1]}")
        clear_output(wait=True)
        return im, line1, line2, im_basler, point

    
    ani = FuncAnimation(fig, update, frames=len(images_with_names), interval=100, blit=True)
    plt.close()  # Fermer la figure après avoir créé l'animation

    return ani.to_jshtml()


def visualize_images_and_integration_animation_no_display(images_with_names, qh_list, ih_list, qv_list, iv_list, basler_images_with_names, positions, max_x, max_z, min_x, min_z, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot (0, 0): Image Eiger
    im = axs[0, 0].imshow(images_with_names[0][0], cmap='viridis')
    axs[0, 0].axis('off')

    # Subplot (0, 1): Courbes d'intégration
    qh, ih, qv, iv = qh_list[0], ih_list[0], qv_list[0], iv_list[0]
    line1, = axs[0, 1].loglog(qh, ih, label='horizontal')
    line2, = axs[0, 1].loglog(qv, iv, label='vertical')
    axs[0, 1].set_ylabel('I (a.u.)')
    axs[0, 1].set_xlabel('q (1/nm)')
    axs[0, 1].legend()

    # Subplot (1, 0): Image Basler
    im_basler = axs[1, 0].imshow(basler_images_with_names[0][0], cmap='gray')
    axs[1, 0].axis('off')

    # Subplot (1, 1): Coordonnées (x, z) avec point rouge
    point, = axs[1, 1].plot(positions[0][0], positions[0][1], 'ro', markersize=10)
    axs[1, 1].set_xlim(min_x, max_x)
    axs[1, 1].set_ylim(min_z, max_z)
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('z')

    # Positionner le titre au-dessus de la figure et ajuster les marges
    fig.subplots_adjust(top=0.92, bottom=0.08)

    fig.suptitle(f"File: {images_with_names[0][1]}", fontsize=12)

    def update(frame):
        im.set_data(images_with_names[frame][0])
        line1.set_data(qh_list[frame], ih_list[frame])
        line2.set_data(qv_list[frame], iv_list[frame])
        fig.suptitle(f"File: {images_with_names[frame][1]}", fontsize=12)
        im_basler.set_data(basler_images_with_names[frame][0])
        point.set_data(positions[frame][0], positions[frame][1])
        # point.set_data(positions[frame][0], positions[frame][2])  # Ajouter y pour le point (x, y, z)
        plt.close()  # Fermer la figure après chaque mise à jour pour éviter l'accumulation des figures en mémoire
        print(f"File {frame} on {len(images_with_names)} : {images_with_names[frame][1]}")
        clear_output(wait=True)
        return im, line1, line2, im_basler, point

    ani = FuncAnimation(fig, update, frames=len(images_with_names), interval=100, blit=True)

    # Enregistrer l'animation en tant que fichier vidéo MP4 ou GIF (comme vous l'avez implémenté)
    save_animation_as_video(ani, save_path)

    # Ne pas retourner ani pour l'affichage dans Jupyter, car nous n'utilisons pas ani.to_jshtml()

def save_animation_as_video(animation, filename):
    # Utiliser ffmpeg pour enregistrer l'animation en tant que fichier vidéo MP4
    animation.save(filename, writer='ffmpeg', fps=10)

