import numpy as np
import os
import glob

def extract_number(file_path: str) -> int:
    """
    Extract the number from the filename.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        int: Extracted number from the filename.
    """
    # Split the filename to get the number
    filename = file_path.split('/')[-1]  # Get the last part of the path
    number = filename.split('_')[1]       # Get the number after 'lacroix_'
    return int(number)

def convert_npy2ascii(data_folder):
    #data_folder='/home-local/ratel-ra/Documents/SAXS_data/SAXS_SWING/20231871/Rohan/rohan_2_s2b/'

    directories = [dir for dir in glob.glob(os.path.join(data_folder, '*/')) if os.path.isdir(dir)]
    print(len(directories))
    
    integrations_path=directories[0]+'integration/'
    
    file_list=glob.glob(integrations_path+'*_integrations.npy')
    file_list=sorted(file_list,key=extract_number)
    q_file_list=glob.glob(integrations_path+'*_q.npy')
    q_file_list=sorted(q_file_list,key=extract_number)
    nbre_lignes=len(file_list)
    nbre_colonnes=np.shape(np.load(file_list[0]))[0]
    print('nbre colonnes',nbre_colonnes)
    print('nbre_lignes',nbre_lignes)

    output_folder=directories[0]+'ascii_file'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        pass

    for k in range(nbre_lignes):
        directory,filename=os.path.split(file_list[k])
        name, extension = os.path.splitext(filename)

        
        ifile=file_list[k]
        qfile=q_file_list[k]


        i_array=np.load(ifile)
        q_array=np.load(qfile)
        print('q_array',np.shape(q_array))
        print('i_array',np.shape(i_array))


        for j in range(nbre_colonnes):
            outputname=output_folder+'/'+name+'_line'+str(k)+'_column'+str(j)+'.txt'
            print(outputname)
            if np.shape(q_array)[0]==nbre_colonnes:
                q=q_array[j]
            if np.shape(q_array)[0]==3:
                q=q_array[2]
            i=i_array[j,2,:] #isototropic averaging
            data=np.column_stack((q,i))
            np.savetxt(outputname,data)

data_folder='/home-local/ratel-ra/Bureau/Lien_vers_NCO/Manips/DATA_SAXS/SWING_µSAXS/20231871/2024/Run3/capillaires/Rohan/cpag_serie/cpag200a_s2'

convert_npy2ascii(data_folder)
