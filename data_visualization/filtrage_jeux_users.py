import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Nombre maximum de

def get_nb_users_avis(df : pd.DataFrame, depedency = False) -> np.ndarray:
    """ Retourne un array de 1 à max(nb_avis/jeux, nb_avis/users)
    
    Parameters:
    --------
        df : le dataframe
    
    Returns :
    -------
    """ 
    if(not(depedency)):
        # Création de l'axe des X
        jeux = df.groupby("User id").size().to_frame('size')
        users = (df.groupby("Game id")).size().to_frame('size')
        maxi = max(np.max(jeux["size"]), np.max(users["size"]))
    
        
        x_data = np.arange(start=25, stop=100, step=1)

        # Création de l'axe des Y
        
        vect_jeux = np.vectorize(lambda x : len(jeux[jeux["size"] >= x]))
        vect_users = np.vectorize(lambda x : len(users[users["size"] >= x]))
        y_data_jeux = vect_jeux(x_data)
        y_data_users = vect_users(x_data)

    else:
        

    # Création du graphique
    print("Affichage")
    plt.plot(x_data,y_data_jeux, label ="jeux")
    plt.plot(x_data,y_data_users, label = "users")
    plt.legend()
    plt.title("Nombres d'entrées valides en fonction du nombre min d'avis par users/joueurs")
    plt.show()
    

    return None
folder = "../database_cleaned"
avis_clean = pd.read_csv(f'{folder}/avis_clean.csv', header=None, names=["Game id", "User id", "Game name UI", "Username", "Datetime", "Rating", "Comment title", "Comment body"]).drop_duplicates()

get_nb_users_avis(avis_clean)