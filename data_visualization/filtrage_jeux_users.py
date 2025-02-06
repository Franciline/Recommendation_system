import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Nombre maximum de
def drop_min_avis(df : pd.DataFrame, avis_min : int) -> pd.DataFrame : 
    """
    Retire garde les users ayant un min d'avis ok
    """
    df_copy = df.copy()
    # Côté users
    users = df.groupby('User id').size().reset_index(name='size')
    users_in = users.loc[ users["size"] < avis_min, ['User id']]

    df_copy = df_copy.drop(df_copy[np.isin(df_copy["User id"], users_in.to_numpy()) ].index)

    # Côté jeux
    jeux = df.groupby('Game id').size().reset_index(name='size')
    jeux_in = jeux.loc[ jeux["size"] < avis_min,['Game id']]

    df_copy = df_copy.drop(df_copy[np.isin(df_copy["Game id"],jeux_in.to_numpy()) ].index)
    return df_copy 


def get_nb_users_avis(df : pd.DataFrame, depedency = False) -> np.ndarray:
    """ Retourne un array de 1 à max(nb_avis/jeux, nb_avis/user
    
    Parameters:
    --------
        df : le dataframe
    
    Returns :
    -------
    """ 
    if(not(depedency)):
        # Création de l'axe des X
        users = df.groupby("User id").size().to_frame('size')
        jeux = (df.groupby("Game id")).size().to_frame('size')
        maxi = max(np.max(jeux["size"]), np.max(users["size"]))
    
        
        x_data = np.arange(start=25, stop=100, step=1)

        # Création de l'axe des Y
        
        vect_jeux = np.vectorize(lambda x : len(jeux[jeux["size"] >= x]))
        vect_users = np.vectorize(lambda x : len(users[users["size"] >= x]))
        y_data_jeux = vect_jeux(x_data)
        y_data_users = vect_users(x_data)

    else:
        x_data = np.arange(start=25, stop=1000, step=1)
        vect_jeux = np.vectorize(lambda x : (len(drop_min_avis(df,x).groupby("Game id"))))
        vect_users = np.vectorize(lambda x : (len(drop_min_avis(df,x).groupby("User id"))))

        y_data_jeux = vect_jeux(x_data)
        y_data_users = vect_users(x_data)

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

get_nb_users_avis(avis_clean, depedency=True)