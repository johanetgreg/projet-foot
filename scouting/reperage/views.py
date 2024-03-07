# views.py

from django.shortcuts import render
import pandas as pd
from .models import Joueur
import glob as glob
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from unidecode import unidecode
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from scipy.spatial.distance import cityblock, cosine
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.

# Liste des chemins vers vos fichiers Excel
chemins_fichiers_excel = glob.glob('reperage/excel/*.xlsx')
print(chemins_fichiers_excel)

# Assuming df_temp1 and df_temp2 are your DataFrames
df_temp1 = pd.read_excel('reperage/excel/ligue1_2022.xlsx',index_col=0)
df_temp2 = pd.read_excel('reperage/excel/seria_2022.xlsx',index_col=0)
df_temp3 = pd.read_excel('reperage/excel/bundes_2022.xlsx',index_col=0)
df_temp4 = pd.read_excel('reperage/excel/liga_2022.xlsx',index_col=0)
df_temp5 = pd.read_excel('reperage/excel/pl_2022.xlsx',index_col=0)

print(df_temp1)

# Set the age limit (adjust the limit as needed)
age_limit = 27

# Concatenating along columns
df_final = pd.concat([df_temp1, df_temp2, df_temp3, df_temp4, df_temp5], axis=1)

# Transposer le DataFrame final
df_final = df_final.transpose()
# df_final = df_final[df_final['age'] <= age_limit]
df_final = df_final.reset_index(drop=True)

# Now df_final should have 25 rows and 1033 columns
print(df_final.shape)
print(df_final)
print(df_final.index)
print(df_final.columns)


def players_list(request):
    search_query = request.GET.get('search', '')
    
    # Convertir la requête de recherche en une chaîne ASCII sans accents
    search_query_ascii = unidecode(search_query)

    # Filter players based on the ASCII search query
    filtered_players = df_final[df_final['nom'].str.contains(search_query_ascii, case=False, na=False)]

    # Paginate the players
    paginator = Paginator(filtered_players, 50)  # Show 50 players per page

    page = request.GET.get('page')
    try:
        players = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        players = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        players = paginator.page(paginator.num_pages)

    return render(request, 'index.html', {'players': players, 'search_query': search_query})

def find_closest_player(request):
    if request.method == 'POST':
        selected_player_id = request.POST.get("player")
        weight_but = float(request.POST.get("weight_but", 1))
        weight_peno = float(request.POST.get("weight_peno", 1))
        weight_interceptions = float(request.POST.get("weight_interceptions", 1))
        weight_duels_gagnes = float(request.POST.get("weight_duels_gagnes", 1))
        weight_block = float(request.POST.get("weight_block", 1))
        weight_tacle = float(request.POST.get("weight_tacle", 1))
        weight_dribbles_reussis = float(request.POST.get("weight_dribbles_reussis", 1))
        weight_faute_obtenue = float(request.POST.get("weight_faute_obtenue", 1))
        weight_assists = float(request.POST.get("weight_assists", 1))
        weight_passes = float(request.POST.get("weight_passes", 1))
        weight_passekey = float(request.POST.get("weight_passekey", 1))
        weight_mtitu = float(request.POST.get("weight_mtitu", 1))
        # selected_criteria = request.POST.getlist("selected_criteria")
        
        print(f"Selected player ID: {selected_player_id}")

        if selected_player_id:

            # Convert the selected_player_id to an integer
            player_index = int(selected_player_id)

            # Use iloc to get the player by positional index
            player = df_final.iloc[player_index]

            # Define criteria based on player position
            position_criteria = {
                'Attacker': ['but','peno','interceptions', 'duels-gagnes', 'block', 'tacle','dribbles-reussis','faute-obtenue','assists','passes','passekey','mtitu'],
                'Midfielder': ['but','peno','interceptions', 'duels-gagnes', 'block', 'tacle','dribbles-reussis','faute-obtenue','assists','passes','passekey','mtitu'],
                'Defender': ['but','peno','interceptions', 'duels-gagnes', 'block', 'tacle','dribbles-reussis','faute-obtenue','assists','passes','passekey','mtitu'],
            }

            # Define weights for each criterion
            weights = {
                'but': weight_but,
                'peno': weight_peno,
                'age' : 1.0,
                'dribbles-reussis': 1.2,
                'faute-obtenue': 0.9,
                'assists': 1.1,
                'passes': 0.7,
                'passekey': 0.9,
                'duels-gagnes': 0.6,
                'interceptions': 1.0,
                'block': 0.8,
                'tacle': 0.7,
                'mtitu': 0.4,
            }

            # Get the player's position
            player_position = player['poste']
            print(player_position)

            if player_position in position_criteria:

                # Use position-specific criteria
                columns_for_distance = position_criteria[player_position]
                numeric_columns = df_final[columns_for_distance].apply(pd.to_numeric, errors='coerce').drop(index=player_index)
                print(numeric_columns)

                # Apply weights to the distance metrics
                weights_for_distance_metrics = {
                    'cara': {
                    'but': weight_but,
                    'peno': weight_peno,
                    'interceptions': weight_interceptions,
                    'duels-gagnes': weight_duels_gagnes,
                    'block': weight_block,
                    'tacle': weight_tacle,
                    'dribbles-reussis': weight_dribbles_reussis,
                    'faute-obtenue': weight_faute_obtenue,
                    'assists': weight_assists,
                    'passes': weight_passes,
                    'passekey': weight_passekey,
                    'mtitu': weight_mtitu,
                    },
                }

                # Calculate distances using different distance metrics
                distances_euclidean = numeric_columns.apply(
                lambda row: np.sqrt(np.sum(((row - player[columns_for_distance]) ** 2)
                                            * np.array([weights_for_distance_metrics['cara'][col] for col in columns_for_distance]))),axis=1)


                distances_manhattan = numeric_columns.apply(
                lambda row: np.sum(np.abs(row - player[columns_for_distance]) 
                                   * np.array([weights_for_distance_metrics['cara'][col] for col in columns_for_distance])),
                axis=1)
                
                
                # distances_euclidean = numeric_columns.apply(lambda row: euclidean(row, player[columns_for_distance]), axis=1)
                # distances_manhattan = numeric_columns.apply(lambda row: cityblock(row, player[columns_for_distance]), axis=1)
                # distances_correlation = numeric_columns.corrwith(player[columns_for_distance])

                print(distances_euclidean)
                print(distances_manhattan)
                
                # Normalizing data for cosine similarity
                scaler = MinMaxScaler(feature_range=(-1, 1))
                numeric_columns_normalized = scaler.fit_transform(numeric_columns)
                print(numeric_columns_normalized)

                print("After scaling:")
                print(numeric_columns_normalized.min(axis=0))
                print(numeric_columns_normalized.max(axis=0))

                player_normalized = scaler.transform(player[columns_for_distance].values.reshape(1, -1))
                print(player_normalized)
                distances_cosine = cosine_similarity(numeric_columns_normalized, player_normalized)

                # Find the indices of the players sorted by distance for each metric
                sorted_indices_euclidean = distances_euclidean.sort_values().index[:20]
                sorted_indices_manhattan = distances_manhattan.sort_values().index[:20]
                # sorted_indices_correlation = distances_correlation.abs().sort_values(ascending=False).index[:5]
                sorted_indices_cosine = distances_cosine.flatten().argsort()[::-1][:20]

                # Print the distances of the first 5 players
                # print("Distances of the first 20 players:")
                # for i in range(20):
                #     print(f"Player {sorted_indices_cosine[i]}: {distances_cosine[sorted_indices_cosine[i]]}")

                # Select the 5 players with the smallest distances for each metric
                closest_players_euclidean = df_final.iloc[sorted_indices_euclidean]
                closest_players_manhattan = df_final.iloc[sorted_indices_manhattan]
                # closest_players_correlation = df_final.iloc[sorted_indices_correlation]
                closest_players_cosine = df_final.iloc[sorted_indices_cosine]

                # Filter the closest players to have the same position as the selected player
                # closest_players_same_position_euclidean = closest_players_euclidean[closest_players_euclidean['poste'] == player_position].head(5)
                # closest_players_same_position_manhattan = closest_players_manhattan[closest_players_manhattan['poste'] == player_position].head(5)
                # closest_players_same_position_cosine = closest_players_cosine[closest_players_cosine['poste'] == player_position].head(5)

                # Filter the closest players to have the same position as the selected player
                closest_players_same_position_euclidean = closest_players_euclidean['nom'].head(5)
                closest_players_same_position_manhattan = closest_players_manhattan['nom'].head(5)
                closest_players_same_position_cosine = closest_players_cosine['nom'].head(5)

                print(weight_but)
                print(weight_peno)

                return render(request, 'closest_player_result.html', {
                    'player': player,
                    'closest_players_euclidean': closest_players_same_position_euclidean,
                    'closest_players_manhattan': closest_players_same_position_manhattan,
                    # 'closest_players_correlation': closest_players_correlation,
                    'closest_players_cosine': closest_players_same_position_cosine,
                })
            else:
                return render(request, 'find_closest_player.html', {'players': df_final, 'error_message': 'No numeric columns available for distance calculation'})

    # If it's a GET request or no player is selected, pass the list of players to select from
    return render(request, 'find_closest_player.html', {'players': df_final})
