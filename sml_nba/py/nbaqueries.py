# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:48:01 2024

@author: Alex Salce
"""

# File: nba_player_stats.py

import os
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats
from nba_api.stats.endpoints import boxscoreadvancedv3, boxscoreplayertrackv3, boxscorehustlev2, playergamelog
from tqdm import tqdm

def get_active_players():
    """
    Fetch all active NBA players.

    Returns:
        list: A list of dictionaries containing active player data.
    """
    try:
        all_players = players.get_players()
        active_players = [player for player in all_players if player['is_active']]
        return active_players
    except Exception as e:
        print(f"Error fetching players data: {e}")
        return []

def fetch_player_stats(player_id):
    """
    Fetch various stats for a specific player.

    Args:
        player_id (int): The player's ID.

    Returns:
        dict: A dictionary containing the player's stats.
    """
    stats = {}
    try:
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
        stats['career'] = career_stats
    except Exception as e:
        print(f"Error fetching career stats for player ID {player_id}: {e}")
    
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id).get_data_frames()[0]
        stats['gamelog'] = gamelog
    except Exception as e:
        print(f"Error fetching game log for player ID {player_id}: {e}")

    return stats

def fetch_game_stats(game_id):
    """
    Fetch various stats for a specific game.

    Args:
        game_id (int): The unique game ID

    Returns:
        dict: A dictionary containing the game's stats.
    """
    stats = {}
    try:
        boxscore_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id).get_data_frames()[0]
        stats['boxscore_adv'] = boxscore_adv
    except Exception as e:
        print(f"Error fetching advanced boxscore for game ID {game_id}: {e}")

    try:
        boxscore_track = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id).get_data_frames()[0]
        stats['boxscore_track'] = boxscore_track
    except Exception as e:
        print(f"Error fetching player tracking boxscore for game ID {game_id}: {e}")
              
    try:
        boxscore_hustle = boxscorehustlev2.BoxScoreHustleV2(game_id=game_id).get_data_frames()[0]
        stats['boxscore_hustle'] = boxscore_hustle
    except Exception as e:
        print(f"Error fetching hustle boxscore for game ID {game_id}: {e}")

    return stats



def main():
    # Get list of all active players
    active_players = get_active_players()
    
    # Initialize lists to hold data
    career_data = []
    gamelog_data = []
    boxscore_adv_data = []
    boxscore_track_data = []
    boxscore_hustle_data = []

    # Fetch stats for each player
    for player in tqdm(active_players, desc="Fetching player stats"):
        player_id = player['id']
        player_name = player['full_name']
        stats = fetch_player_stats(player_id)
        
        if 'career' in stats:
            stats['career']['PlayerName'] = player_name
            career_data.append(stats['career'])
        if 'gamelog' in stats:
            stats['gamelog']['PlayerName'] = player_name
            gamelog_data.append(stats['gamelog'])


        time.sleep(1)  # To avoid hitting rate limits

    # Combine all data into one DataFrame
    combined_df = pd.concat(
        career_data + gamelog_data + boxscore_adv_data + boxscore_track_data + boxscore_hustle_data, 
        ignore_index=True, 
        sort=False
    )
    
    # Save to a CSV file
    output_file = "combined_nba_player_stats.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
