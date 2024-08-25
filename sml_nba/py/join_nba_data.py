# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:43:41 2023

@author: Alex Salce
"""

import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import boxscorehustlev2
from nba_api.stats.library.parameters import SeasonAll
import sys

# =============================================================================
# data=[]
# df=pd.DataFrame(data)
# =============================================================================


# =============================================================================
# initialize data
# =============================================================================

#initialize player and games datasets
nba_players = pd.read_csv('nbaplayers.csv')
nba_players_allseasons = pd.read_csv('nbaplayers_allseasonsdata.csv')
badgames = pd.read_csv('badgames.csv')
badgames['Game_ID'] = badgames['0']
badgames = pd.DataFrame(badgames['Game_ID'] )
nba_games = pd.DataFrame(nba_players_allseasons['Game_ID'])
nba_games = pd.DataFrame(nba_games).drop_duplicates()
# nba_players_allseasons = pd.DataFrame(nba_games).drop_duplicates()


#initialize stats datasets
nbagames_player_advancedstats = pd.read_csv('nbagames_player_advancedstats.csv')
nbagames_hustledata = pd.read_csv('nbagames_hustledata.csv')
nbagames_playertracking = pd.read_csv('player_playertrack.csv')


# =============================================================================
# join datasets
# =============================================================================



join_hustle_advanced = pd.merge(nbagames_hustledata, 
                                nbagames_player_advancedstats,  
                                how='left', left_on=['Player_ID','Game_ID'], 
                                right_on = ['personId','gameId'])

join_allseasons = pd.merge(nba_players_allseasons, join_hustle_advanced,
                               how='left', left_on=['Player_ID','Game_ID'],
                               right_on = ['personId','gameId'])

join_alldata = pd.merge(join_allseasons, nbagames_playertracking,
                               how='left', left_on=['Player_ID_x','Game_ID_x'],
                               right_on = ['personId','gameId'])

join_allplayer = pd.merge(nba_players, join_alldata,
                               how='left', left_on=['PERSON_ID'],
                               right_on = ['Player_ID_x'])


df = join_allplayer.dropna(subset=['minutes_y'])

#all data to CSV win UNCOMMENT ME AFTER RUNNING
df.to_csv(path_or_buf='C:/Users/Alex Salce/OneDrive/Desktop/School/Courses/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbaplayerdata.csv')


        
