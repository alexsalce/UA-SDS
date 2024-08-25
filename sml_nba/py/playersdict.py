# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:04:13 2023

@author: Alex Salce
"""

import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import commonplayerinfo

nba_players = players.get_active_players()

# i=0
# nba_players_info = {}
# for player in nba_players:
#     player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
    
#     nba_players_info[i]  = player_info.get_data_frames()
#     i = i+1
    
data=[]
df=pd.DataFrame(data)
for entry in nba_players_info:
    df_new = nba_players_info[entry][0]
    df = pd.concat([df,df_new], axis=0)

df.to_csv(path_or_buf='C:/Users/Alex Salce/Desktop/DS/NBA/nbaplayers.csv')
    
