#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:18:23 2023

@author: alexsalceschool
"""

import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import boxscorehustlev2
from nba_api.stats.endpoints import boxscoreplayertrackv3
from nba_api.stats.library.parameters import SeasonAll
import sys


data=[]
df=pd.DataFrame(data)
prefix = '00'
i = 0
def percentage(part, whole):
  return 100 * float(part)/float(whole)

#initialize player dataset
nba_players = pd.read_csv('nbaplayers.csv')
nba_players_allseasons = pd.read_csv('nbaplayers_allseasonsdata.csv')
badgames = pd.read_csv('badgames.csv')
badgames['Game_ID'] = badgames['0']
badgames = pd.DataFrame(badgames['Game_ID'] )
nba_games = pd.DataFrame(nba_players_allseasons['Game_ID'])
nba_games = pd.DataFrame(nba_games).drop_duplicates()


#exception handling, bad games data

exceptions = 0
# badgames = []
nba_games = pd.concat([nba_games, badgames]).drop_duplicates(keep=False)


#final number of games to look thru
num_games = len(nba_games.index)

#get all advanced stats for active player games
for GAME_ID in nba_games['Game_ID']:
    try:
        #counter and status
        i=i+1
        gameid = prefix + str(GAME_ID)
        percentdone = str(round(percentage(i, num_games),3)) + '% complete'
        print("Retrieving " + gameid + " data..." + percentdone + ", " + str(exceptions) + " games have no data")
        # print("Retrieving " + gameid + " data..." + percentdone, end='\r')
        #build df
        player_playertrack = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=gameid).player_stats.data
        player_playertrack_df = pd.DataFrame(player_playertrack['data'])
        player_playertrack_df = player_playertrack_df.set_axis(player_playertrack['headers'],axis=1)
        df = pd.concat([df,player_playertrack_df], axis=0)
    except KeyboardInterrupt:
        print("You pressed ctrl c...")
        break
    except Exception as e: # Any other exception
      # print(str(e)) # Displays the exception without raising it
      exceptions = exceptions + 1
      badgames.append(GAME_ID) 
      pass

#advanced data to CSV win UNCOMMENT ME AFTER RUNNING
df.to_csv(path_or_buf='C:/Users/Alex Salce/OneDrive/Desktop/School/Courses/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/player_playertrack.csv')

#hustle data to CSV mac
# df.to_csv(path_or_buf='/Users/alexsalceschool/Desktop/School/Statistics and Data Science NDP - UArizona/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbagames_hustledata.csv')
   




# =============================================================================
# #get all advanced stats for active player games
# for GAME_ID in nba_games['Game_ID']:
#     try:
#         #counter and status
#         i=i+1
#         gameid = prefix + str(GAME_ID)
#         percentdone = str(round(percentage(i, num_games),3)) + '% complete'
#         print("Retrieving " + gameid + " data..." + percentdone + ", " + str(exceptions) + " games have no data")
#         # print("Retrieving " + gameid + " data..." + percentdone, end='\r')
#         #build df
#         player_advancedstats = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=gameid).player_stats.data
#         player_advancedstats_df = pd.DataFrame(player_advancedstats['data'])
#         player_advancedstats_df = player_advancedstats_df.set_axis(player_advancedstats['headers'],axis=1)
#         df = pd.concat([df,player_advancedstats_df], axis=0)
#     except KeyboardInterrupt:
#         print("You pressed ctrl c...")
#         break
#     except Exception as e: # Any other exception
#       # print(str(e)) # Displays the exception without raising it
#       exceptions = exceptions + 1
#       badgames.append(GAME_ID) 
#       pass
# 
# #advanced data to CSV win UNCOMMENT ME AFTER RUNNING
# df.to_csv(path_or_buf='C:/Users/Alex Salce/OneDrive/Desktop/School/Courses/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbagames_player_advancedstats.csv')
# 
# #hustle data to CSV mac
# # df.to_csv(path_or_buf='/Users/alexsalceschool/Desktop/School/Statistics and Data Science NDP - UArizona/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbagames_hustledata.csv')
#        
# =============================================================================


# =============================================================================
# #get all hustle stats for active player games
# for GAME_ID in nba_games['Game_ID']:
#     try:
#         #counter and status
#         i=i+1
#         gameid = prefix + str(GAME_ID)
#         percentdone = str(round(percentage(i, num_games),3)) + '% complete'
#         print("Retrieving " + gameid + " data..." + percentdone + ", " + str(exceptions) + " games have no data")
#         # print("Retrieving " + gameid + " data..." + percentdone, end='\r')
#         #build df
#         player_hustle = boxscorehustlev2.BoxScoreHustleV2(game_id=gameid).player_stats.data
#         player_hustle_df = pd.DataFrame(player_hustle['data'])
#         player_hustle_df = player_hustle_df.set_axis(player_hustle['headers'],axis=1)
#         df = pd.concat([df,player_hustle_df], axis=0)
#     except KeyboardInterrupt:
#       print("You pressed ctrl c...")
#       break
#     except Exception as e: # Any other exception
#       # print(str(e)) # Displays the exception without raising it
#       exceptions = exceptions + 1
#       badgames.append(GAME_ID) 
#       pass
# 
# #hustle data to CSV win
# df.to_csv(path_or_buf='C:/Users/Alex Salce/OneDrive/Desktop/School/Courses/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbagames_hustledata.csv')
#
# #hustle data to CSV mac
# df.to_csv(path_or_buf='/Users/alexsalceschool/Desktop/School/Statistics and Data Science NDP - UArizona/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbagames_hustledata.csv')
#        
# =============================================================================


# =============================================================================
# 
# #get all players all season stats
# for PERSON_ID in nba_players['PERSON_ID']:
#     print("Retrieving " + nba_players[nba_players['PERSON_ID']==PERSON_ID]['FIRST_NAME'].values + " " + nba_players[nba_players['PERSON_ID']==PERSON_ID]['LAST_NAME'].values + " data...")
#     player_gamelog = playergamelog.PlayerGameLog(player_id = PERSON_ID, season=SeasonAll.all)
#     df_new = player_gamelog.get_data_frames()[0]
#     df = pd.concat([df,df_new], axis=0)
# 
# df.to_csv(path_or_buf='/Users/alexsalceschool/Desktop/School/Statistics and Data Science NDP - UArizona/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/nbaplayers_allseasonsdata.csv')
# 
# =============================================================================



badgames_df = pd.DataFrame(badgames)
# windirectory
# badgames_df.to_csv(path_or_buf='badgames_df.to_csv(path_or_buf='/Users/alexsalceschool/Desktop/School/Statistics and Data Science NDP - UArizona/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/badgames.csv')
#
# macdirectory
# badgames_df.to_csv(path_or_buf='/Users/alexsalceschool/Desktop/School/Statistics and Data Science NDP - UArizona/2023 FALL SDS MS/MATH574M STATISTICAL MACHINE LEARNING/Final Project/data/nba_api/badgames.csv')

