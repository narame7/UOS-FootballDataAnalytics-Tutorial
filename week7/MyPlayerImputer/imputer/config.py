import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(base_path)

sys.path.append(base_path)
team_colors = {
    'H': '#0000BF', # 홈팀 색
    'A': '#FF0000'  # 어웨이 색
}
# import socceraction.spadl.config as _spadl

frame_rate = 25
max_agents = 40

field_length = 105 # _spadl.field_length
field_width = 68 # _spadl.field_width
max_speed = 12

# BEPRO config.py
actiontypes: list[str] = [
    "Pass",
    "Cross",

    "Throw-In",
    "Pass_Freekick",
    "Pass_Corner",
    "Shot_Freekick",
    "Shot_Corner",
    "Penalty Kicks",
    "Goal Kick",

    "Take-On",
    "Carry",
    "Foul",
    "Tackle",
    "Interception",
    "Shot",
    "Aerial Control",
    "Defensive Line Supports",
    "Save",
    "Block",
    "Clearance",
    "bad_touch",
    "Receive",
    "Recovery",
    "Duel",
    "Offside",

    "Foul Won",
    "non_action"
]
bodyparts: list[str] = [
    "foot", 
    "other"
]
results: list[str] = [
    "fail",
    "success",
    #"offside",
    "owngoal",
    "yellow_card",
    "red_card",
]
on_ball_actions = [
    "Carry",
    "Clearance",
    "Cross",
    # "Mistake",
    "bad_touch",
    "Take-On",
    "Pass",
    "Cross",
    "Carry",
    # "Own Goal",
    "Set-piece",
    "Shot",
    "Take-On",
    "Goal Kick"
]
off_ball_actions = [
    "Aerial Control",
    "Block",
    "Defensive Line Supports",
    "Duel",
    "Foul",
    "Save",
    "Interception",
    # "Mistake",
    "bad_touch",
    "Offside",
    "Receive",
    "Recovery",
    "Tackle"
]
exception_actions = [
    "Other"
]
EVENT_LABEL_MAPPING = {
    "None": 0,
    "Pass": 1,
    "Cross": 2,
    "Throw-In": 3,
    "Pass_Freekick": 4,
    "Pass_Corner": 5,
    "Shot_Freekick": 6,
    "Shot_Corner": 7,
    "Penalty Kicks": 8,
    "Goal Kick": 9,
    "Take-On": 10,
    "Carry": 11,
    "Foul": 12,
    "Tackle": 13,
    "Interception": 14,
    "Shot": 15,
    "Aerial Control": 16,
    "Defensive Line Supports": 17,
    "Save": 18,
    "Block": 19,
    "Clearance": 20,
    "bad_touch": 21,
    "Receive": 22,
    "Recovery": 23,
    "Duel": 24,
    "Offside": 25,
    # "Foul Won": 26,
    # "non_action": 27
}

PLAYER_ROLE_MAPPING = {
    "None": 0,
    "CAM": 1,
    "CB": 2,
    "CDM": 3,
    "CF": 4,
    "CM": 5,
    "GK": 6,
    "LB": 7,
    "LM": 8,
    "LW": 9,
    "LWB": 10,
    "RB": 11,
    "RM": 12,
    "RW": 13,
    "RWB": 14,
}

# #DFL 버전 config.py
# on_ball_actions = [
#     "Clearance",
#     "Save",
#     "Set-piece",
#     "Shot",
#     "Take-On",
#     "Pass",
#     "Cross",
#     "Carry"
# ]

# off_ball_actions = [
#     "Block",
#     "Interception",
#     "Receive",
#     "Foul",
#     "Duel"
# ]

# exception_actions = [
#     "Other"
# ]

# on_ball_actions = [
#     "KickOff_Play_Pass",
#     "Play_Pass",
#     "ThrowIn_Play_Pass",
#     "Play_Cross",
#     "FreeKick_Play_Pass",
#     "GoalKick_Play_Pass",
#     "CornerKick_Play_Cross",
#     "ShotAtGoal_ShotWide",
#     "ShotAtGoal_SavedShot",
#     "ShotAtGoal_BlockedShot",
#     "ThrowIn_Play_Cross",
#     "ShotAtGoal_SuccessfulShot",
#     "ShotAtGoal_ShotWoodWork",
#     "FreeKick_Play_Cross",
#     "CornerKick_Play_Pass",
#     "SpectacularPlay",
#     "OtherBallAction",
#     "Penalty_ShotAtGoal_SuccessfulShot"
# ]

# off_ball_actions = [
#     "TacklingGame",
#     "BallClaiming",
#     "Foul",
#     "PlayerNotSentOff",
#     "Run",
#     "Caution",
#     "OutSubstitution",
#     "ChanceWithoutShot",
#     "Offside",
#     "BallDeflection",
#     "GoalDisallowed",
#     "OtherPlayerAction",
#     "PossessionLossBeforeGoal",
#     "SitterPrevented"
# ]

# exception_actions = [
#     "Delete",
#     "RefereeBall",
#     "FinalWhistle",
#     "VideoAssistantAction",
#     "PenaltyNotAwarded",
# ]
# EVENT_LABEL_MAPPING = {
#     "None": 0,
#     'Play_Pass': 1,
#     'OtherBallAction': 2,
#     'TacklingGame': 3,
#     'ThrowIn_Play_Pass': 4,
#     'BallClaiming': 5,
#     'Foul': 6,
#     'FreeKick_Play_Pass': 7,
#     'Play_Cross': 8,
#     'OutSubstitution': 9,
#     'GoalKick_Play_Pass': 10,
#     'ShotAtGoal_ShotWide': 11,
#     'CornerKick_Play_Cross': 12,
#     'ShotAtGoal_SavedShot': 13,
#     'ShotAtGoal_BlockedShot': 14,
#     'Caution': 15,
#     'KickOff_Play_Pass': 16,
#     'ThrowIn_Play_Cross': 17,
#     'ShotAtGoal_SuccessfulShot': 18,
#     'Offside': 19,
#     'Nutmeg': 20,
#     'FreeKick_Play_Cross': 21,
#     'SpectacularPlay': 22,
#     'ChanceWithoutShot': 23,
#     'Run': 24,
#     'PossessionLossBeforeGoal': 25,
#     'PlayerNotSentOff': 26,
#     'FairPlay': 27,
#     'BallDeflection': 28,
#     'CornerKick_Play_Pass': 29,
#     'ShotAtGoal_OtherShot': 30,
#     'ShotAtGoal_ShotWoodWork': 31,
#     'Penalty_ShotAtGoal_SuccessfulShot': 32,
#     'ThrowIn': 33,
#     'OtherPlayerAction': 34,
#     'SitterPrevented': 35,
#     'CautionTeamofficial': 36,
#     'FreeKick_ShotAtGoal_BlockedShot': 37,
#     'GoalDisallowed': 38,
#     'FreeKick_ShotAtGoal_SavedShot': 39,
#     'FreeKick_ShotAtGoal_ShotWide': 40,
#     'Delete': 41,
#     'RefereeBall': 42,
#     'FinalWhistle': 43,
#     'VideoAssistantAction': 44,
#     'PenaltyNotAwarded': 45
# }
# PLAYER_ROLE_MAPPING = {
#     'None': 0,
#     # Rechter Verteidiger <-> Right Back
#     'RV': 1,
#     # Linker Verteidiger <-> Left Back
#     'LV': 2,
#     # Innenverteidiger Rechts / Zentral / Links  <-> Center Back Right / Central / Left 
#     'IVR': 3,
#     'IVZ': 3,
#     'IVL': 3,
#     # Defensives Mittelfeld Rechts / Zentral / Links <-> Defensive Midfielder Right / Central / Left
#     'DRM': 4,
#     'DMR': 4,
#     'DML': 4,
#     'DLM': 4,
#     'DMZ': 5,
#     # Halb Links / Halb Rechts / Mittelfeld Zentral<-> Half Left / Half Right / Central Midfielder
#     'HL': 5,
#     'HR': 5,
#     'MZ': 5,
#     # Linkes Mittelfeld <-> Left Midfielder
#     'LM': 6,
#     # Rechtes Mittelfeld <-> Right Midfielder
#     'RM': 7,
#     # Stürmer Rechts / Zentral / Links  <->  Striker Right / Central / Left
#     'STR': 8,
#     'STZ': 8,
#     'STL': 8,
#     # Offensives Rechtes / Offensives Linkes /Zentrales Offensives  Mittelfeld  <-> Attacking Midfielder Right / Left
#     'ORM': 9,
#     'ZO': 9,
#     'OLM': 9, 
#     # Linkes Außenstürmer <-> Left Winger
#     'RA': 10,
#     # Rechtes Außenstürmer <-> Right Winger
#     'LA': 11,
#     # Torwart<->Goalkeeper    
#     'TW': 12, 
#     # Offensives Halb Links / Rechts <-> Attacking Half Left / Right
#     'OHL': 13,
#     'OHR': 13,
# }

# # feautre 활용
# spatial_features=[
#     'prevAgentX', 'prevAgentY', 
#     'nextAgentX', 'nextAgentY',
#     'eventX', 'eventY',
#     'avgAgentX', 'avgAgentY',
#     'coordinates', 'prevDeltaAngle',
#     'prevAvgX', 'prevAvgY',
#     'nextAvgX', 'nextAvgY',
#     'observeEventX','observeEventY'
# ]

# time_features=[
#     'prevAgentTime',
#     'nextAgentTime',
#     'velocity',
#     'elapsedTime',
#     'possesRatio',
#     'minutePlayed'
# ]

# # angle_features=[
# #     "startangletogoal",
# #     "endangletogoal"
# # ]

# velocity_labels=[
#     'velocity'
# ]

# coordinates_labels=[
#     'coordinates'
# ]
# categorical_features={
#     "agentObserved":2,
#     "agentRole":len(PLAYER_ROLE_MAPPING),
#     "agentSide":2,
#     "eventType":len(EVENT_LABEL_MAPPING),#45,
#     # "results": len(results),
#     # "bodyparts": len(bodyparts),
#     # "period": 2
#     # "goalDiff":9
# }
