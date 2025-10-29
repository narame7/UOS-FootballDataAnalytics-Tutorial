import argparse
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


import iso8601
import numpy as np
import pandas as pd
from lxml import etree
from scipy.signal import savgol_filter
from tqdm import tqdm

from floodlight.core.code import Code
from floodlight.core.events import Events
from floodlight.core.pitch import Pitch
from floodlight.core.teamsheet import Teamsheet
from floodlight.core.xy import XY
from floodlight.io.dfl import (
    _create_periods_from_dat,
    _get_event_description,
    _get_event_outcome,
    _get_event_team_and_player,
    read_event_data_xml,
    read_pitch_from_mat_info_xml,
    read_position_data_xml,
    read_teamsheets_from_mat_info_xml,
)

import functions.config as config
from functions.utils_data import infer_ball_carrier

# Constants
PERIOD_DICT = {"firstHalf": 1, "secondHalf": 2}

def load_data(path: str, file_name_pos: str, file_name_infos: str, file_name_events: str) -> Tuple[Dict, Dict, Pitch]:
    """Loads DFL data using floodlight library functions.
    
    Args:
        path: Base path containing the data files.
        file_name_pos: Name of the position data file.
        file_name_infos: Name of the match information file.
        file_name_events: Name of the event data file.
        
    Returns:
        Tuple containing (xy_objects, events, pitch) from floodlight library.
    """
    xy_objects, possession, ballstatus, teamsheets, pitch = read_position_data_xml(
        os.path.join(path, file_name_pos), 
        os.path.join(path, file_name_infos)
    )
    events, _, _ = read_event_data_xml(
        os.path.join(path, file_name_events), 
        os.path.join(path, file_name_infos)
    )
    xy_objects["firstHalf"]["Home"].rotate(180)
    return xy_objects, events, pitch

def load_team_sheets(path: str) -> pd.DataFrame:
    """Loads team sheets from match information file.
    
    Args:
        path: Path to the match directory containing team information.
        
    Returns:
        DataFrame containing combined team information for both home and away teams.
    """
    file_name_info = next((os.path.join(path, filename) for filename in os.listdir(path)
                           if "matchinformation" in filename or "Spielinformationen" in filename), None)

    team_sheets = read_teamsheets_from_mat_info_xml(file_name_info)

    # Add 'team' attribute
    home_team = team_sheets["Home"].teamsheet
    home_team["team"] = "Home"
    away_team = team_sheets["Away"].teamsheet
    away_team["team"] = "Away"

    return pd.concat([home_team, away_team], axis=0, sort=False).reset_index(drop=True)

def extract_match_id(filename: str) -> str:
    """Extracts match ID from filename.
    
    Args:
        filename: Name of the file containing match ID.
        
    Returns:
        Extracted match ID string.
    """
    parts = os.path.splitext(filename)[0].split('_')
    return parts[-1]

def load_event_data(path: str, teamsheet_home: Optional[Teamsheet] = None, 
                   teamsheet_away: Optional[Teamsheet] = None) -> pd.DataFrame:
    """Loads event data from DFL XML files.
    
    Args:
        path: Path to the match directory containing event data.
        teamsheet_home: Optional teamsheet for home team.
        teamsheet_away: Optional teamsheet for away team.
        
    Returns:
        DataFrame containing all event data from both halves.
    """
    file_name_info = next((os.path.join(path, filename) for filename in os.listdir(path)
                           if "matchinformation" in filename or "Spielinformationen" in filename), None)
    file_name_event = next((os.path.join(path, filename) for filename in os.listdir(path)
                                if "events_raw" in filename or "Ereignisdaten-Spiel-Roh" in filename), None)

    events, _, _ = read_event_data_xml(file_name_event, file_name_info,
                                        teamsheet_home=teamsheet_home, teamsheet_away=teamsheet_away)
    events_fullmatch = pd.DataFrame()
    for half in events:
        for team in events[half]:
            # Add 'period' and 'team' attributes
            half_df = events[half][team].events
            
            half_df["period_id"] = PERIOD_DICT[half]
            half_df["team"] = team

            events_fullmatch = pd.concat(
                [events_fullmatch, half_df],
                axis=0, sort=False
            ).sort_values(by=["period_id", "gameclock"])

    return events_fullmatch.reset_index(drop=True)

def convert_dfl_traces_to_wide(dfl_traces: pd.DataFrame, dfl_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DFL-style tracking data (long format) into wide-format table 
    with all players and the ball as separate columns.

    Args:
        dfl_traces (pd.DataFrame): Tracking data in long format (id, x, y, vx, vy, timestamp, etc.)
        dfl_teams (pd.DataFrame): Team sheet including player IDs (pID), xID, and team (Home/Away)

    Returns:
        pd.DataFrame: Wide-format tracking DataFrame with transformed coordinates (0~105, 0~68)
    """
    print("Restoring wide format...")
    # --- 선수 및 공 정보 정리 ---
    player_agents = dfl_teams.rename(columns={'pID': 'id'})
    player_agents['prefix'] = (
        player_agents['team'].map({'Home': 'H', 'Away': 'A'}) 
        + player_agents['xID'].astype(str).str.zfill(2)
    )

    # 공 정보 추가
    ball_agent = pd.DataFrame([{'id': 'ball', 'prefix': 'B00'}])

    # 선수 + 공 병합
    all_agents = pd.concat([
        player_agents[['id', 'prefix']],
        ball_agent
    ], ignore_index=True)

    # --- 시간 변환 ---
    if np.issubdtype(dfl_traces['timestamp'].dtype, np.timedelta64):
        dfl_traces['time'] = dfl_traces['timestamp'].dt.total_seconds()
    else:
        dfl_traces['time'] = pd.to_timedelta(dfl_traces['timestamp']).dt.total_seconds()

    # --- prefix 병합 ---
    merged = dfl_traces.merge(all_agents, on='id', how='left')

    # --- 피벗 (Wide-format 변환) ---
    pivoted = merged.pivot_table(
        index=['frame_id', 'time', 'period_id', 'ball_state', 'ball_owning_team_id'],
        columns='prefix',
        values=['x', 'y', 'vx', 'vy']
    )

    # MultiIndex → 단일 열 이름으로 정리
    pivoted.columns = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()

    # --- 열 순서 정리 ---
    meta_cols = ['frame_id', 'time', 'period_id', 'ball_state', 'ball_owning_team_id']
    ball_cols = ['B00_x', 'B00_y']
    player_prefixes = player_agents['prefix'].unique()
    player_cols = [
        f"{prefix}_{val}" for prefix in player_prefixes 
        for val in ['x', 'y', 'vx', 'vy', 'speed']
    ]

    final_column_order = meta_cols + ball_cols + sorted(player_cols)
    pivoted_final = pivoted.reindex(columns=final_column_order)

    # --- speed 계산 ---
    for prefix in player_prefixes:
        vx_col, vy_col = f"{prefix}_vx", f"{prefix}_vy"
        speed_col = f"{prefix}_speed"
        if vx_col in pivoted_final.columns and vy_col in pivoted_final.columns:
            pivoted_final[speed_col] = np.sqrt(
                pivoted_final[vx_col]**2 + pivoted_final[vy_col]**2
            )

    # --- 좌표 변환 (DFL → SPADL scale) ---
    x_cols = [col for col in pivoted_final.columns if col.endswith('_x')]
    y_cols = [col for col in pivoted_final.columns if col.endswith('_y')]

    pivoted_final[x_cols] = pivoted_final[x_cols] + 52.5
    pivoted_final[y_cols] = pivoted_final[y_cols] + 34
    pivoted_final[x_cols] = pivoted_final[x_cols].clip(lower=0, upper=105)
    pivoted_final[y_cols] = pivoted_final[y_cols].clip(lower=0, upper=68)

    return pivoted_final


def extract_event_pos_optimized(dfl_events, dfl_traces, dfl_teams):
    """
    Vectorized and grouped approach to interpolate player positions for events.
    """
    print("extracting coordinates from tracking data...")
    events = dfl_events.merge(
        dfl_teams[['pID', 'player', 'position']],
        on='pID',  # key 이름이 같으므로 'on' 사용
        how='left'
    )
    # 1. 선수별 prefix를 미리 계산하여 events DataFrame에 열로 추가합니다.
    team_dict = {row.pID: f"{row.team[0]}{row.xID:02d}" for row in dfl_teams.itertuples()}
    events['prefix'] = events['pID'].map(team_dict)

    # 2. 결과를 저장할 'start_x', 'start_y' 열을 미리 생성합니다.
    events['start_x'] = np.nan
    events['start_y'] = np.nan

    # 3. 선수(prefix)와 경기 시간(period_id)으로 그룹화하여 순회합니다.
    #    (이 루프는 전체 이벤트 수보다 훨씬 적게 반복됩니다)
    for group_keys, group_indices in events.groupby(['prefix', 'period_id']).groups.items():
        prefix, period_id = group_keys
        
        # pID가 없는 이벤트 그룹(prefix=NaN)은 건너뜁니다.
        if pd.isna(prefix):
            continue

        # 4. 각 그룹에 대해 단 한 번만 트래킹 데이터를 필터링합니다.
        player_tracking = dfl_traces[dfl_traces['period_id'] == period_id].copy()
        
        x_col = f"{prefix}_x"
        y_col = f"{prefix}_y"

        # 해당 선수의 트래킹 데이터가 없는 경우 건너뜁니다 (교체 선수 등).
        if x_col not in player_tracking.columns:
            continue
            
        # 5. 보간에 필요한 데이터를 추출하고 결측치를 채웁니다.
        #    (np.interp는 NaN 값이 있으면 안 됩니다)
        timestamps = player_tracking['time'].values
        x_coords = player_tracking[x_col].interpolate(method='linear', limit_direction='both').values
        y_coords = player_tracking[y_col].interpolate(method='linear', limit_direction='both').values
        
        # 6. 현재 그룹에 속한 이벤트들의 시간(gameclock)을 가져옵니다.
        event_times = events.loc[group_indices, 'gameclock'].values
        
        # 7. NumPy의 초고속 보간 함수를 사용하여 모든 좌표를 한 번에 계산합니다.
        interp_x = np.interp(event_times, timestamps, x_coords)
        interp_y = np.interp(event_times, timestamps, y_coords)
        
        # 8. 계산된 좌표를 원본 events DataFrame의 올바른 위치에 삽입합니다.
        events.loc[group_indices, 'start_x'] = interp_x
        events.loc[group_indices, 'start_y'] = interp_y
    
    events = events.drop(columns=['prefix'])
    events = events.rename(columns={
        'gameclock': 'time_seconds',
        'pID': 'player_id',
        'position': 'position_name'
    })
    # prefix 열은 더 이상 필요 없으므로 제거합니다.
    return events

def load_position_data(path: str, teamsheet_home: Optional[Teamsheet] = None, 
                      teamsheet_away: Optional[Teamsheet] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads position data from DFL XML files.
    
    Args:
        path: Path to the match directory containing position data.
        teamsheet_home: Optional teamsheet for home team.
        teamsheet_away: Optional teamsheet for away team.
        
    Returns:
        Tuple containing (tracking_fullmatch, teams) DataFrames.
    """
    file_name_info = next((os.path.join(path, filename) for filename in os.listdir(path)
                           if "matchinformation" in filename or "Spielinformationen" in filename), None)
    file_name_pos = next((os.path.join(path, filename) for filename in os.listdir(path) 
                          if "positions_raw" in filename or "Positionsdaten-Spiel-Roh_Observed" in filename), None)

    positions, possession, ballstatus, teamsheets, pitch = read_position_data_xml(
        file_name_pos, file_name_info,
        teamsheet_home=teamsheet_home, teamsheet_away=teamsheet_away
    )
    
    # Add 'team' attribute
    home_team = teamsheets["Home"].teamsheet
    home_team["team"] = "Home"
    
    away_team = teamsheets["Away"].teamsheet
    away_team["team"] = "Away"

    teams = pd.concat([home_team, away_team], axis=0, sort=False).reset_index(drop=True)

    tracking_fullmatch = pd.DataFrame()
    for half in positions:
        tracking_halfmatch = pd.DataFrame()
        for team in positions[half]:
            # Add 'period' and 'time' attributes
            half_df = pd.DataFrame(positions[half][team].xy)

            # Home: H, Away: A, Ball: B
            # Even column index: x coordinate, odd column index: y coordinate
            half_df.columns = [f"{team[0]}{i//2:02d}_{'x' if i % 2 == 0 else 'y'}" for i in range(len(half_df.columns))]

            # Merge horizontally for same time
            tracking_halfmatch = pd.concat([tracking_halfmatch, half_df], axis=1, sort=False)

        tracking_halfmatch["period_id"] = PERIOD_DICT[half]
        tracking_halfmatch["time"] = tracking_halfmatch.index * 0.04  # 1 / 25 = 0.04 seconds per frame

        # axis=0: Different periods/times (from another half) should be appended vertically, stacking the rows in chronological order.
        tracking_fullmatch = pd.concat(
            [tracking_fullmatch, tracking_halfmatch],
            axis=0, sort=False
        ).sort_values(by=["period_id", "time"], kind="mergesort").reset_index(drop=True)

    return tracking_fullmatch, teams

def convert_locations(positions: pd.DataFrame) -> pd.DataFrame:
    """Converts DFL locations to SPADL coordinates.
    
    DFL field dimensions: Pitch(xlim=(-52.5, 52.5), ylim=(-34.0, 34.0)
    SPADL field dimensions: Pitch(xlim=(0, 105), ylim=(0, 68))
    
    Args:
        positions: DataFrame containing position data in DFL coordinates.
        
    Returns:
        DataFrame with coordinates converted to SPADL format.
    """
    x_cols = [col for col in positions.columns if col.endswith("_x")]
    y_cols = [col for col in positions.columns if col.endswith("_y")] 
    positions[x_cols] += (config.field_length / 2)
    positions[y_cols] += (config.field_width / 2)

    positions[x_cols] = np.clip(positions[x_cols], 0, config.field_length)
    positions[y_cols] = np.clip(positions[y_cols], 0, config.field_width)

    return positions

def add_position_info(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Adds position information to events DataFrame.
    
    Args:
        events: DataFrame containing event data.
        teams: DataFrame containing team information.
        
    Returns:
        DataFrame with position information added.
    """
    team_position_mapping = teams[['player_id', 'position']]
    events = events.merge(team_position_mapping, on='player_id', how='left')
    return events
    

def read_event_data_xml(
    filepath_events: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet], Pitch]:
    """Parses a DFL Match Event XML file and extracts the event data as well as
    teamsheets.

    The structure of the official tracking system of the DFL (German Football League)
    contains two separate xml files, one containing the actual data as well as a
    metadata file containing information about teams, pitch size, and start- and
    endframes of match periods. This function provides high-level access to DFL data by
    parsing "the full match" and returning Events-objects parsed from the event data
    xml-file as well as Teamsheet-objects parsed from the metadata xml-file. The number
    of segments is inferred from the data, yet data for each segment is stored in a
    separate object.

    Parameters
    ----------
    filepath_events: str or pathlib.Path
        Full path to XML File where the Event data in DFL format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.
    teamsheet_home: Teamsheet, optional
        Teamsheet-object for the home team used to assign the tIDs of the teams to the
        "Home" and "Away" position. If given as None (default), teamsheet is extracted
        from the Match Information XML file.
    teamsheet_away: Teamsheet, optional
        Teamsheet-object for the away team. If given as None (default), teamsheet is
        extracted from the Match Information XML file. See teamsheet_home for details.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet], Pitch]
        Tuple of (nested) floodlight core objects with shape (events_objects,
        teamsheets, pitch).

        ``events_objects`` is a nested dictionary containing ``Events`` objects for
        each team and segment of the form ``events_objects[segment][team] = Events``.
        For a typical league match with two halves and teams this dictionary looks like:
        ``{
        'firstHalf': {'Home': Events, 'Away': Events},
        'secondHalf': {'Home': Events,'Away': Events}
        }``.

        ``teamsheets`` is a dictionary containing ``Teamsheet`` objects for each team
        of the form ``teamsheets[team] = Teamsheet``.

        ``pitch`` is a ``Pitch`` object corresponding to the data.

    Notes
    -----
    The DFL format of handling event data information involves an elaborate use of
    certain event attributes, which attach additional information to certain events.
    There also exist detailed definitions for these attributes. Parsing this information
    involves quite a bit of logic and is planned to be included in further releases. As
    of now, qualifier information is parsed as a string in the `qualifier` column of the
    returned DataFrame and might be transformed to a dict of the form:
    `{attribute: value}`.
    """
    # set up XML tree
    tree = etree.parse(str(filepath_events))
    root = tree.getroot()

    # read metadata
    pitch = read_pitch_from_mat_info_xml(filepath_mat_info)

    # create or check teamsheet objects
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_away = teamsheets["Away"]
    else:
        pass
        # potential check

    # find start of halves
    start_times = {}
    start_events = root.findall("Event/KickoffWhistle")
    # look at different encodings as the data format changed over time
    if not bool(start_events):  # if no KickoffWhistle is in data search for Kickoff
        start_events = root.findall("Event/Kickoff")
    if not bool(start_events):  # if no Kickoff is in data search for KickOff
        start_events = root.findall("Event/KickOff")
    for event in start_events:
        if event.get("GameSection") is not None:
            start_times[event.get("GameSection")] = iso8601.parse_date(
                event.getparent().get("EventTime")
            )

    # find end of halves
    end_times = {}
    end_events = root.findall("Event/FinalWhistle")
    for event in end_events:
        if event.get("GameSection") is not None:
            end_times[event.get("GameSection")] = iso8601.parse_date(
                event.getparent().get("EventTime")
            )

    # initialize periods
    segments = list(start_times.keys())
    periods = {}
    for segment in segments:
        periods[segment] = (start_times[segment], end_times[segment])

    # set up bins
    team_events = {segment: {} for segment in segments}

    # loop over events
    for elem in root.findall("Event"):
        # initialize
        event = {}

        # check for structure that is an element Event with a single child
        if len(elem) != 1:
            warnings.warn(
                "An XML Event has multiple children. This likely causes imprecise "
                "Event descriptions and outcomes."
            )

        # absolute time information (timestamp)
        event["timestamp"] = iso8601.parse_date(elem.get("EventTime"))
        event["gameclock"] = np.nan

        # segment in which event took place
        segment = None
        for seg in segments:
            if periods[seg][0] <= event["timestamp"] <= periods[seg][1]:
                segment = seg
        # assign to closest start point if not within any segments
        if segment is None:
            seg_ind = np.argmin(
                [np.abs(event["timestamp"] - periods[seg][0]) for seg in segments]
            )
            segment = segments[int(seg_ind)]

        # relative time information (gameclock)
        event["gameclock"] = (event["timestamp"] - periods[segment][0]).total_seconds()
        event["minute"] = np.floor(event["gameclock"] / 60)
        event["second"] = np.floor(event["gameclock"] - event["minute"] * 60)

        # description, outcome, team, and player
        child = next(iter(elem))
        eID, attrib = _get_event_description(child)
        outcome = _get_event_outcome(eID, attrib)
        tID, pID = _get_event_team_and_player(eID, attrib)
        event["eID"] = eID
        event["qualifier"] = attrib
        event["outcome"] = outcome
        event["tID"] = tID
        event["pID"] = pID

        # insert to bin
        if tID not in team_events[segment]:
            team_events[segment][tID] = []
        if event["eID"] == "Substitution":  # split for the special case substitution
            # in-sub
            event["eID"] = "InSubstitution"
            event["pID"] = event["qualifier"]["PlayerIn"]
            team_events[segment][tID].append(event)
            # out-sub
            event["eID"] = "OutSubstitution"
            event["pID"] = event["qualifier"]["PlayerOut"]
            team_events[segment][tID].append(event)
        else:
            team_events[segment][tID].append(event)

    # postprocessing
    team_dfs = {segment: {} for segment in segments}
    for segment in segments:

        # teams
        teams = [tID for tID in team_events[segment] if tID is not None]

        # loop over teams
        for tID in teams:

            # assign events with tID None to both teams
            team_events[segment][tID] += team_events[segment][None]

            # transform to data DataFrame
            team_dfs[segment][tID] = pd.DataFrame(team_events[segment][tID])

            # columns to standard order
            team_dfs[segment][tID] = team_dfs[segment][tID][
                [
                    "eID",
                    "gameclock",
                    "tID",
                    "pID",
                    "outcome",
                    "timestamp",
                    "minute",
                    "second",
                    "qualifier",
                ]
            ]
            team_dfs[segment][tID] = team_dfs[segment][tID].sort_values("gameclock")
            team_dfs[segment][tID] = team_dfs[segment][tID].reset_index(drop=True)

    # check for teams
    team1 = list(team_dfs[segments[0]].keys())[0]
    team2 = list(team_dfs[segments[0]].keys())[1]
    if not np.all([team1 in team_dfs[segment].keys() for segment in segments]):
        KeyError(
            f"Found tID {team1} of the first segment missing in at least one "
            f"other segment!"
        )
    if not np.all([team2 in team_dfs[segment].keys() for segment in segments]):
        KeyError(
            f"Found tID {team2} of the first segment missing in at least one "
            f"other segment!"
        )

    # link team1 and team2 to home and away
    home_tID = teamsheet_home.teamsheet.at[0, "tID"]
    away_tID = teamsheet_away.teamsheet.at[0, "tID"]
    links_team_to_role = {
        "Home": home_tID,
        "Away": away_tID,
    }

    # check if home and away tIDs occur in event data
    if team1 != home_tID and team2 != home_tID:
        raise AttributeError(
            f"Neither tID of teams in the event data ({team1} and {team2}) "
            f"matches the tID of the home team from the "
            f"teamsheet_home ({home_tID})!"
        )
    if team1 != away_tID and team2 != away_tID:
        raise AttributeError(
            f"Neither tID of teams in the event data ({team1} and {team2}) "
            f"matches the tID of the away team from the "
            f"teamsheet_away ({away_tID})!"
        )

    # create objects
    events_objects = {}
    for segment in segments:
        events_objects[segment] = {}
        for team in ["Home", "Away"]:
            events_objects[segment][team] = Events(
                events=team_dfs[segment][links_team_to_role[team]],
            )
    teamsheets = {
        "Home": teamsheet_home,
        "Away": teamsheet_away,
    }

    # pack objects
    data_objects = (events_objects, teamsheets, pitch)

    return data_objects


def read_position_data_xml(
    filepath_positions: Union[str, Path],
    filepath_mat_info: Union[str, Path],
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> Tuple[
    Dict[str, Dict[str, XY]],
    Dict[str, Code],
    Dict[str, Code],
    Dict[str, Teamsheet],
    Pitch,
]:
    """Parse DFL files and extract position data, possession and ballstatus codes as
    well as pitch information and teamsheets.

    The structure of the official tracking system of the DFL (German Football League)
    contains two separate xml files, one containing the actual data as well as a
    metadata file containing information about teams, pitch size, and start- and
    endframes of match periods. However, since no information about framerate is
    contained in the metadata, the framerate is estimated from the time difference
    between individual frames. This function provides high-level access to DFL data by
    parsing "the full match" and returning XY- and Code-objects parsed from the position
    data xml-file as well as Pitch- and Teamsheet-objects parsed from the metadata
    xml-file.

    Parameters
    ----------
    filepath_positions: str or pathlib.Path
        Full path to XML File where the Position data in DFL format is saved.
    filepath_mat_info: str or pathlib.Path
        Full path to XML File where the Match Information data in DFL format is saved.
    teamsheet_home: Teamsheet, optional
        Teamsheet-object for the home team used to create link dictionaries of the form
        `links[team][jID] = xID` and  `links[team][pID] = jID`. The links are used to
        map players to a specific xID in the respective XY objects. Should be supplied
        for custom ordering. If given as None (default), teamsheet is extracted from the
        Match Information XML file and its xIDs are assigned in order of appearance.
    teamsheet_away: Teamsheet, optional
        Teamsheet-object for the away team. If given as None (default), teamsheet is
        extracted from the Match Information XML file. See teamsheet_home for details.

    Returns
    -------
    data_objects: Tuple[Dict[str, Dict[str, XY]], Dict[str, Code], Dict[str, Code], \
     Dict[str, Teamsheet], Pitch]
        Tuple of (nested) floodlight core objects with shape (xy_objects,
        possession_objects, ballstatus_objects, teamsheets, pitch).

        ``xy_objects`` is a nested dictionary containing ``XY`` objects for each team
        and segment of the form ``xy_objects[segment][team] = XY``. For a typical
        league match with two halves and teams this dictionary looks like:
        ``{'firstHalf': {'Home': XY, 'Away': XY}, 'secondHalf': {'Home': XY, 'Away':
        XY}}``.

        ``possession_objects`` is a dictionary containing ``Code`` objects with
        possession information (home or away) for each segment of the form
        ``possession_objects[segment] = Code``.

        ``ballstatus_objects`` is a dictionary containing ``Code`` objects with
        ballstatus information (dead or alive) for each segment of the form
        ``ballstatus_objects[segment] = Code``.

        ``teamsheets`` is a dictionary containing ``Teamsheet`` objects for each team
        of the form ``teamsheets[team] = Teamsheet``.

        ``pitch`` is a ``Pitch`` object corresponding to the data.
    """
    # read metadata
    pitch = read_pitch_from_mat_info_xml(filepath_mat_info)

    # create or check teamsheet objects
    if teamsheet_home is None and teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
        teamsheet_away = teamsheets["Away"]
    elif teamsheet_home is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_home = teamsheets["Home"]
    elif teamsheet_away is None:
        teamsheets = read_teamsheets_from_mat_info_xml(filepath_mat_info)
        teamsheet_away = teamsheets["Away"]
    else:
        pass
        # potential check

    # create links
    if "xID" not in teamsheet_home.teamsheet.columns:
        teamsheet_home.add_xIDs()
    if "xID" not in teamsheet_away.teamsheet.columns:
        teamsheet_away.add_xIDs()
    links_jID_to_xID = {
        "Home": teamsheet_home.get_links("jID", "xID"),
        "Away": teamsheet_away.get_links("jID", "xID"),
    }
    links_pID_to_jID = {
        "Home": teamsheet_home.get_links("pID", "jID"),
        "Away": teamsheet_away.get_links("pID", "jID"),
    }

    # create periods
    periods, framerate_est = _create_periods_from_dat(filepath_positions)
    segments = list(periods.keys())

    # infer data array shapes
    number_of_home_players = max(links_jID_to_xID["Home"].values()) + 1
    number_of_away_players = max(links_jID_to_xID["Away"].values()) + 1
    number_of_frames = {}
    for segment in segments:
        start = periods[segment][0]
        end = periods[segment][1]
        number_of_frames[segment] = end - start + 1

    # bins
    # xydata = {
    #     "Home": {
    #         segment: np.full(
    #             [number_of_frames[segment], number_of_home_players * 2], np.nan
    #         )
    #         for segment in segments
    #     },
    #     "Away": {
    #         segment: np.full(
    #             [number_of_frames[segment], number_of_away_players * 2], np.nan
    #         )
    #         for segment in segments
    #     },
    #     "Ball": {
    #         segment: np.full([number_of_frames[segment], 2], np.nan)
    #         for segment in segments
    #     },
    # }
    xydsdata = {
        "Home": {
            segment: np.full(
                [number_of_frames[segment], number_of_home_players * 4], np.nan
            )
            for segment in segments
        },
        "Away": {
            segment: np.full(
                [number_of_frames[segment], number_of_away_players * 4], np.nan
            )
            for segment in segments
        },
        "Ball": {
            segment: np.full([number_of_frames[segment], 4], np.nan)
            for segment in segments
        },
    }
    # codes = {
    #     code: {segment: [] for segment in segments}
    #     for code in ["possession", "ballstatus"]
    # }
    codes = {
        code: {segment: [] for segment in segments}
        for code in ["possession", "ballstatus", "timestamp", "frame_id"]
    }

    # loop over frame sets containing player & ball positions for all segments
    for _, frame_set in etree.iterparse(filepath_positions, tag="FrameSet"):

        # ball
        if frame_set.get("TeamId").lower() == "ball":
            # (x, y, z, speed)
            segment = frame_set.get("GameSection")
            xydsdata["Ball"][segment][:, 0] = np.array(
                [float(frame.get("X")) for frame in frame_set.iterfind("Frame")]
            )
            xydsdata["Ball"][segment][:, 1] = np.array(
                [float(frame.get("Y")) for frame in frame_set.iterfind("Frame")]
            )
            xydsdata["Ball"][segment][:, 2] = np.array(
                [float(frame.get("Z")) for frame in frame_set.iterfind("Frame")]
            )
            xydsdata["Ball"][segment][:, 3] = np.array(
                [float(frame.get("S")) for frame in frame_set.iterfind("Frame")]
            )
            # codes
            codes["ballstatus"][segment] = [
                float(frame.get("BallStatus")) for frame in frame_set.iterfind("Frame")
            ]
            codes["possession"][segment] = [
                float(frame.get("BallPossession"))
                for frame in frame_set.iterfind("Frame")
            ]
            codes["timestamp"][segment] = [
                frame.get("T")
                for frame in frame_set.iterfind("Frame")
            ]
            codes["frame_id"][segment] = [
                int(frame.get("N"))
                for frame in frame_set.iterfind("Frame")
            ]

        # teams
        else:
            # find identity of frame set
            frames = [frame for frame in frame_set.iterfind("Frame")]
            segment = frame_set.get("GameSection")
            if frame_set.get("PersonId") in links_pID_to_jID["Home"]:
                team = "Home"
                jrsy = links_pID_to_jID[team][frame_set.get("PersonId")]
            elif frame_set.get("PersonId") in links_pID_to_jID["Away"]:
                team = "Away"
                jrsy = links_pID_to_jID[team][frame_set.get("PersonId")]
            else:
                continue
                # possible error or warning

            # insert (x,y) data to correct place in bin
            start = int(frames[0].get("N")) - periods[segment][0]
            end = int(frames[-1].get("N")) - periods[segment][0] + 1
            x_col = (links_jID_to_xID[team][jrsy]) * 4
            y_col = (links_jID_to_xID[team][jrsy]) * 4 + 1
            d_col = (links_jID_to_xID[team][jrsy]) * 4 + 2
            s_col = (links_jID_to_xID[team][jrsy]) * 4 + 3
            xydsdata[team][segment][start:end, x_col] = np.array(
                [float(frame.get("X")) for frame in frames]
            )
            xydsdata[team][segment][start:end, y_col] = np.array(
                [float(frame.get("Y")) for frame in frames]
            )
            xydsdata[team][segment][start:end, d_col] = np.array(
                [float(frame.get("D")) for frame in frames]
            )
            xydsdata[team][segment][start:end, s_col] = np.array(
                [float(frame.get("S")) for frame in frames]
            )

        frame_set.clear()

    # create objects
    xy_objects = {}
    possession_objects = {}
    ballstatus_objects = {}
    timestamp_objects = {}
    frame_objects = {}
    for segment in segments:
        xy_objects[segment] = {}
        possession_objects[segment] = Code(
            code=np.array(codes["possession"][segment]),
            name="possession",
            definitions={1: "Home", 2: "Away"},
            framerate=framerate_est,
        )
        ballstatus_objects[segment] = Code(
            code=np.array(codes["ballstatus"][segment]),
            name="ballstatus",
            definitions={0: "Dead", 1: "Alive"},
            framerate=framerate_est,
        )
        timestamp_objects[segment] = Code(
            code=np.array(codes["timestamp"][segment]),
            name="timestamp",
            framerate=framerate_est,
        )
        frame_objects[segment] = Code(
            code=np.array(codes["frame_id"][segment]),
            name="frame_id",
            framerate=framerate_est,
        )
        for team in ["Home", "Away", "Ball"]:
            xy_objects[segment][team] = XY(
                xy=xydsdata[team][segment],
                framerate=framerate_est,
            )
    teamsheets = {
        "Home": teamsheet_home,
        "Away": teamsheet_away,
    }

    # pack objects
    data_objects = (
        xy_objects,
        possession_objects,
        ballstatus_objects,
        timestamp_objects,
        frame_objects,
        teamsheets,
        pitch,
    )

    return data_objects

def _apply_smoothing_and_outlier_removal(period_df: pd.DataFrame, col: str, 
                                       is_outlier: pd.Series, smoothing_params: Dict) -> pd.DataFrame:
    """Helper function to apply smoothing and outlier removal to a column.
    
    This function implements a comprehensive smoothing and outlier removal pipeline
    using Savitzky-Golay filtering. It first masks outliers, interpolates missing
    values, and then applies smoothing with appropriate parameter validation.
    
    Args:
        period_df: DataFrame containing the period data.
        col: Column name to apply smoothing to.
        is_outlier: Boolean Series indicating outlier values.
        smoothing_params: Dictionary containing smoothing parameters including
                         'window_length' and 'polyorder'.
        
    Returns:
        DataFrame with smoothed column values.
        
    Example:
        >>> smoothed_df = _apply_smoothing_and_outlier_removal(
        ...     period_df, 'vx', is_outlier, {'window_length': 11, 'polyorder': 3}
        ... )
    """
    period_df[col] = period_df[col].mask(is_outlier)
    period_df[col] = period_df[col].interpolate(limit_direction='both')
    
    data_to_smooth = period_df[col].fillna(0)
    window_length = min(smoothing_params['window_length'], len(data_to_smooth))
    if window_length % 2 == 0:
        window_length -= 1
    
    if window_length >= smoothing_params['polyorder'] + 1 and window_length > 0:
        period_df[col] = savgol_filter(data_to_smooth, window_length=window_length, polyorder=smoothing_params['polyorder'])
    else:
        period_df[col] = data_to_smooth
    
    return period_df

def _calculate_kinematics(df: pd.DataFrame, smoothing_params: dict, max_speed: float, max_acceleration: float, is_ball: bool = False):
    """Calculates velocity and acceleration for a single agent over periods."""
    df_out = pd.DataFrame()
    required_cols = ['x', 'y', 'z', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns in input dataframe. Found: {df.columns.tolist()}")
        return df_out

    for period_id in df['period_id'].unique():
        period_df = df[df['period_id'] == period_id].copy()
        period_df = period_df.sort_values(by='timestamp').reset_index(drop=True)

        # Interpolate coordinates
        period_df['x'] = period_df['x'].interpolate()
        period_df['y'] = period_df['y'].interpolate()

        # Calculate time difference
        dt = period_df['timestamp'].diff().dt.total_seconds()

        # Calculate velocities
        coord_cols = ['x', 'y', 'z']
        vel_cols = ['vx', 'vy', 'vz']
        accel_cols = ['ax', 'ay', 'az']
        
        for vel_col, coord_col in zip(vel_cols, coord_cols):
            period_df[vel_col] = period_df[coord_col].diff() / dt
            if not is_ball and vel_col == 'vz':
                period_df[vel_col] = 0.0

        # Calculate speed and apply outlier removal
        period_df['v'] = np.sqrt(sum(period_df[col]**2 for col in vel_cols))
        is_speed_outlier = period_df['v'] > max_speed
        
        for col in vel_cols:
            period_df = _apply_smoothing_and_outlier_removal(period_df, col, is_speed_outlier, smoothing_params)
        
        # Recalculate speed after smoothing
        period_df['v'] = np.sqrt(sum(period_df[col]**2 for col in vel_cols))
        
        # Calculate accelerations
        for accel_col, vel_col in zip(accel_cols, vel_cols):
            period_df[accel_col] = period_df[vel_col].diff() / dt
            if not is_ball and accel_col == 'az':
                period_df[accel_col] = 0.0

        # Calculate acceleration magnitude and apply outlier removal
        period_df['a'] = np.sqrt(sum(period_df[col]**2 for col in accel_cols))
        is_accel_outlier = period_df['a'] > max_acceleration
        
        for col in accel_cols:
            period_df = _apply_smoothing_and_outlier_removal(period_df, col, is_accel_outlier, smoothing_params)
        
        # Recalculate acceleration after smoothing
        period_df['a'] = np.sqrt(sum(period_df[col]**2 for col in accel_cols))
        
        # Limit speed and acceleration
        period_df['v'] = np.minimum(period_df['v'], max_speed)
        period_df['a'] = np.minimum(period_df['a'], max_acceleration)

        df_out = pd.concat([df_out, period_df], ignore_index=True)

    return df_out

def load_and_assemble_tracking_data(path: str, match_id: str) -> Tuple[pd.DataFrame, Optional[dict], Optional[dict]]:
    """Loads and preprocesses raw XML tracking data for a given match.
    
    Args:
        path: Base path containing match folders.
        match_id: The ID of the match to process.
        
    Returns:
        Tuple containing (tracking_fullmatch, teams_dict, pitch_meta).
        Returns empty DataFrame and None values if loading or processing fails.
    """
    print(f"Processing match: {match_id}")
    match_path = os.path.join(path, match_id)

    if not os.path.isdir(match_path):
        print(f"Error: Match directory not found at {match_path}")
        return pd.DataFrame(), None, None

    # Find required files
    try:
        # Handle different file naming patterns for dfl-confidential and dfl-spoho
        # dfl-confidential: "Positionsdaten-Spiel-Roh_Observed_DFL-MAT-*.xml"
        # dfl-spoho: "positions_raw_observed_DFL-COM-*.xml"
        file_name_pos = next((f for f in os.listdir(match_path) 
                             if any(pattern in f for pattern in 
                                   ["positions_raw_observed", "Positionsdaten-Spiel-Roh_Observed"])), None)
        
        # Handle different match information file patterns
        # Both use "matchinformation" but with different prefixes
        file_name_info = next((f for f in os.listdir(match_path) 
                              if "matchinformation" in f or "Spielinformationen" in f), None)

        if not file_name_pos or not file_name_info:
            print(f"Error: Position or Match Information file not found in {match_path}")
            print(f"Available files: {os.listdir(match_path)}")
            return pd.DataFrame(), None, None
    except Exception as e:
        print(f"Error finding files in {match_path}: {e}")
        return pd.DataFrame(), None, None

    # Load raw data using external function
    try:
        print("Reading XML data...")
        pos_filepath = os.path.join(match_path, file_name_pos)
        info_filepath = os.path.join(match_path, file_name_info)
        xyds_objects, possession, ballstatus, timestamp, frames, teamsheets, pitch_meta = read_position_data_xml(pos_filepath, info_filepath)
    except Exception as e:
        print(f"Error reading XML data for {match_id}: {e}")
        return pd.DataFrame(), None, None

    # Prepare team info
    try:
        home_team = teamsheets["Home"].teamsheet
        home_team["team"] = "Home"
        away_team = teamsheets["Away"].teamsheet
        away_team["team"] = "Away"
        teams_dict = {'Home': home_team, 'Away': away_team}
    except Exception as e:
        print(f"Error processing teamsheets for {match_id}: {e}")
        return pd.DataFrame(), None, None

    # Initial DataFrame assembly (wide format)
    print("Assembling initial wide DataFrame...")
    tracking_fullmatch = pd.DataFrame()
    try:
        for half, period_id_val in PERIOD_DICT.items():
            if half not in xyds_objects:
                continue

            tracking_halfmatch = pd.DataFrame()
            processed_teams_in_half = set()

            # Process ball first if exists
            if 'Ball' in xyds_objects[half]:
                ball_df = pd.DataFrame(xyds_objects[half]['Ball'].xy)
                ball_df.columns = ['ball_x', 'ball_y', 'ball_z', 'ball_speed']
                tracking_halfmatch = pd.concat([tracking_halfmatch, ball_df], axis=1)
                processed_teams_in_half.add('Ball')

            # Process home and away teams
            for team_name in ['Home', 'Away']:
                if team_name in xyds_objects[half]:
                    team_id_info = teams_dict[team_name]
                    agent_ids = team_id_info['pID'].values
                    team_df = pd.DataFrame(xyds_objects[half][team_name].xy)
                    team_df.columns = [f'{p_id}_{axis}' for p_id in agent_ids for axis in ['x', 'y', 'd', 's']]
                    tracking_halfmatch = pd.concat([tracking_halfmatch, team_df], axis=1)
                    processed_teams_in_half.add(team_name)

            if tracking_halfmatch.empty:
                continue

            # Add metadata columns
            tracking_halfmatch["period_id"] = period_id_val
            tracking_halfmatch['frame_id'] = frames[half].code
            timestamp_half = pd.to_datetime(timestamp[half].code)
            timestamp_half.index = tracking_halfmatch.index
            tracking_halfmatch['timestamp'] = timestamp_half - timestamp_half[0]
            
            if half in ballstatus:
                tracking_halfmatch['ball_state'] = ballstatus[half].code
                tracking_halfmatch['ball_state'] = tracking_halfmatch['ball_state'].map(
                    lambda x: ballstatus[half].definitions[x].lower() if x in ballstatus[half].definitions else 'unknown'
                )
            else:
                tracking_halfmatch['ball_state'] = 'unknown'

            if half in possession:
                tracking_halfmatch['ball_owning_team_id'] = possession[half].code
                def map_possession(code):
                    try:
                        team_name = possession[half].definitions[int(code)]
                        return teams_dict[team_name]['tID'].unique()[0]
                    except (KeyError, ValueError, IndexError):
                        return 'Unknown'
                tracking_halfmatch['ball_owning_team_id'] = tracking_halfmatch['ball_owning_team_id'].map(map_possession)
            else:
                tracking_halfmatch['ball_owning_team_id'] = 'Unknown'

            tracking_fullmatch = pd.concat([tracking_fullmatch, tracking_halfmatch], axis=0, sort=False)

        if tracking_fullmatch.empty:
            print("Error: No tracking data assembled.")
            return pd.DataFrame(), None, None

        # Sort by time and reset index
        tracking_fullmatch = tracking_fullmatch.sort_values(
            by=["period_id", "timestamp"], kind="mergesort"
        ).reset_index(drop=True)

    except Exception as e:
        print(f"Error assembling wide DataFrame for {match_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None, None
    
    return tracking_fullmatch, teams_dict, pitch_meta

def reshape_and_calculate_kinematics(
    tracking_fullmatch: pd.DataFrame,
    teams_dict: dict,
    match_id: str,
    player_smoothing_params: dict = config.DEFAULT_PLAYER_SMOOTHING_PARAMS,
    ball_smoothing_params: dict = config.DEFAULT_BALL_SMOOTHING_PARAMS,
    max_player_speed: float = config.MAX_PLAYER_SPEED,
    max_player_acceleration: float = config.MAX_PLAYER_ACCELERATION,
    max_ball_speed: float = config.MAX_BALL_SPEED,
    max_ball_acceleration: float = config.MAX_BALL_ACCELERATION
) -> pd.DataFrame:
    """Reshapes tracking data to long format and calculates kinematics for all agents.
    
    Args:
        tracking_fullmatch: Wide format DataFrame containing tracking data.
        teams_dict: Dictionary containing team information.
        match_id: Match ID for context.
        player_smoothing_params: Savgol filter parameters for players.
        ball_smoothing_params: Savgol filter parameters for the ball.
        max_player_speed: Maximum plausible player speed.
        max_player_acceleration: Maximum plausible player acceleration.
        max_ball_speed: Maximum plausible ball speed.
        max_ball_acceleration: Maximum plausible ball acceleration.
        
    Returns:
        DataFrame with tracking data in long format including calculated kinematics.
    """
    all_players_teamsheet = pd.concat([teams_dict['Home'], teams_dict['Away']], axis=0, sort=False).reset_index(drop=True)
    
    # Reshape to long format and calculate kinematics
    print("Reshaping to long format and calculating kinematics...")
    tracking_long_final = pd.DataFrame()
    base_cols = ['period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id']
    base_cols = [col for col in base_cols if col in tracking_fullmatch.columns]

    # Identify all unique agent IDs present
    player_cols = [col for col in tracking_fullmatch.columns if col.startswith("DFL")]
    agent_ids_present = set(col.split('_')[0] for col in player_cols)
    if 'ball_x' in tracking_fullmatch.columns:
        agent_ids_present.add('ball')

    for agent_id in tqdm(agent_ids_present, desc="Processing agents"):
        is_ball = (agent_id == 'ball')

        # Select columns for this agent
        if is_ball:
            agent_cols = [col for col in tracking_fullmatch.columns if col.startswith('ball_') and col[-2:] in ['_x', '_y', '_z', '_d', '_s']]
            rename_map = {col: col.split('_')[1] for col in agent_cols}
        else:
            agent_cols = [col for col in tracking_fullmatch.columns if col.startswith(f"{agent_id}_") and col[-1] in ['x', 'y', 'z', 'd', 's']]
            rename_map = {f"{agent_id}_x": "x", f"{agent_id}_y": "y"}

        # Check if essential coordinate columns exist
        essential_coords = ['x', 'y'] if not is_ball else ['x', 'y', 'z']
        if not all(new_name in rename_map.values() for new_name in essential_coords):
            print(f"Warning: Missing essential coordinates for agent {agent_id}. Skipping.")
            continue

        # Extract base + agent columns, then rename
        current_agent_df = tracking_fullmatch[base_cols + agent_cols].copy()
        current_agent_df = current_agent_df.rename(columns=rename_map)

        # Add z=0 for players if not present
        if not is_ball and 'z' not in current_agent_df.columns:
            current_agent_df['z'] = 0.0

        # Drop rows with NaN coordinates for players
        if not is_ball:
            current_agent_df = current_agent_df.dropna(subset=['x', 'y']).copy()
            if current_agent_df.empty:
                continue

        # Calculate kinematics using helper function
        smoothing = ball_smoothing_params if is_ball else player_smoothing_params
        max_v = max_ball_speed if is_ball else max_player_speed
        max_a = max_ball_acceleration if is_ball else max_player_acceleration

        kinematics_df = _calculate_kinematics(current_agent_df, smoothing, max_v, max_a, is_ball)

        if kinematics_df.empty:
            continue

        # Add agent ID and team/position info
        kinematics_df['id'] = agent_id
        if is_ball:
            kinematics_df['team_id'] = 'ball'
            kinematics_df['position_name'] = 'ball'
        else:
            try:
                player_info = all_players_teamsheet[all_players_teamsheet['pID'] == agent_id].iloc[0]
                kinematics_df['team_id'] = player_info['tID']
                kinematics_df['position_name'] = player_info['position']
            except (IndexError, KeyError):
                print(f"Warning: Could not find team/position info for player {agent_id}. Setting defaults.")
                kinematics_df['team_id'] = 'Unknown'
                kinematics_df['position_name'] = 'Unknown'

        # Append to the final long dataframe
        tracking_long_final = pd.concat([tracking_long_final, kinematics_df], ignore_index=True)

    # Final sorting and cleanup
    if tracking_long_final.empty:
        print(f"Error: No data processed for {match_id}.")
        return pd.DataFrame()

    # Add back is_ball_carrier information - requires merging or careful alignment
    # try:
    #      bc_info = tracking_fullmatch[['period_id', 'frame_id', 'ball_owning_team_id']].drop_duplicates()
    #      # Create a 'potential' ball carrier flag
    #      tracking_long_final['is_potential_bc'] = (tracking_long_final['team_id'] == tracking_long_final['ball_owning_team_id'])
    #      # Need to refine this - typically only ONE ball carrier per frame.
    #      # This requires knowing which player ID actually matches the ball_owning_team_id logic
    #      # For now, just keep the flag based on team ownership
    #      print("Warning: 'is_ball_carrier' logic needs refinement based on specific player ID per frame.")
    #      tracking_long_final['is_ball_carrier'] = False # Placeholder - Actual logic needed


    # except KeyError:
    #      print("Warning: Could not determine ball carrier information.")
    #      tracking_long_final['is_ball_carrier'] = False

    # Sort final DataFrame
    tracking_long_final = tracking_long_final.sort_values(
        by=["period_id", "timestamp", "id"], kind="mergesort"
    ).reset_index(drop=True)

    # Define final column order
    final_cols_order = [
        'game_id', 'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
        'x', 'y', 'z', 'vx', 'vy', 'vz', 'v', 'ax', 'ay', 'az', 'a',
        'id', 'team_id', 'position_name'
    ]
    
    # Add game_id
    tracking_long_final['game_id'] = match_id
    # Reorder, keeping only existing columns
    tracking_long_final = tracking_long_final[[col for col in final_cols_order if col in tracking_long_final.columns]]

    print(f"Finished processing {match_id}. Final shape: {tracking_long_final.shape}")
    return tracking_long_final

# def load_all_data(data_path: str) -> None:
#     """Loads and processes all data in the specified directory.
    
#     Args:
#         data_path: Path to the directory containing match data.
#     """
#     match_ids = os.listdir(data_path)
#     print(match_ids)
#     total_dict = {match_id: {} for match_id in match_ids}
#     processed_path = os.path.join(os.path.dirname(data_path), "processed")
    
#     for match_id in match_ids:
#         print(f"Preprocessing {match_id}")
#         tracking_df, teams_dict, pitch_meta = load_and_assemble_tracking_data(data_path, match_id)
#         if teams_dict is not None:
#             tracking_df = reshape_and_calculate_kinematics(tracking_df, teams_dict, match_id)
#             tracking_df = infer_ball_carrier(tracking_df)
#         else:
#             tracking_df = None
#         match_path = os.path.join(data_path, match_id)
#         event_df = load_event_data(match_path)
#         total_dict[match_id]['tracking_df'] = tracking_df
#         total_dict[match_id]['event_df'] = event_df
#         total_dict[match_id]['teams'] = teams_dict
#         total_dict[match_id]['pitch_meta'] = pitch_meta
        
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(f"{processed_path}/{match_id}/{match_id}_processed_dict.pkl"), exist_ok=True)
#         with open(f"{processed_path}/{match_id}/{match_id}_processed_dict.pkl", "wb") as f:
#             pickle.dump(total_dict[match_id], f)


def load_all_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes all data in the specified directory, returning aggregated DataFrames.
    
    Args:
        data_path: Path to the directory containing match data folders.
        
    Returns:
        A tuple containing three pandas DataFrames:
        - events_df: All events from all matches concatenated.
        - tracking_df: All processed tracking data from all matches concatenated.
        - teams_df: All team/player information from all matches concatenated.
    """
    match_ids = [d for d in os.listdir(data_path) if "DFL" in d and not d.startswith(".")]
    print(f"Found matches: {match_ids}")
    
    # 각 데이터를 담을 리스트 초기화
    all_tracking_dfs = []
    all_event_dfs = []
    all_teams_dfs = []
    
    for match_id in match_ids:
        print(f"--- Processing {match_id} ---")
        
        # 1. 트래킹 데이터 처리
        tracking_df, teams_dict, pitch_meta = load_and_assemble_tracking_data(data_path, match_id)
        
        if teams_dict is not None and tracking_df is not None and not tracking_df.empty:
            tracking_df = reshape_and_calculate_kinematics(tracking_df, teams_dict, match_id)
            tracking_df = infer_ball_carrier(tracking_df)
            all_tracking_dfs.append(tracking_df)
            
            # 3. 팀/선수 데이터 처리
            home_team = teams_dict.get('Home', pd.DataFrame()).copy()
            away_team = teams_dict.get('Away', pd.DataFrame()).copy()
            
            if not home_team.empty or not away_team.empty:
                match_teams_df = pd.concat([home_team, away_team], ignore_index=True)
                match_teams_df['match_id'] = match_id # 어떤 경기 소속인지 명시
                all_teams_dfs.append(match_teams_df)
        
        # 2. 이벤트 데이터 처리
        match_path = os.path.join(data_path, match_id)
        event_df = load_event_data(match_path)
        if not event_df.empty:
            event_df['match_id'] = match_id # 어떤 경기 소속인지 명시
            all_event_dfs.append(event_df)
        
    # 모든 경기 데이터를 하나로 합치기
    # 리스트가 비어있을 경우 에러 방지를 위해 빈 DataFrame 생성
    events_df = pd.concat(all_event_dfs, ignore_index=True) if all_event_dfs else pd.DataFrame()
    tracking_df = pd.concat(all_tracking_dfs, ignore_index=True) if all_tracking_dfs else pd.DataFrame()
    final_teams_df = pd.concat(all_teams_dfs, ignore_index=True) if all_teams_dfs else pd.DataFrame()
    

    final_tracking_df = convert_dfl_traces_to_wide(tracking_df, final_teams_df)
    final_events_df = extract_event_pos_optimized(events_df, final_tracking_df, final_teams_df)
    print("\n--- All matches processed. Returning aggregated DataFrames. ---")
    return final_events_df, final_tracking_df, final_teams_df


def print_match_info(match_info_path: str) -> Optional[dict]:
    """Extracts and prints match information from the XML tree.
    
    Args:
        match_info_path: Path to the match information XML file.
        
    Returns:
        Dictionary containing general match attributes, or None if not found.
    """
    # Set up XML tree
    tree = etree.parse(match_info_path)
    root = tree.getroot()
    
    # Find the <MatchInformation> node within the root
    match_info_node = None
    for child in root:
        if child.tag == "MatchInformation":
            match_info_node = child
            break
    
    if match_info_node is None:
        print("No <MatchInformation> tag found.")
        return None
    
    # Initialize dictionaries to hold attributes from the child nodes
    general_attrib = {}
    env_attrib = {}
    other_info_attrib = {}
    
    # Iterate through the children of <MatchInformation>
    for subchild in match_info_node:
        if subchild.tag == "General":
            general_attrib = subchild.attrib
        elif subchild.tag == "Environment":
            env_attrib = subchild.attrib
        elif subchild.tag == "OtherGameInformation":
            other_info_attrib = subchild.attrib
    
    # Print general match information in a readable format
    print(f" - Competition: {general_attrib.get('CompetitionName', 'N/A')}")
    print(f" - Match Day: {general_attrib.get('MatchDay', 'N/A')}, Season: {general_attrib.get('Season', 'N/A')}")
    print(f" - Match Title: {general_attrib.get('MatchTitle', 'N/A')}")
    print(f" - Result: {general_attrib.get('Result', 'N/A')}")
    
    return general_attrib
    # 4. Print environment details
    # print("\n=== [Environment Info] ===")
    # print(f" - Stadium: {env_attrib.get('StadiumName', 'N/A')} (ID: {env_attrib.get('StadiumId', '')})")
    # print(f" - Address: {env_attrib.get('StadiumAddress', 'N/A')}")
    # print(f" - Capacity: {env_attrib.get('StadiumCapacity', 'N/A')}")
    # print(f" - Spectators: {env_attrib.get('NumberOfSpectators', 'N/A')}")
    # print(f" - Roof: {env_attrib.get('Roof', 'N/A')}")
    # print(f" - Temperature: {env_attrib.get('Temperature', 'N/A')}°C")
    # print(f" - Humidity: {env_attrib.get('AirHumidity', 'N/A')}%")
    # print(f" - Pitch Size: {env_attrib.get('PitchX', 'N/A')} × {env_attrib.get('PitchY', 'N/A')}")
    
    # 5. Print additional game information if available
    # if other_info_attrib:
    #     print("\n=== [Other Game Info] ===")
    #     print(f" - TotalTimeFirstHalf: {other_info_attrib.get('TotalTimeFirstHalf', 'N/A')}")
    #     print(f" - TotalTimeSecondHalf: {other_info_attrib.get('TotalTimeSecondHalf', 'N/A')}")
    #     print(f" - PlayingTimeFirstHalf: {other_info_attrib.get('PlayingTimeFirstHalf', 'N/A')}")
    #     print(f" - PlayingTimeSecondHalf: {other_info_attrib.get('PlayingTimeSecondHalf', 'N/A')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess DFL raw tracking data.")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to raw DFL data directory")
    
    args = parser.parse_args()
    data_path = args.data_path
    load_all_data(data_path)
    print('Done')


