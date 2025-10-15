import os
import numpy as np
import pandas as pd
import ast
import scipy.signal as signal
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml
import imputer.config as config


period_dict = {"firstHalf": 1, "secondHalf": 2}
# Load Data
def load_data(path, file_name_pos, file_name_infos, file_name_events):
    xy_objects, possession, ballstatus, teamsheets, pitch = read_position_data_xml(os.path.join(path, file_name_pos), 
                                                                                   os.path.join(path, file_name_infos))
    events, _, _ = read_event_data_xml(os.path.join(path, file_name_events), 
                                       os.path.join(path, file_name_infos))
    xy_objects["firstHalf"]["Home"].rotate(180)
    return xy_objects, events, pitch

# Load Team Sheets
def load_team_sheets(path):
    file_name_info = next((os.path.join(path, filename) for filename in os.listdir(path)
                           if "matchinformation" in filename), None)

    team_sheets = read_teamsheets_from_mat_info_xml(file_name_info)

    # add 'team' attribute
    home_team = team_sheets["Home"].teamsheet
    home_team["team"] = "Home"
    away_team = team_sheets["Away"].teamsheet
    away_team["team"] = "Away"

    return pd.concat(
        [home_team, away_team],
        axis=0, sort=False
    ).reset_index(drop=True)

# Extract match ID from filename
def extract_match_id(filename):
    parts = os.path.splitext(filename)[0].split('_')
    return parts[-1]

# Load Event Data
def load_event_data(path, teamsheet_home=None, teamsheet_away=None):
    file_name_info = next((os.path.join(path, filename) for filename in os.listdir(path)
                           if "matchinformation" in filename), None)
    file_name_event = next((os.path.join(path, filename) for filename in os.listdir(path)
                                if "events_raw" in filename), None)

    # Return type: Tuple[Dict[str, Dict[str, Events]], Dict[str, Teamsheet], Pitch]
    events, _, _ = read_event_data_xml(file_name_event, file_name_info,
                                        teamsheet_home=teamsheet_home, teamsheet_away=teamsheet_away)
    
    events_fullmatch = pd.DataFrame()
    for half in events:
        for team in events[half]:
            # add 'period' and 'team' attributes
            half_df = events[half][team].events
            
            half_df["period_id"] = period_dict[half]
            half_df["team"] = team

            events_fullmatch = pd.concat(
                [events_fullmatch, half_df],
            axis=0, sort=False
            ).sort_values(by=["period_id", "gameclock"])

    return events_fullmatch.reset_index(drop=True)

# Load Position Data
def load_position_data(path, teamsheet_home=None, teamsheet_away=None, ):
    file_name_info = next((os.path.join(path, filename) for filename in os.listdir(path)
                           if "matchinformation" in filename), None)
    file_name_pos = next((os.path.join(path, filename) for filename in os.listdir(path) 
                          if "positions_raw" in filename), None)


    # Return type: Tuple[Dict[str, Dict[str, XY]], Dict[str, Code], Dict[str, Code], Dict[str, Teamsheet], Pitch]
    positions, possession, ballstatus, teamsheets, pitch = read_position_data_xml(file_name_pos, file_name_info,
                                                    teamsheet_home=teamsheet_home, teamsheet_away=teamsheet_away)
    
    # add 'team' attribute
    home_team = teamsheets["Home"].teamsheet
    home_team["team"] = "Home"
    
    away_team = teamsheets["Away"].teamsheet
    away_team["team"] = "Away"

    teams = pd.concat([home_team, away_team],
        axis=0, sort=False
    ).reset_index(drop=True)

    tracking_fullmatch = pd.DataFrame()
    for half in positions:
        tracking_halfmatch = pd.DataFrame()
        for team in positions[half]:
            # add 'period' and 'time' attributes
            half_df = pd.DataFrame(positions[half][team].xy)

            # Home: H, Away: A, Ball: B
            # even column index: x coordinate
            # odd  column index: y coordinate, 
            half_df.columns = [f"{team[0]}{i//2:02d}_{'x' if i % 2 == 0 else 'y'}" for i in range(len(half_df.columns))]

            # axis=1: For the same time (index), home/away data should be merged horizontally so that both teams' positions for a given time appear side by side. 
            tracking_halfmatch = pd.concat(
                [tracking_halfmatch, half_df],
                axis=1, sort=False
            )

        tracking_halfmatch["period_id"] = period_dict[half]
        tracking_halfmatch["time"] = tracking_halfmatch.index * 0.04  # 1 / 25 = 0.04 seconds per frame

        # axis=0: Different periods/times (from another half) should be appended vertically, stacking the rows in chronological order.
        tracking_fullmatch = pd.concat(
            [tracking_fullmatch, tracking_halfmatch],
            axis=0, sort=False
        ).sort_values(by=["period_id", "time"], kind="mergesort").reset_index(drop=True)

    return tracking_fullmatch, teams
    return n_frames

# Display Data Summary
def display_data_summary(path):
    print(f"📂 Dataset path: {path}")

    team_sheets_all = load_team_sheets(path)
    all_events = load_event_data(path)
    n_frames = load_position_data(path)

    print(f"📊 team_sheets_all 데이터 타입: {type(team_sheets_all)}")
    print(f"📏 team_sheets_all 크기: {team_sheets_all.shape if isinstance(team_sheets_all, pd.DataFrame) else 'N/A'}")
    print(f"📜 team_sheets_all 컬럼 목록: {team_sheets_all.columns if isinstance(team_sheets_all, pd.DataFrame) else 'N/A'}")

    print("🎯 Unique player IDs:", team_sheets_all["pID"].nunique())
    print("🏆 Unique teams:", team_sheets_all["team"].nunique())
    print("📈 Total number of events:", len(all_events))
    print("📊 Unique event ID counts:\n", all_events["eID"].value_counts())
    print("📉 Total number of position frames:", n_frames)

def extract_event_pos(events, tracking_data, team_sheets, fps=config.frame_rate):
    '''
    Linearly interpolate the position of a player at a given time.
    '''
    team_dict = {row.pID: f"{row.team[0]}{row.xID:02d}" for row in team_sheets.itertuples()} # {'pID': 'tracking_data_column_name'}

    def interpolate_pos(row):
        if row.pID is None:
            return pd.Series([None, None])

        copy_tracking_data = tracking_data[tracking_data["period_id"] == row.period_id].reset_index(drop=True)
        closest_idx = copy_tracking_data["time"].sub(row.gameclock).abs().idxmin()
        
        x_col = team_dict[row.pID] + "_x"
        y_col = team_dict[row.pID] + "_y"

        lower_index = closest_idx
        # DFL-MAT-J03YLO: row1645데이터는 tracking_data의 마지막 데이터와 동일한 시점에 위치하고 있음 -> upper_index > len(copy_tracking_data)인 경우가 존재함
        upper_index = closest_idx + 1 if closest_idx + 1 < len(copy_tracking_data) else closest_idx
        x1, y1 = copy_tracking_data[x_col].iloc[lower_index], copy_tracking_data[y_col].iloc[lower_index]
        x2, y2 = copy_tracking_data[x_col].iloc[upper_index], copy_tracking_data[y_col].iloc[upper_index]

        alpha = (row.gameclock - copy_tracking_data.at[lower_index, "time"]) / config.frame_rate
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)

        return pd.Series([x, y])
    
    events[["start_x", "start_y"]] = events.apply(
        lambda row: interpolate_pos(row), axis=1
    )

    return events

def convert_locations(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DFL locations to spadl coordinates.

    DFL field dimensions: Pitch(xlim=(-52.5, 52.5), ylim=(-34.0, 34.0)
    SPADL field dimensions: Pitch(xlim=(0, 105), ylim=(0, 68))
    """

    x_cols = [col for col in positions.columns if col.endswith("_x")]
    y_cols = [col for col in positions.columns if col.endswith("_y")] 
    positions[x_cols] += (config.field_length / 2)
    positions[y_cols] += (config.field_width / 2)

    positions[x_cols] = np.clip(positions[x_cols], 0, config.field_length)
    positions[y_cols] = np.clip(positions[y_cols], 0, config.field_width)

    return positions


def add_position_info(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    team_position_mapping = teams[['player_id', 'position']]
    events = events.merge(team_position_mapping, on='player_id', how='left')
    return events
    

def calc_player_velocities(positions, team_sheets, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """
    Parameters
    -----------
        positions: the tracking DataFrame for home or away team
        home_df, away_df: Team dataframes
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN.
    Returns
    -----------
       positions : the tracking DataFrame with columns for speed in the x & y direction and total speed added
    """
    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = 1 / config.frame_rate # 0.04 seconds
    # index of first frame in second half
    second_half_idx = positions['period_id'].idxmax(axis=0)
    # estimate velocities for players in team
    player_ids = [p[:3] for p in positions.columns if p.endswith("_x") and not p.startswith("Ball")]
    velocity_data = {}
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        x = pd.Series([float(p) for p in positions[player+"_x"]])
        y = pd.Series([float(p) for p in positions[player+"_y"]])
        vx = x.diff() / dt
        vy = y.diff() / dt
        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt(vx**2 + vy**2)
            vx[raw_speed>maxspeed] = np.nan
            vy[raw_speed>maxspeed] = np.nan
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' )
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' )
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' )
            else:
                raise ValueError("Invalid filter type. Must be either 'Savitzky-Golay' or 'moving average'")
        # Store player speed in x, y direction, and total speed in the dictionary
        velocity_data[player + "_vx"] = vx
        velocity_data[player + "_vy"] = vy
        velocity_data[player + "_speed"] = np.sqrt(vx**2 + vy**2)
    velocity_df = pd.DataFrame(velocity_data).round(2)
    return pd.concat([positions, velocity_df], axis=1)

def extract_related_positions(events, position, teams, fps=config.frame_rate):
    """
    선수와 관련 선수의 위치를 선형 보간법으로 계산합니다.
    """
    events["related_x"] = None
    events["related_y"] = None
    events["related_id"] = None
    
    # 선수 ID를 트래킹 데이터 컬럼 이름으로 매핑
    team_dict = {row.player_id: f"{row.team[0]}{row.xID:02d}" for row in teams.itertuples()}
    target_events = events[(events['type_name'].str.endswith('Pass')) | (events['type_name'] == "TacklingGame")]
    
    for i, row in target_events.iterrows():
        if pd.isna(row["player_id"]) or pd.isna(row["team"]):
            continue
            
        copy_tracking_data = position[position["period_id"] == row.period_id].reset_index(drop=True)
        closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()
        
        # qualifier 파싱하여 관련 선수 ID 찾기
        qualifier_str = row['qualifier']
        parsed = ast.literal_eval(qualifier_str) if isinstance(qualifier_str, str) else qualifier_str
        
        # 이벤트 타입에 따라 관련 ID 결정
        related_id = None
        if row.type_name == "TacklingGame":
            if row.player_id == parsed.get('Winner', None):
                related_id = parsed.get('Loser', None)
            else:
                related_id = parsed.get('Winner', None)
        elif row['type_name'].endswith('Pass'):
            related_id = parsed.get('Recipient', None)

        events.at[i, "related_id"] = related_id
            
        if related_id is None:
            continue        

        related_x_col = team_dict[related_id] + "_x"
        related_y_col = team_dict[related_id] + "_y"

        lower_index = closest_idx
        upper_index = closest_idx + 1 if closest_idx + 1 < len(copy_tracking_data) else closest_idx

        x1, y1 = copy_tracking_data[related_x_col].iloc[lower_index], copy_tracking_data[related_y_col].iloc[lower_index]
        x2, y2 = copy_tracking_data[related_x_col].iloc[upper_index], copy_tracking_data[related_y_col].iloc[upper_index]

        alpha = (row.time_seconds - copy_tracking_data.at[lower_index, "time"]) / config.frame_rate
        related_x  = x1 + alpha * (x2 - x1)
        related_y  = y1 + alpha * (y2 - y1)

        events.at[i, "related_x"] = related_x
        events.at[i, "related_y"] = related_y
    
    return events