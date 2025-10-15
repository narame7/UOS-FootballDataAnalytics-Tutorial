"""Implements the features used in exPressComponent."""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

import ast
import numpy as np
import pandas as pd

from functools import wraps
from typing import Dict,Callable, List, no_type_check
from scipy.ndimage import gaussian_filter
from mplsoccer import Pitch

import imputer.config as config
from scipy.stats import binned_statistic_2d, circmean
from mplsoccer import dimensions

from imputer.config import on_ball_actions, PLAYER_ROLE_MAPPING, EVENT_LABEL_MAPPING
from datatools.preprocess import display_data_summary, load_event_data, load_position_data, load_data, extract_match_id, load_team_sheets

def heatmap(events: pd.DataFrame, teams: pd.DataFrame):
    """
    Computes a heatmap for all players at each event timestamp.
    
    This function serves as a baseline experiment to assess whether player_id detection.
    the player ID with the maximum probability for a given (x, y) coordinate from tracking 
    data (or 360 data) can be used for detection.

    returns : pd.DataFrame(shape=(n_events, n_players))
    """
    # parameters: sigma, time_window, bins, include_event_types
    sigma = 45
    time_window = 45+20 # considier additional time in period
    bins = (int(config.field_length), int(config.field_width))
    # bins = (25, 25)

    pitch = Pitch(pitch_type="custom", pitch_length=config.field_length, pitch_width=config.field_width,
                    line_zorder=2, pitch_color="#22312b", line_color="#efefef")
    
    #events["time_bin"] = (events["time_seconds"] // (time_window * 60)).astype(int)
    # 각 period 내 상대 시간 계산: 시간축이 DFL과 다름: 전후반이 이어지는  시간대가 아님
    events["period_time_seconds"] = events.groupby("period_id")["time_seconds"].transform(lambda x: x - x.min())
    events["time_bin"] = (events["period_time_seconds"] // (time_window * 60)).astype(int)

    player_ids = teams["player_id"].unique()
    # Initialize heatmap dictionary
    heatmap = {
        player_id: {
            (period_id, time_bin): np.nan
            for time_bin in events["time_bin"].unique()
            for period_id in events["period_id"].unique()
        }
        for player_id in player_ids
    }

    grouped = events.groupby(["period_id", "time_bin"])
    for (period_id, time_bin), group in grouped:
    # grouped = events.groupby(["time_bin"])

    # for time_bin, group in grouped:

        for player_id in player_ids:
            player_df = group[group["player_id"] == player_id]
           
            # global features1: if a player does not have any event in the Time Window, use a first event after time bin and a last event before time bin
            if player_df.empty:
                player_df1 = events[(events["player_id"] == player_id) & 
                                    (events["time_bin"] > time_bin) &
                                    (events["period_id"] == period_id)].head(1)
                player_df2 = events[(events["player_id"] == player_id) & 
                                    (events["time_bin"] < time_bin) &
                                    (events["period_id"] == period_id)].tail(1) 
                # player_df1 = events[(events["player_id"] == player_id) & 
                #                     (events["time_bin"] > time_bin)].head(1)
                # player_df2 = events[(events["player_id"] == player_id) & 
                #                     (events["time_bin"] < time_bin)].tail(1) 
                player_df = pd.concat([player_df1, player_df2], axis=0) # maximum 2 rows, minimun 1  row

            # global features2: if a player does not have any event in the period, use a uniform distribution(p = 1 / (105 * 68) = 0.00014)
            if player_df.empty:
                prob = 1 / (bins[0] * bins[1])
                heatmap[player_id][(period_id, time_bin)] = np.full((bins[1], bins[0]), prob) # after off-ball contribution(only 2d matrix)
                #heatmap[player_id][time_bin] = np.full((bins[1], bins[0]), prob)
            else:
                bin_statistic = pitch.bin_statistic(player_df["start_x"].values, player_df["start_y"].values, 
                                    statistic="count", bins=bins, normalize=True)    
                heatmap[player_id][(period_id, time_bin)] = gaussian_filter(bin_statistic["statistic"], sigma=sigma) # 68 x 105
                #heatmap[player_id][time_bin] = gaussian_filter(bin_statistic["statistic"], sigma=sigma)

    heatmap_df = pd.DataFrame(index=events.index, columns=player_ids)
    for row in events.itertuples():
        for player_id in heatmap.keys():
            heatmap_df.at[row.Index, player_id] = heatmap[player_id][(row.period_id, row.time_bin)]
            #heatmap_df.at[row.Index, player_id] = heatmap[player_id][row.time_bin]
    return heatmap_df

def required_fields(fields):
    """필요한 컬럼이 있는지 검사하는 데코레이터"""
    def decorator(func):
        func.required_fields = fields
        return func
    return decorator

@no_type_check
def simple(actionfn: Callable):
    """Make a function decorator to apply action features to game states.

    Parameters
    ----------
    actionfn : Callable
        A feature transformer that operates on actions.

    Returns
    -------
    FeatureTransformer
        A feature transformer that operates on game states.
    """

    @wraps(actionfn)
    def _wrapper(events) -> pd.DataFrame:
        result_df = actionfn(events)
        return result_df

    return _wrapper

def get_player_events_x_data(events: pd.DataFrame, player_id):
    # 전체 x좌표
    event_x = events['start_x']

    # 전반/후반 분리
    events_FH = events[events['period_id'] == 1]
    events_SH = events[events['period_id'] == 2]
    FH_indexes = events_FH.index
    SH_indexes = events_SH.index

    # 각 선수의 등장 위치를 기반으로 fill
    prev_player_x_FH = pd.Series(np.where(events_FH['player_id'] == player_id, events_FH['start_x'], np.nan), index=FH_indexes).ffill().bfill()
    next_player_x_FH = pd.Series(np.where(events_FH['player_id'] == player_id, events_FH['start_x'], np.nan), index=FH_indexes).bfill().ffill()
    prev_player_x_SH = pd.Series(np.where(events_SH['player_id'] == player_id, events_SH['start_x'], np.nan), index=SH_indexes).ffill().bfill()
    next_player_x_SH = pd.Series(np.where(events_SH['player_id'] == player_id, events_SH['start_x'], np.nan), index=SH_indexes).bfill().ffill()

    # 전/후반 통합
    prev_player_x = pd.concat([prev_player_x_FH, prev_player_x_SH], axis=0)
    next_player_x = pd.concat([next_player_x_FH, next_player_x_SH], axis=0)

    # 평균 x좌표 계산 (선수 등장 시점 기준)
    appeared = events[events['player_id'] == player_id]
    avg_player_x = appeared['start_x'].mean()
    # 동일한 값을 모든 행에 넣음
    avg_player_x_col = pd.Series(avg_player_x, index=events.index)


    appeared_x = events.copy()
    appeared_x['is_target'] = appeared_x['player_id'] == player_id
    appeared_x['player_x'] = np.where(appeared_x['is_target'], appeared_x['start_x'], np.nan)
    prev_avg_x = appeared_x['player_x'].expanding().mean().shift(1).ffill().bfill()

    reversed_player_x = appeared_x['player_x'][::-1]
    next_avg_x = reversed_player_x.expanding().mean().shift(1)[::-1].bfill().ffill()

    related_x = events['related_x']
    
    return pd.DataFrame({
        'event_x': event_x,
        'prev_player_x': prev_player_x,
        'next_player_x': next_player_x,
        'avg_player_x': avg_player_x_col,
        'prev_avg_x': prev_avg_x,
        'next_avg_x': next_avg_x,
        'related_x' : related_x
    })



def get_player_events_y_data(events: pd.DataFrame, player_id):
    # 전체 y좌표
    event_y = events['start_y']
    # 전반/후반 분리
    events_FH = events[events['period_id']==1]
    events_SH = events[events['period_id']==2]
    FH_indexes = events_FH.index
    SH_indexes = events_SH.index
    # 각 선수의 등장 위치를 기반으로 fill
    prev_player_y_FH = pd.Series(np.where(events_FH['player_id'] == player_id, events_FH['start_y'], np.nan),index=FH_indexes).ffill().bfill()
    next_player_y_FH = pd.Series(np.where(events_FH['player_id'] == player_id, events_FH['start_y'], np.nan),index=FH_indexes).bfill().ffill()
    prev_player_y_SH = pd.Series(np.where(events_SH['player_id'] == player_id, events_SH['start_y'], np.nan),index=SH_indexes).ffill().bfill()
    next_player_y_SH = pd.Series(np.where(events_SH['player_id'] == player_id, events_SH['start_y'], np.nan),index=SH_indexes).bfill().ffill()
    
    # 전/후반 통합
    prev_player_y = pd.concat([prev_player_y_FH,prev_player_y_SH],axis=0)
    next_player_y = pd.concat([next_player_y_FH,next_player_y_SH],axis=0)

    # 평균 y좌표 계산 (선수 등장 시점 기준)
    appeared = events[events['player_id'] == player_id]
    avg_player_y = appeared['start_y'].mean()

    # 동일한 값을 모든 행에 넣음
    avg_player_y_col = pd.Series(avg_player_y, index=events.index)

    appeared_y = events.copy()
    appeared_y['is_target'] = appeared_y['player_id'] == player_id
    appeared_y['player_y'] = np.where(appeared_y['is_target'], appeared_y['start_y'], np.nan)
    prev_avg_y = appeared_y['player_y'].expanding().mean().shift(1).ffill().bfill()

    reversed_player_y = appeared_y['player_y'][::-1]
    next_avg_y = reversed_player_y.expanding().mean().shift(1)[::-1].bfill().ffill()
    related_y = events['related_y']

    return pd.DataFrame({
        'event_y': event_y,
        'prev_player_y': prev_player_y,
        'next_player_y':next_player_y,
        'avg_player_y': avg_player_y_col,
        'prev_avg_y': prev_avg_y,
        'next_avg_y': next_avg_y,
        'related_y': related_y
    })

def get_player_categorical_data(events: pd.DataFrame, player_id, teams: pd.DataFrame):
    position_str = teams[teams['player_id'] == player_id]['position'].values[0]
    position = config.PLAYER_ROLE_MAPPING.get(position_str, 0)  # 알 수 없는 포지션은 0 처리
    str_event_type = events['type_name']
    # 문자열 → 숫자 매핑
    encoded_event_type = str_event_type.map(EVENT_LABEL_MAPPING).fillna(0).astype(int)
    event_time = events['time_seconds']
    player_team = teams[teams['player_id'] == player_id]['team'].values[0]    
    # 해당 시점 이벤트의 팀이 player_team과 같은지
    team_on_ball = events['team'] == player_team    
    player_on_ball = pd.Series(np.where(events['player_id'] == player_id, True, False))
    return pd.DataFrame({
        'event_time':event_time,
        'player_id':player_id,
        'position':position,
        'event_type':encoded_event_type,
        'team_on_ball':team_on_ball, 
        'player_on_ball':player_on_ball
    })

@required_fields(["player_id", "time_seconds", "type_name"])
def prevAgentTime(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    각 선수별 마지막 on-ball 이벤트 관측 시간과 현재 시간 차이를 계산하는 함수

    Parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    Returns
    -------
    pd.DataFrame
        선수별 마지막 관측 시간과 현재 시간 차이를 포함한 데이터프레임
    """
    agent_ids=events["player_id"].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들

    current_period = None
    time_diffs_dict = {}  
    last_seen = {}  

    for idx, row in events.iterrows():
        current_pID = row["player_id"]
        current_time = row["time_seconds"]
        type_name = row["type_name"]
        period = row["period_id"]

        if period != current_period:
            current_period = period
            last_seen = {}
        # 현재 선수의 마지막 관측 시간과의 차이 계산
        time_diffs = {
            pid: 0 if pid == current_pID else (current_time - last_seen[pid] if pid in last_seen else None) 
            for pid in agent_ids
        }

        # 현재 선수가 공을 소유한 이벤트라면 마지막 관측 시간 업데이트
        if pd.notna(current_pID):
            last_seen[current_pID] = current_time

        time_diffs_dict[idx] = time_diffs  # 결과 저장

    # 데이터프레임 변환 및 NaN → 0으로 채우기
    df = pd.DataFrame.from_dict(time_diffs_dict, orient="index").ffill().bfill()
    df = df.reindex(columns=agent_ids.tolist()+list(added_ids)).astype("float")
    return df


@required_fields(["player_id", "start_x", "type_name"])
def prevAgentX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    prev_player_x_dict = {}

    for pid in player_ids:
        player_df = get_player_events_x_data(events, pid)
        prev_player_x_dict[pid] = player_df['prev_player_x']
    prev_player_x_df = pd.DataFrame(prev_player_x_dict)
    return prev_player_x_df


@required_fields(["player_id", "start_y", "type_name"])
def prevAgentY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    prev_player_y_dict = {}

    for pid in player_ids:
        player_df = get_player_events_y_data(events, pid)
        prev_player_y_dict[pid] = player_df['prev_player_y']
    prev_player_y_df = pd.DataFrame(prev_player_y_dict)
    return prev_player_y_df


@required_fields(["player_id", "time_seconds", "type_name"])
def nextAgentTime(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    선수별 다음 관측까지의 시간 차이 (nextAgentTime)를 계산하는 함수.

    Parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터 (각 선수의 이벤트 포함)

    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
    result_df : pd.DataFrame
        선수별 다음 관측까지의 시간 차이를 저장한 DataFrame
    """

    last_seen_time = {}  
    last_seen_idx = {}  
    next_time_diff_dict = {
        idx: {pid: 0 for pid in events["player_id"].dropna().unique()} 
        for idx in events.index
    }

    # 이벤트에 등장한 선수 리스트
    agent_ids = events["player_id"].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들


    for idx, row in events.iterrows():  
        current_pID = row["player_id"]
        current_time = row["time_seconds"]  
        event_type = row["type_name"]
        period = row["period_id"]        
        key = (period, current_pID)
        if key in last_seen_idx:
            prev_idx = last_seen_idx[key]
            for fill_idx in range(prev_idx, idx + 1):  # 같은 피리어드 내에서만
                if events.loc[fill_idx, "period_id"] != period:
                    continue
                time_diff = current_time - events.loc[fill_idx, "time_seconds"]
                next_time_diff_dict[fill_idx][current_pID] = time_diff
        else:
            for fill_idx in range(0, idx + 1):
                if events.loc[fill_idx, "period_id"] != period:
                    continue
                time_diff = current_time - events.loc[fill_idx, "time_seconds"]
                next_time_diff_dict[fill_idx][current_pID] = time_diff

        last_seen_time[key] = current_time
        last_seen_idx[key] = idx

    # ✅ 마지막 stretch 보완
    for pid in agent_ids:
        for period in events["period_id"].unique():
            key = (period, pid)
            if key in last_seen_idx:
                last_idx = last_seen_idx[key]
                if last_idx > 0:
                    last_value = next_time_diff_dict[last_idx - 1][pid]
                else:
                    last_value = 0

                for idx in range(last_idx, events.index[-1] + 1):
                    if events.loc[idx, "period_id"] != period:
                        continue
                    next_time_diff_dict[idx][pid] = last_value

    result_df = pd.DataFrame.from_dict(next_time_diff_dict, orient="index")
    result_df[agent_ids] = result_df[agent_ids].fillna(0)
    result_df = result_df.reindex(columns=agent_ids.tolist() + list(added_ids))
    return result_df

@required_fields(["player_id", "start_x", "type_name"])
def nextAgentX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    next_player_x_dict = {}
    for pid in player_ids:
        player_df = get_player_events_x_data(events, pid)
        next_player_x_dict[pid] = player_df['next_player_x']
    next_player_x_df = pd.DataFrame(next_player_x_dict)
    return next_player_x_df

@required_fields(["player_id", "start_y", "type_name"])
def nextAgentY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    next_player_y_dict = {}
    for pid in player_ids:
        player_df = get_player_events_y_data(events, pid)
        next_player_y_dict[pid] = player_df['next_player_y']
    next_player_y_df = pd.DataFrame(next_player_y_dict)
    return next_player_y_df

@required_fields(["player_id", "start_x", "type_name"])
def avgAgentX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    경기에 출전하지 않은 선수는 NaN값 할당
    '''
    
    player_ids = events['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들

    avg_player_x_dict = {}
    for pid in player_ids:
        player_df = get_player_events_x_data(events, pid)
        avg_player_x_dict[pid] = player_df['avg_player_x']

    for pid in added_ids:
        avg_player_x_dict[pid] = pd.Series(np.nan * len(events), index=events.index)

    avg_player_x_df = pd.DataFrame(avg_player_x_dict)
    return avg_player_x_df

@required_fields(["player_id", "start_y", "type_name"])
def avgAgentY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들

    avg_player_y_dict = {}
    for pid in player_ids:
        player_df = get_player_events_y_data(events, pid)
        avg_player_y_dict[pid] = player_df['avg_player_y']

    for pid in added_ids:
        avg_player_y_dict[pid] = pd.Series(np.nan * len(events), index=events.index)

    avg_player_y_df = pd.DataFrame(avg_player_y_dict)
    return avg_player_y_df

@required_fields(["player_id", "team", "type_name"])
#categorical
def agentSide(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    공을 소유한 선수의 팀 여부를 0, 1, 2로 표현하는 데이터프레임 생성.

    Parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
    agent_side_df : pd.DataFrame
        공을 소유한 선수의 팀 여부 (0: 뛰지 않음, 1: 같은 팀, 2: 상대 팀)
    """
    player_ids = events['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(player_ids)
    agent_side_dict = {}
    for pid in player_ids:
        player_df = get_player_categorical_data(events, pid, teams)
        agent_side_dict[pid] = player_df['team_on_ball'].map({True: 1, False: 2})

    for pid in added_ids:
        agent_side_dict[pid] = pd.Series([0] * len(events), index=events.index)

    agent_side_df = pd.DataFrame(agent_side_dict)
    return agent_side_df

@required_fields(["player_id"])
def agentRole(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        선수별 역할을 저장한 데이터프레임(0번은 None, 1번부터 포지션)
    '''
    player_ids = events['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(player_ids)

    agent_role_dict = {}
    for pid in player_ids:
        player_df = get_player_categorical_data(events, pid, teams)
        agent_role_dict[pid] = player_df['position']

    for pid in added_ids:
        agent_role_dict[pid] = pd.Series([0] * len(events), index=events.index)
    agent_role_df = pd.DataFrame(agent_role_dict)
    return agent_role_df


@required_fields(["player_id", "type_name"])
#categorical
def agentObserved(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    이벤트별 선수의 on-ball 여부를 나타내는 데이터프레임 생성.

    Parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터 (player_id, type_name 포함)

    Returns
    -------
    on_ball_df : pd.DataFrame
        선수별 on-ball 여부 (0: 뛰지 않음, 1: 공 소유, 2: 공 비소유)
    """
    player_ids = events['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    agent_observed_dict = {}
    for pid in player_ids:
        player_df = get_player_categorical_data(events, pid, teams)
        series = player_df['player_on_ball'].map({True: 1, False: 2})
        agent_observed_dict[pid] = series
    for pid in added_ids:
        agent_observed_dict[pid] = pd.Series([0] * len(events), index=events.index)

    agent_observed_df = pd.DataFrame(agent_observed_dict)
    return agent_observed_df

@required_fields(["player_id", "team", "type_name", "time_seconds"])
def goalDiff(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    이벤트별 선수의 현재 경기 골 차이를 계산. (뛰지 않은 선수는 NaN)

    Parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터 (player_id, type_name, team, time_seconds 포함)
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
    goal_diff_df : pd.DataFrame
        선수별 현재 경기 골 차이 (Home 팀 선수는 home_score - away_score, Away 팀 선수는 반대)
    """
    events = events.copy()
    
    # BEPRO버전
    # is_goal_conceded = (
    #     (events['type_id'] == config.actiontypes.index("Save")) & 
    #     (events['result_id'] == config.results.index("fail"))
    # )

    # DFL버전
    is_goal_conceded =(
        (events['type_name'] == "KickOff_Play_Pass") &
        (events['time_seconds'] != 0)
    )
    # 팀별 골득실 누적 시계열 계산
    team_ids = teams['team_id'].unique()
    team_goal_diff = {}

    for team_id in team_ids:
        gc = (is_goal_conceded & (events['team_id'] == team_id)).cumsum()
        gs = (is_goal_conceded & (events['team_id'] != team_id)).cumsum()
        goal_diff = gs - gc
        team_goal_diff[team_id] = goal_diff

    # player_id 기준 시리즈 생성
    player_goal_diff = {}

    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    for _, row in teams.iterrows():
        pid = row['player_id']
        tid = row['team_id']
        player_goal_diff[pid] = team_goal_diff[tid]
    for pid in added_ids:
        player_goal_diff[pid] = pd.Series(np.nan * len(events), index=events.index)

    return pd.DataFrame(player_goal_diff)

@required_fields(["start_x"])
def eventX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트의 x좌표를 선수 전체에게 저장한 데이터프레임
        (이벤트에 등장하지 않은 선수는 nan으로 채움)
    '''
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    event_x_dict = {}
    for pid in player_ids:
        player_df = get_player_events_x_data(events, pid)
        event_x_dict[pid] = player_df['event_x']
    for pid in added_ids:
        event_x_dict[pid] = pd.Series(np.nan, index=events.index)
    event_x_df = pd.DataFrame(event_x_dict)
    return event_x_df

@required_fields(["start_y"])
def eventY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트의 y좌표를 선수 전체에게 저장한 데이터프레임
        (이벤트에 등장하지 않은 선수는 nan으로 채움)
    '''
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    event_y_dict = {}
    for pid in player_ids:
        player_df = get_player_events_y_data(events, pid)
        event_y_dict[pid] = player_df['event_y']
    for pid in added_ids:
        event_y_dict[pid] = pd.Series(np.nan, index=events.index)
    event_y_df = pd.DataFrame(event_y_dict)
    return event_y_df

@required_fields(["type_name"])
def eventType(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트의 유형을 선수 전체에게 저장한 데이터프레임
        (경기에 뛰지 않은 선수는 0으로 채움)
    '''
    player_ids = events['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(player_ids)
    event_type_dict = {}
    for pid in player_ids:
        player_df = get_player_categorical_data(events, pid, teams)
        event_type_dict[pid] = player_df['event_type']
    for pid in added_ids:
        event_type_dict[pid] = pd.Series([0] * len(events), index=events.index)
    event_type_df = pd.DataFrame(event_type_dict)
    return event_type_df

def prevAvgX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트 직전까지의 관측된 평균 x좌표
        (경기에 뛰지 않은 선수는 nan으로 채움)
    '''
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    prev_avg_x_dict = {}

    for pid in player_ids:
        player_df = get_player_events_x_data(events, pid)
        prev_avg_x_dict[pid] = player_df['prev_avg_x']

    for pid in added_ids:
        prev_avg_x_dict[pid] = pd.Series(np.nan, index=events.index)

    prev_avg_x_df = pd.DataFrame(prev_avg_x_dict)
    return prev_avg_x_df

def prevAvgY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트 직전까지의 관측된 평균 y좌표
        (경기에 뛰지 않은 선수는 nan으로 채움)
    '''
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    prev_avg_y_dict = {}

    for pid in player_ids:
        player_df = get_player_events_y_data(events, pid)
        prev_avg_y_dict[pid] = player_df['prev_avg_y']

    for pid in added_ids:
        prev_avg_y_dict[pid] = pd.Series(np.nan, index=events.index)

    prev_avg_y_df = pd.DataFrame(prev_avg_y_dict)
    return prev_avg_y_df

def nextAvgX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트 이후부터의 관측된 평균 x좌표
        (경기에 뛰지 않은 선수는 nan으로 채움)
    '''
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    next_avg_x_dict = {}

    for pid in player_ids:
        player_df = get_player_events_x_data(events, pid)
        next_avg_x_dict[pid] = player_df['next_avg_x']

    for pid in added_ids:
        next_avg_x_dict[pid] = pd.Series(np.nan, index=events.index)

    next_avg_x_df = pd.DataFrame(next_avg_x_dict)
    return next_avg_x_df

def nextAvgY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        해당 이벤트 이후부터의 관측된 평균 y좌표
        (경기에 뛰지 않은 선수는 nan으로 채움)
    '''
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    next_avg_y_dict = {}

    for pid in player_ids:
        player_df = get_player_events_y_data(events, pid)
        next_avg_y_dict[pid] = player_df['next_avg_y']

    for pid in added_ids:
        next_avg_y_dict[pid] = pd.Series(np.nan, index=events.index)

    next_avg_y_df = pd.DataFrame(next_avg_y_dict)
    return next_avg_y_df


def possessRatio(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        이벤트 당시 기준으로 선수의 소속팀의 평균 점유율을 계산하는 함수.
    '''
    home_possession = 0.0
    away_possession = 0.0
    last_time = None
    last_team = None

    home_ratio_list = []
    away_ratio_list = []

    pending_start_time = None  # 델타 계산을 보류할 때 필요한 변수
    valid_events = events[events["time_seconds"] >= 0].copy()
    last_period=events.loc[0,'period_id']
    player_team_map = teams.set_index("player_id")["team"].to_dict()
    agent_ids = events["player_id"].dropna().unique()

    added_ids = set(teams["player_id"].unique()) - set(agent_ids)
    possession_by_player = {}
    for i, (idx, row) in enumerate(valid_events.iterrows()):
        curr_time = row["time_seconds"]
        curr_team = row["team"]
        event_type = row["type_name"]
        cur_period= row["period_id"]

        if cur_period!=last_period:
            # home_possession = 0.0
            # away_possession = 0.0
            pending_start_time = None
            last_team = None
        last_period = cur_period  # 마지막 period 업데이트
        # 이전에 누가 점유하고 있었는지에 따라 possession time 누적
        if event_type in on_ball_actions:
            if last_team in ["Home", "Away"] and pending_start_time is not None:
                delta_time = curr_time - pending_start_time
                if last_team == "Home":
                    home_possession += delta_time
                elif last_team == "Away":
                    away_possession += delta_time

            last_team = curr_team
            pending_start_time = curr_time

        # 해당 시점까지 점유율 계산
        total_time = home_possession + away_possession
        if total_time > 0:
            home_ratio = home_possession / total_time
            away_ratio = away_possession / total_time
        else:
            home_ratio = 0.5  # 초기엔 50:50으로 가정
            away_ratio = 0.5

        home_ratio_list.append(home_ratio)
        away_ratio_list.append(away_ratio)

        # 마지막 점유 팀 업데이트
        if event_type in on_ball_actions:
            last_team = curr_team
            last_time = curr_time
        row_dict = {}
        for pid in agent_ids:
            team = player_team_map.get(pid, None)
            if team == "Home":
                row_dict[pid] = home_ratio
            elif team == "Away":
                row_dict[pid] = away_ratio
            else:
                row_dict[pid] = None
        possession_by_player[idx] = row_dict

    possession_df = pd.DataFrame.from_dict(possession_by_player, orient="index")
    possession_df[agent_ids] = possession_df[agent_ids].fillna(0.0)

    # reindex로 added_ids 추가 (→ NaN으로 들어감)
    possession_df = possession_df.reindex(columns=agent_ids.tolist() + list(added_ids))
    return possession_df

def elapsedTime(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        직전 이벤트와 현재 이벤트 간의 시간 차이를 계산
        직전 이벤트를 어느정도 하고 현재 넘어왔는지 계산
    '''
    agent_ids = events["player_id"].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(agent_ids)
    all_ids = list(agent_ids) + list(added_ids)

    # 이벤트 간 시간 차이 계산
    delta_times = events["time_seconds"].diff().fillna(0).values

    # 등장한 선수만 delta_time 입력
    elapsed_time_df = pd.DataFrame({pid: delta_times for pid in agent_ids}, index=events.index)

    # 등장하지 않은 선수들은 NaN으로 초기화
    elapsed_time_df = elapsed_time_df.reindex(columns=list(agent_ids) + list(added_ids))
    return elapsed_time_df

def observeEventXY(events: pd.DataFrame, teams: pd.DataFrame):
    '''
    parameters
    ----------
    events : pd.DataFrame
        이벤트 데이터
    teams : pd.DataFrame
        팀 데이터
    Returns
    -------
        관측된 선수의 좌표를 저장한 DataFrame       
        각 이벤트에서 actor와 recipient는 실제 좌표(start_x/y, end_x/y)를,
        나머지 선수들은 0,0을 사용.
    '''

    agent_ids = events["player_id"].dropna().unique()
    added_ids = set(teams["player_id"].unique()) - set(agent_ids)
    all_ids = list(agent_ids) + list(added_ids)
 
    obs_x = []
    obs_y = []

    for row in events.itertuples():
        obs_x_dict = {pid: np.nan if pid in added_ids else 0.0 for pid in all_ids}
        obs_y_dict = {pid: np.nan if pid in added_ids else 0.0 for pid in all_ids}

        actor_id = row.player_id
        related_id = row.related_id
        start_x, start_y = row.start_x, row.start_y
        related_x, related_y = row.related_x, row.related_y

        if pd.notna(actor_id):
            obs_x_dict[actor_id] = start_x
            obs_y_dict[actor_id] = start_y


        if pd.notna(related_id):
            obs_x_dict[related_id] = related_x
            obs_y_dict[related_id] = related_y

        obs_x.append(obs_x_dict)
        obs_y.append(obs_y_dict)

    obs_x_df = pd.DataFrame(obs_x)
    obs_y_df = pd.DataFrame(obs_y)

    return obs_x_df, obs_y_df
    
def observeEventX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    observe_x, _ = observeEventXY(events, teams)
    return observe_x

def observeEventY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    _, observe_y = observeEventXY(events, teams)
    return observe_y

@required_fields(["freeze_frame"])
def freeze_frame(events: pd.DataFrame, teams: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """
    freeze frame을 feature로 변환하는 함수.
    """
    team_lookup = {
        pid: (team, pos, str(int(xid)).zfill(2))
        for pid, team, pos, xid in teams[["player_id", "team", "position", "xID"]].values
    }
    # 임시버전: 팀 정보가 없음
    # team_lookup = {
    #     pid: (team, str(int(xid)).zfill(2))
    #     for pid, team, xid in teams[["player_id", "team", "xID"]].values
    # }
    period_tracking_dict = {
        period_id: df.reset_index(drop=True)
        for period_id, df in positions.groupby("period_id")
    }

    agent_ids = teams["player_id"].dropna().unique()
    agent_vector = []

    for row in events.itertuples():
        copy_tracking_data = period_tracking_dict[row.period_id] # time기반으로 필터링하면 전/후반 두개의 tracking-data가 추출되기 때문에
        closest_idx = copy_tracking_data["time"].sub(row.time_seconds).abs().idxmin()
        
        freeze_frame_lst = []
        for pid in agent_ids:
            team_info, position_info, xid = team_lookup[pid]
            key = f"H{xid}" if team_info == "Home" else f"A{xid}"
           
            freeze_frame_lst.append([
                key,  # sorted by pID
                copy_tracking_data.at[closest_idx, key + "_x"],
                copy_tracking_data.at[closest_idx, key + "_y"],
                pid == row.player_id,  # actor
                team_info == row.team, # teammate
                position_info == "GK"  # goalkeeper
            ]) 

        agent_vector.append(freeze_frame_lst)

    events["freeze_frame"] = agent_vector
    return events[["freeze_frame"]] 

def get_player_vaep_data(events: pd.DataFrame):
    _goal_x: float = config.field_length
    _goal_y: float = config.field_width / 2

    dx = (_goal_x - events["start_x"]).abs().values
    dy = (_goal_y - events["start_y"]).abs().values
    start_dist_to_goal = np.sqrt(dx**2 + dy**2)

    angle_raw = np.divide(dy, dx, out=np.zeros_like(dy), where=dx!=0)
    start_angle_to_goal = np.arctan(angle_raw)
    #start_angle_to_goal = np.nan_to_num(np.arctan(dy / dx))

    dx = (_goal_x - events["end_x"]).abs().values
    dy = (_goal_y - events["end_y"]).abs().values
    end_dist_to_goal = np.sqrt(dx**2 + dy**2)
    angle_raw = np.divide(dy, dx, out=np.zeros_like(dy), where=dx!=0)
    end_angle_to_goal = np.arctan(angle_raw)
    # end_angle_to_goal = np.nan_to_num(np.arctan(dy / dx))

    return pd.DataFrame({
        'results': events.result_name.apply(lambda r: config.results.index(r)+1), # 0은 패딩 값
        'bodyparts': events.bodypart_name.apply(lambda b: config.bodyparts.index(b)+1),
        'end_x': events.end_x,
        'end_y': events.end_y,
        'period_id': events.period_id, # 1: 전반, 2: 후반
        'start_dist_to_goal': start_dist_to_goal,
        'start_angle_to_goal': start_angle_to_goal,
        'end_dist_to_goal': end_dist_to_goal,
        'end_angle_to_goal': end_angle_to_goal,
    })

def results(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids=set(teams['player_id'].unique())-set(events['player_id'].unique()) #events에 기록이 없는 team_sheets의 선수들 
    
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['results']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    result_df = pd.DataFrame(result_dict)
    return result_df

def bodyparts(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['bodyparts']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def period(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['period_id']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def endX(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['end_x']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def endY(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['end_y']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def startdisttogoal(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['start_dist_to_goal']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def startangletogoal(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['start_angle_to_goal']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def enddisttogoal(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['end_dist_to_goal']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

def endangletogoal(events: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    player_ids = teams['player_id'].dropna().unique()
    added_ids = set(teams['player_id'].unique()) - set(events['player_id'].unique())
    result_dict = {}

    for pid in player_ids:
        player_df = get_player_vaep_data(events)
        result_dict[pid] = player_df['end_angle_to_goal']

    for pid in added_ids:
        result_dict[pid] = pd.Series(np.nan, index=events.index)

    return pd.DataFrame(result_dict)

if __name__=="__main__":
    path = os.path.join(os.getcwd(), "./data/DFL")
    match_ids = [extract_match_id(filename) for filename in os.listdir(path) if filename.startswith("DFL")]
    all_events, unique_value_list=[], []

    match_path = os.path.join(path, match_ids[9])  
    teams_path = os.path.join(match_path, "teams.csv")  
    events_path = os.path.join(match_path, "events.csv")  
    postions_path = os.path.join(match_path, "positions.csv")

    teams = pd.read_csv(teams_path)
    events = pd.read_csv(events_path)
    positions = pd.read_csv(postions_path)

    goal_diff=goalDiff(events,teams)
    agent_side=agentSide(events,teams)
    frame= freeze_frame(events, positions,teams)

    print(frame)
    print(frame.iloc[0])
    
    # # ✅ match_id 별로 파일 불러오기
    # for match_id in match_ids:
    #     match_path = os.path.join(path, match_id)  # ✅ match_id를 path와 결합
    #     print(match_path)
    #     teams_path = os.path.join(match_path, "teams.csv")  # ✅ teams.csv 경로 설정
    #     events_path = os.path.join(match_path, "events.csv")  # ✅ events.csv 경로 설정

    #     if os.path.exists(teams_path) and os.path.exists(events_path):  # ✅ 파일이 존재하는지 확인
    #         teams = pd.read_csv(teams_path)
    #         events = pd.read_csv(events_path)
    #         print(f"✅ Loaded data for match {match_id}")
    #         goal_diff=goalDiff(events, teams)
    #         #event_type=eventType(events, teams)
    #         print(goal_diff.shape)
    #         all_events.append(goal_diff)
    #         unique_value_list.append(set(goal_diff.values.flatten()))
    #     else:
    #         print(f"❌ Missing files for match {match_id}")
    
    # df_goal_diff = pd.concat(all_events, ignore_index=True)

    # # ✅ 전체 유니크 값 개수 계산 (🔥 여기서 set을 이용해 중복 제거)
    # total_unique_values = set().union(*unique_value_list)  # 🔥 리스트에 있는 여러 set을 하나로 합침
    # total_unique_count = len(total_unique_values)

    #     # ✅ 결과 출력
    # print(f"✅ 전체 데이터프레임에서 유니크한 값 개수: {total_unique_count}")
    # print(total_unique_values)


       