"""
Code adapted from Friends of Tracking Github - https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.git
"""
import os
import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(base_path)
sys.path.append(base_path)
import numpy as np
import pandas as pd
from datatools.visualization import plot_frame
from numba import njit, prange
from imputer.config import team_colors
from matplotlib.colors import LinearSegmentedColormap

@njit(parallel=True, fastmath=True)
def compute_ppcf_grid(xgrid, ygrid,
                      att_pos, att_vel, att_is_gk, att_lambda,   # (Na,2), (Na,2), (Na,), (Na,)
                      def_pos, def_vel, def_is_gk, def_lambda,   # (Nd,2), ...
                      ball_xy, average_ball_speed,
                      reaction_time, tti_sigma,
                      time_to_control_att, time_to_control_def,
                      int_dt, max_int_time, converge_tol):
    ny = ygrid.shape[0]
    nx = xgrid.shape[0]
    PPCFa = np.zeros((ny, nx), dtype=np.float32)
    PPCFd = np.zeros((ny, nx), dtype=np.float32)

    for iy in prange(ny):
        for ix in range(nx):
            tx = xgrid[ix]
            ty = ygrid[iy]
            # --- ball travel time
            dx = tx - ball_xy[0]
            dy = ty - ball_xy[1]
            if np.isnan(ball_xy[0]) or np.isnan(ball_xy[1]):
                ball_t = 0.0
            else:
                ball_t = np.sqrt(dx*dx + dy*dy) / average_ball_speed

            # --- time-to-intercept (최소값만 먼저 계산)
            tau_min_att = 1e9
            tau_min_def = 1e9

            # 반응 지점(= r_reaction) 미리 계산: pos + vel * rt
            # → 격자마다 바뀌는 건 타깃만이라 r_reaction은 상수화 효과 (루프 내에서 곱만 1회)
            # 여기선 간단히 인라인 (메모리/캐시 비용 고려해도 이득)

            for k in range(att_pos.shape[0]):
                rx = att_pos[k,0] + att_vel[k,0]*reaction_time
                ry = att_pos[k,1] + att_vel[k,1]*reaction_time
                tt = reaction_time + np.sqrt((tx-rx)**2 + (ty-ry)**2) / 5.0  # vmax=5.0 가정; 전달 인자로 교체 가능
                if tt < tau_min_att: tau_min_att = tt

            for k in range(def_pos.shape[0]):
                rx = def_pos[k,0] + def_vel[k,0]*reaction_time
                ry = def_pos[k,1] + def_vel[k,1]*reaction_time
                tt = reaction_time + np.sqrt((tx-rx)**2 + (ty-ry)**2) / 5.0
                if tt < tau_min_def: tau_min_def = tt

            # --- 쇼트컷 (원문과 동일 로직)
            if tau_min_att - max(ball_t, tau_min_def) >= time_to_control_def:
                PPCFa[iy, ix] = 0.0
                PPCFd[iy, ix] = 1.0
                continue
            if tau_min_def - max(ball_t, tau_min_att) >= time_to_control_att:
                PPCFa[iy, ix] = 1.0
                PPCFd[iy, ix] = 0.0
                continue

            # --- 프루닝: tau_min과의 차이가 큰 선수는 제외
            # 허용 오차 내 선수만 리스트업 (동적 리스트는 느리므로 2패스: 카운트 → 배열 할당 → 채우기)
            tol_att = time_to_control_att
            tol_def = time_to_control_def

            # 적분 초기화
            ptot = 0.0
            PPCFatt = 0.0
            PPCFdef = 0.0

            # 적분 루프
            T = ball_t
            maxT = ball_t + max_int_time
            # 시그모이드 상수 (성능 위해 미리)
            sig_c = -np.pi/np.sqrt(3.0)/tti_sigma

            while (1.0 - ptot) > converge_tol and T < maxT:
                # 공격
                for k in range(att_pos.shape[0]):
                    rx = att_pos[k,0] + att_vel[k,0]*reaction_time
                    ry = att_pos[k,1] + att_vel[k,1]*reaction_time
                    tt = reaction_time + np.sqrt((tx-rx)**2 + (ty-ry)**2) / 5.0
                    if tt - tau_min_att >= tol_att:
                        continue
                    f = 1.0/(1.0 + np.exp(sig_c * (T - tt)))
                    dP = (1.0 - PPCFatt - PPCFdef) * f * att_lambda[k]
                    if dP < 0.0: dP = 0.0
                    PPCFatt += dP*int_dt

                # 수비
                for k in range(def_pos.shape[0]):
                    rx = def_pos[k,0] + def_vel[k,0]*reaction_time
                    ry = def_pos[k,1] + def_vel[k,1]*reaction_time
                    tt = reaction_time + np.sqrt((tx-rx)**2 + (ty-ry)**2) / 5.0
                    if tt - tau_min_def >= tol_def:
                        continue
                    f = 1.0/(1.0 + np.exp(sig_c * (T - tt)))
                    lam = def_lambda[k] * (3.0 if def_is_gk[k] else 1.0)
                    dP = (1.0 - PPCFatt - PPCFdef) * f * lam
                    if dP < 0.0: dP = 0.0
                    PPCFdef += dP*int_dt

                ptot = PPCFatt + PPCFdef
                T += int_dt

            PPCFa[iy, ix] = PPCFatt
            PPCFd[iy, ix] = PPCFdef

    return PPCFa, PPCFd

# 전처리 단계에서 한 번만:
def to_arrays(players, is_attacking_team: bool):
    pos = []
    vel = []
    is_gk = []
    lam = []
    for p in players:
        pos.append(p.position.astype(np.float32))
        vel.append(p.velocity.astype(np.float32))
        is_gk.append(1 if p.is_gk else 0)
        
        # 💥 수정: is_attacking_team 값에 따라 람다 값을 올바르게 할당
        if is_attacking_team:
            lam.append(np.float32(p.lambda_att))
        else: # 수비팀일 경우
            lam.append(np.float32(p.lambda_def))
            
    return (np.asarray(pos, np.float32),
            np.asarray(vel, np.float32),
            np.asarray(is_gk, np.int32),
            np.asarray(lam, np.float32))


#Pass in players for attacking/defending team, their team name, and params
def initialise_players(team_df,teamname,params, att, event):
    team_players = []
    for i,p in team_df.iterrows():
        if att == True:
            team_player = player(p,teamname,params,p['position'], False)
        else:
            team_player = player(p,teamname,params,p['position'], False)
        if team_player.inframe:
            team_players.append(team_player)
    return team_players

#Plots pitch control for an event
def plot_pitchcontrol_for_event(PPCF, action, locs, attacking_team,
                                home_color='blue',   # 홈팀 색상 인자 추가 (기본값 'red')
                                away_color='red',  # 원정팀 색상 인자 추가 (기본값 'blue')
                                alpha = 0.9, Pitch_Control=True, 
                                include_player_velocities=True, annotate=False, field_dimen = (105.0,68), ax=None):
    if attacking_team == 'H':   # 홈팀이 공격 중
        att_color, def_color = home_color, away_color
    else:                       # 어웨이팀이 공격 중
        att_color, def_color = away_color, home_color

    if ax is None:
        fig, ax = plot_frame(action, locs, att_color, def_color)  # 기존 함수가 fig,ax 생성
    else:
        fig = ax.figure

    if Pitch_Control == True:
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', 
            [def_color, 'white', att_color]
        )
        pc = ax.imshow(np.flipud(PPCF), extent=(0, field_dimen[0], 0, field_dimen[1]),vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
        
        for _, player in locs.iterrows():
            if 'key' in player and pd.notna(player['key']):
                ax.text(
                    player['x'],      # 텍스트의 x좌표
                    player['y'],      # 텍스트의 y좌표
                    player['key'],    # 표시할 텍스트 (선수 ID)
                    fontsize=7,
                    color='white',
                    fontweight='bold',
                    ha='center',      # 수평 중앙 정렬
                    va='center'       # 수직 중앙 정렬
                )

        cbar = fig.colorbar(pc,orientation="horizontal")
        tick_font_size = 25
        cbar.ax.tick_params(labelsize=tick_font_size)
    return fig,ax
    
#Creates a player class
class player(object):
    # Pass in the line for that player (from 360 data), their team name(e.g. home/away), default params and if they are goalkeeper or not
    def __init__(self,team,teamname,params,GK, off):
        if GK == 'GK':#'GK':
            self.is_gk = True
        else:
            self.is_gk = False
        self.is_actor = False
        self.teamname = teamname
        self.accel = params['max_player_accel']
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params['lambda_gk'] if self.is_gk else params['lambda_def'] # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(team)
        self.get_velocity(team)
        self.offside = off
        self.PPCF = 0. # initialise this for later
        
    def get_position(self,team):
        self.position = np.array( [ team['x'], team['y'] ] )
        self.inframe = not np.any( np.isnan(self.position) )
    
    # # 속도가 0인 버전
    # def get_velocity(self,team):
    #     self.velocity = np.array( [ 0, 0 ] )
    #     if np.any( np.isnan(self.velocity) ):
    #         self.velocity = np.array([0.,0.])
    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity*self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
        return self.time_to_intercept

    # 속도를 고려한 버전
    def get_velocity(self,team):
        self.velocity = np.array( [ team['vx'], team['vy']] )

    # def simple_time_to_intercept(self, r_final):
    #     self.PPCF = 0. 
    #     dist_to_loc = np.linalg.norm(r_final-self.position)
    #     time_to_reach_terminal_velocity = self.vmax / self.accel
    #     dist_reached_before_TV = 0.5 * self.accel * (time_to_reach_terminal_velocity ** 2)
        
    #     if dist_to_loc < dist_reached_before_TV:
    #         time = np.sqrt((2*dist_to_loc)/self.accel)
    #     else:
    #         time = time_to_reach_terminal_velocity + ((dist_to_loc - dist_reached_before_TV) / self.vmax)
        
    #     self.time_to_intercept = time
    #     return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f
    
#Sets default Pitch Control parameters
def default_model_params(time_to_control_veto=3):
    
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 5. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1 # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att']) 
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params

#Calculates pitch control probability for attacking and defending team at a given target position
def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """ 
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
     
    tti = [p.simple_time_to_intercept(target_position) for p in attacking_players]
    ttca = params['time_to_control_att']
    if attacking_players[tti.index(np.nanmin(tti))].is_actor == True:
        ttca = params['time_to_control_att']
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= ttca:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in attacking_players:
                if player.is_actor:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * (player.lambda_att)
                else:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # calculate ball control probablity for 'player' in time interval T+dt
                #dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                #if (player.teamname == 'Home'):
                #    behind_loc = player.position[0] < target_position[0] - 1
                #else:
                #    behind_loc = player.position[0] > target_position[0] - 1
            
                # calculate ball control probablity for 'player' in time interval T+dt
                #Gives defender the advantage if they are behind the ball, if not, no advantage
                #if behind_loc:
                if player.is_gk:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * (player.lambda_def * 3)
                else:
                    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * (player.lambda_def)
                #else:
                #    dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1]
    
def generate_pitch_control_for_event(event, locations, params, field_dimen = (105.,68.,), n_grid_cells_x = 16, n_grid_cells_y=16, offsides=False):
    # get the details of the event (frame, team in possession, ball_start_position)
    ball_start_pos = np.array([event['ballx'],event['bally']])
    # break the pitch down into a grid
    #n_grid_cells_y = n_grid_cells_x * 0.75
    
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y

    xgrid = np.arange(n_grid_cells_x)*dx + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy + dy/2.
   
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    att_players = initialise_players(locations[locations['team_on_ball'] == True], "Home", params, True, event)
    def_players = initialise_players(locations[locations['team_on_ball'] == False], "Away", params, False, event)
    
        
    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        att_players = [a for a in att_players if a.offside == False]
        #for att_p in att_players:
        #    if att_p.offside == True:
        #        att_players.remove(att_p)
        
    count = 0
    # 1. 선수 리스트를 Numpy 배열로 변환
    att_pos, att_vel, att_is_gk, att_lam = to_arrays(att_players, is_attacking_team=True)
    def_pos, def_vel, def_is_gk, def_lam = to_arrays(def_players, is_attacking_team=False)

    # 2. 새로운 Numba 함수를 "단 한 번" 호출하여 전체 격자 계산
    PPCFa, PPCFd = compute_ppcf_grid(
        xgrid.astype(np.float32), ygrid.astype(np.float32),
        att_pos, att_vel, att_is_gk, att_lam,
        def_pos, def_vel, def_is_gk, def_lam,
        ball_start_pos.astype(np.float32), np.float32(params['average_ball_speed']),
        np.float32(params['reaction_time']), np.float32(params['tti_sigma']),
        np.float32(params['time_to_control_att']), np.float32(params['time_to_control_def']),
        np.float32(params['int_dt']), np.float32(params['max_int_time']), np.float32(params['model_converge_tol'])
    )
    # check probabilitiy sums within convergence
    # float(count) 대신 float(n_grid_cells_y * n_grid_cells_x) 사용
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y * n_grid_cells_x)
    assert 1 - checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa,xgrid,ygrid
