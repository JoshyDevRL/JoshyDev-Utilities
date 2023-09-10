import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder


class CustomObs(ObsBuilder):
    POS_STD = 2300
    ANG_STD = math.pi

    def __init__(self, team_size=3):
        super().__init__()
        self.team_size = team_size

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)
        if player.team_num == -1:
            for i in range(5):
                self._add_dummy(obs)
        
        else:
            allies = []
            enemies = []

            for other in state.players:
                if other.car_id == player.car_id:
                    continue
                if other.team_num == player.team_num:
                    allies.append(other)
                else:
                    enemies.append(other)

            self._add_player(obs, player_car, enemies[0], ball, inverted)
            enemies.pop(0)
            for i in range(self.team_size-1):
                if len(allies) == i:
                    self._add_dummy(obs)
                else:
                    self._add_player(obs, player_car, allies[i], ball, inverted)
                
                if len(enemies) == i:
                    self._add_dummy(obs)
                else:
                    self._add_player(obs, player_car, enemies[i], ball, inverted)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])
        return player_car

    def _add_dummy(self, obs: List):
        obs.extend([
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [0, 0, 0, 0]])
        obs.extend([np.zeros(3), np.zeros(3)])

    def _add_player(self, obs: List, player_car: PhysicsObject, player: PlayerData, ball: PhysicsObject, inverted: bool):
        other_car = self._add_player_to_obs(obs, player, ball, inverted)
        obs.extend([
            (other_car.position - player_car.position) / self.POS_STD,
            (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
        ])