#!/usr/bin/env python3

import ffai
import numpy as np

from ffai import Action,ActionType,Square
import ffai.ai.pathfinding as pf

def position_abs_distance(pos1, pos2):
    return max(abs(pos1.x - pos2.x), abs(pos1.y - pos2.y))

def is_better_block_action(action_choice1, action_choice2):
    rankings = {}
    rankings[ffai.ActionType.SELECT_ATTACKER_DOWN]     = 0
    rankings[ffai.ActionType.SELECT_BOTH_DOWN]         = 1
    rankings[ffai.ActionType.SELECT_PUSH]              = 2
    rankings[ffai.ActionType.SELECT_DEFENDER_STUMBLES] = 3
    rankings[ffai.ActionType.SELECT_DEFENDER_DOWN]     = 4
    return rankings[action_choice1.action_type] > rankings[action_choice2.action_type]

class MyRandomProcBot(ffai.ProcBot):

    def __init__(self, name, seed=None, log_act=False):
        super().__init__(name)
        self.my_team = None
        self.other_team = None
        self.rnd = np.random.RandomState(seed)
        self.log_act = log_act
        if self.log_act:
            self.log_file = open('log.txt', 'w')

    def new_game(self, game, team):
        self.my_team = team
        self.other_team = game.get_opp_team(self.my_team)
        
    def _random_act(self, game, caller="??"):
        # Most of this code is taken from MyRandomBot in the tutorial
        print("ACTING RANDOMLY")
        if self.log_act:
            self.log_file.write("<{}> actions:\n".format(caller))
            for action in game.state.available_actions:
                self.log_file.write("\t\t{}\n".format(action.to_json()))
        
        # Select a random action type
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            # Ignore PLACE_PLAYER actions
            if action_choice.action_type != ffai.ActionType.PLACE_PLAYER:
                break

        # Select a random position and/or player
        position = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None

        # Make action object
        action = ffai.Action(action_choice.action_type, position=position, player=player)

        # Return action to the framework
        print(action.to_json())
        return action

    def player_action(self, game):
        return self._random_act(game)

    def use_pro(self, game):
        return self._random_act(game)

    def use_juggernaut(self, game):
        return self._random_act(game)

    def use_wrestle(self, game):
        return self._random_act(game)

    def use_stand_firm(self, game):
        return self._random_act(game)

    def coin_toss_flip(self, game):
        return self._random_act(game)

    def coin_toss_kick_receive(self, game):
        return self._random_act(game)

    def setup(self, game):
        return self._random_act(game)

    def reroll(self, game):
        return self._random_act(game, caller='reroll')

    def place_ball(self, game):
        return self._random_act(game, caller='place_ball')

    def high_kick(self, game):
        return self._random_act(game, caller='high_kick')

    def touchback(self, game):
        return self._random_act(game)

    def quick_snap(self, game):
        return self._random_act(game)

    def blitz(self, game):
        return self._random_act(game)

    def block(self, game):
        return self._random_act(game, caller='block')
    
    def push(self, game):
        return self._random_act(game)
    
    def follow_up(self, game):
        return self._random_act(game,  caller='follow_up')

    def apothecary(self, game):
        return self._random_act(game)
    
    def interception(self, game):
        return self._random_act(game)

    def end_game(self, game):
        print("Scores: {} - {}".format(self.my_team.state.score, self.other_team.state.score))


class TerribleBot(MyRandomProcBot):
    def __init__(self, name, seed=None, log_act=False):
        super().__init__(name, seed=seed, log_act=True)
        self.last_num_free = -1
        self.new_turn = True
        self.queued_actions = []
        self.defending = True

    def player_action(self, game):
        '''
        Player actions are always queued up ahead of time, so
        this callback will just pop the top action from the queue
        and execute it, or end the player's turn if the queue is
        empty.
        '''
        if self.queued_actions:
            action = self.queued_actions[0]
            print(action.to_json())
            self.queued_actions = self.queued_actions[1:]
            return action
        return Action(ActionType.END_PLAYER_TURN)

    def turn(self, game):
        '''
        Given a non-player turn, first work out if this is the first
        time I have been called this game turn, then split off into
        seperate defensive and offensive routines.
        '''
        self.my_team = game.get_team_by_id(self.my_team.team_id)
        self.other_team = game.get_opp_team(self.my_team)

        num_free = len([p for p in self.my_team.players if not p.state.used])
        if num_free > self.last_num_free:
            self._setup_new_turn(game)
        self.last_num_free = num_free

        if self.queued_actions:
            action = self.queued_actions[0]
            self.queued_actions = self.queued_actions[1:]
            return action
            
        if self.defending:
            return self.get_defense_action(game)
        else:
            return self.get_offense_action(game)

    def get_defense_action(self, game):
        stand_up_action = self._stand_up(game)
        if stand_up_action:
            return stand_up_action
        
        mark_carrier_action = self._mark_ball_carrier(game)
        if mark_carrier_action:
            return mark_carrier_action

        block_carrier_action = self._block_ball_carrier(game)
        if block_carrier_action:
            return block_carrier_action
        
        obstruct_carrier_action = self._move_in_way_of_carrier(game)
        if obstruct_carrier_action:
            return obstruct_carrier_action
            
        mark_all_action = self._mark_all(game)
        if mark_all_action:
            return mark_all_action
        
        block_action = self._block(game)
        if block_action:
            return block_action

        return self._end_turn()

    def get_offense_action(self, game):
        # TODO: This is a stub. Improve this logic.
        ball_carrier = game.get_ball_carrier()
        if ball_carrier and ball_carrier.team != self.my_team:
            return self.get_defense_action(game)
        
        if ball_carrier:
            return self._get_offense_action_posession(game, ball_carrier)
        else:
            return self._get_offense_action_no_posession(game)

    def block(self, game):
        all_action_choices = game.state.available_actions
        best_action_choice = all_action_choices[0]
        for action_choice in all_action_choices:
            if is_better_block_action(action_choice, best_action_choice):
                best_action_choice = action_choice
        possible_positions = best_action_choice.positions
        position = self.rnd.choice(possible_positions) if possible_positions else None
        possible_players = best_action_choice.players
        player = self.rnd.choice(possible_players) if possible_players else None
        return ffai.Action(best_action_choice.action_type, position=position, player=player)
        

    # ------------------------------------------------------------------
    #  GENERIC UTILITY FUNCTIONS
    # ------------------------------------------------------------------
        
    def _setup_new_turn(self, game):
        self._compute_defending(game)
        if self.defending:
            self._setup_new_turn_defense(game)
        else:
            self._setup_new_turn_offense(game)
        self.new_turn = False

    def _compute_defending(self, game):
        # If a player on either team has the ball, easy to see who is on offense
        ball_carrier = game.get_ball_carrier()
        if ball_carrier:
            self.defending = (ball_carrier.team != self.my_team)
            return

        # If the ball is not held, check if we can get it
        ball_position = game.get_ball_position()
        free_players = [p for p in self.my_team.players if p.position and not p.state.used]
        for player in free_players:
            path = pf.get_safest_path(game, player, ball_position)
            if path and path.prob > 0.7:
                self.defending = False
                return

        # Otherwise, check which team is closer to the ball on average
        my_avg_distance = self._get_avg_distance(self.my_team, ball_position)
        other_avg_distance = self._get_avg_distance(self.other_team, ball_position)
        self.defending = (my_avg_distance > other_avg_distance)

    def _get_avg_distance(self, team, position):
        total_distance = 0.
        for player in team.players:
            if player.position is not None:
                total_distance += max(abs(player.position.x - position.x), abs(player.position.y - position.y))
        return total_distance / len(team.players)

    def _setup_new_turn_defense(self, game):
        pass

    def _setup_new_turn_offense(self, game):
        pass

    def _stand_up(self, game):
        down_players = [p for p in self.my_team.players if p.position and not p.state.used and not p.state.up]
        for player in down_players:
            self.queued_actions.append(Action(ActionType.STAND_UP))
            return Action(ActionType.START_MOVE, player=player)
        return None

    def _get_adjacent_pos(self, position, game):
        positions = []
        for x_diff in [-1,0,1]:
            for y_diff in [-1,0,1]:
                if x_diff == 0 and y_diff == 0:
                    continue
                candidate_pos = Square(position.x+x_diff,position.y+y_diff)
                if not game.is_out_of_bounds(candidate_pos):
                    positions.append(candidate_pos)
        return positions

    def _end_turn(self):
        self.new_turn = True
        self.last_num_free = -1
        self.queued_actions = []
        return Action(ActionType.END_TURN)

    # ------------------------------------------------------------------
    #  DEFENSIVE STRATEGIES
    # ------------------------------------------------------------------
    
    def _mark_ball_carrier(self, game, allow_nonmove=False):
        ball_carrier = game.get_ball_carrier()
        if not ball_carrier: #or ball_carrier.team == self.my_team: # TODO: hack?
            return None
        free_players = [p for p in self.my_team.players if p.position and not p.state.used]
        if not allow_nonmove:
            free_players = [p for p in free_players if position_abs_distance(p.position,ball_carrier.position)>1]
        best_tuple = None
        for player in free_players:
            path = pf.get_safest_path_to_player(game, player, ball_carrier)
            if path and (not best_tuple or path.prob > best_tuple[0]):
                best_tuple = (path.prob, player, path)
        if best_tuple and best_tuple[0] >= 0.9: # TODO: CONSTANT
            player,path = best_tuple[1:]
            if not player.state.up:
                self.queued_actions.append(Action(ActionType.STAND_UP))
            self.queued_actions.extend([Action(ActionType.MOVE, position=s) for s in path.steps])
            return Action(ActionType.START_MOVE, player=player)
        return None

    def _move_in_way_of_carrier(self, game):
        ball_carrier = game.get_ball_carrier()
        if not ball_carrier or ball_carrier.team == self.my_team:
            return None
        path = pf.get_safest_scoring_path(game, ball_carrier, max_search_distance=30)
        if not path:
            return None
        
        free_players = [p for p in self.my_team.players if p.position and not p.state.used]
        best_tuple = (-1,)
        for player in free_players:
            for step in path.steps:
                p_path = pf.get_safest_path(game, player, step)
                if p_path and p_path.prob > best_tuple[0] and len(p_path.steps):
                    best_tuple = (p_path.prob, player, p_path)
        if best_tuple[0] > 0.9: # TODO: CONSTANT
            player,path = best_tuple[1:]
            if not player.state.up:
                self.queued_actions.append(Action(ActionType.STAND_UP))
            self.queued_actions.extend([Action(ActionType.MOVE, position=s) for s in path.steps])
            return Action(ActionType.START_MOVE, player=player)

        return None

    
    def _mark_all(self, game):
        free_players = [p for p in self.my_team.players if p.position and not p.state.used]
        free_players = [p for p in free_players if game.num_tackle_zones_in(p) == 0]
        ball_carrier = game.get_ball_carrier()
        if ball_carrier:
            free_players = [p for p in free_players if position_abs_distance(p.position,ball_carrier.position) > 1]
        other_free_players = [p for p in self.other_team.players if p.position and p.state.up]
        other_free_players = [p for p in other_free_players if game.num_tackle_zones_in(p) <= 1] # TODO!
        
        best_tuple = None
        for player in free_players:
            for other_free_player in other_free_players:
                if player.position.distance(other_free_player.position) > player.num_moves_left():
                    continue
                path = pf.get_safest_path_to_player(game, player, other_free_player)
                if not path or not path.steps:
                    continue
                if not best_tuple or best_tuple[0] < path.prob:
                    best_tuple = (path.prob, player, path)
        if best_tuple and best_tuple[0] > 0.9: #TODO: CONSTANT
            player,path = best_tuple[1:]
            if not player.state.up:
                self.queued_actions.append(Action(ActionType.STAND_UP))
            self.queued_actions.extend([Action(ActionType.MOVE, position=p) for p in path.steps])
            return Action(ActionType.START_MOVE, player=player)

        print("Nothing left")
        if best_tuple:
            print(best_tuple)
        return None

    def _block(self, game):
        free_players = [p for p in self.my_team.players if p.position and not p.state.used and p.state.up]
        best_block = None
        for player in free_players:
            for other_player in game.get_adjacent_opponents(player, down=False):
                me_knock_prob, other_knock_prob, me_fmbl_prob, other_fmbl_prob = game.get_block_probs(player, other_player)
                if me_knock_prob > 0.15: #TODO: CONSTANT
                    continue
                # TODO: include other_fmbl_prob
                if not best_block or other_knock_prob > best_block[0]:
                    best_block = (other_knock_prob, player, other_player)
        if best_block:
            player,other_player = best_block[1:]
            self.queued_actions.append(Action(ActionType.BLOCK, position=other_player.position))
            return Action(ActionType.START_BLOCK, player=player)
        return None

    def _block_ball_carrier(self, game):
        ball_carrier = game.get_ball_carrier()
        if not ball_carrier or ball_carrier.team == self.my_team:
            return None
        free_players = [p for p in self.my_team.players if p.position and not p.state.used and p.state.up]
        best_block = None
        for player in free_players:
            if position_abs_distance(ball_carrier.position,player.position) != 1:
                # TODO: is the the best way to check this?
                continue
            me_knock_prob, other_knock_prob, me_fmbl_prob, other_fmbl_prob = game.get_block_probs(player, ball_carrier)
            if me_knock_prob > 0.2: #TODO: CONSTANT
                continue
            # TODO: include other_fmbl_prob
            if not best_block or other_knock_prob > best_block[0]:
                best_block = (other_knock_prob, player, ball_carrier)
        if best_block:
            player,other_player = best_block[1:]
            self.queued_actions.append(Action(ActionType.BLOCK, position=ball_carrier.position))
            return Action(ActionType.START_BLOCK, player=player)
        return None

    # ------------------------------------------------------------------
    #  OFFENSIVE "STRATEGIES"
    # ------------------------------------------------------------------

    def _get_offense_action_posession(self, game, ball_carrier):
        if game.num_tackle_zones_in(ball_carrier) == 0 and not ball_carrier.state.used:
            path = pf.get_safest_scoring_path(game, ball_carrier, max_search_distance=30)
            if path:
                real_steps = []
                steps = path.steps[:ball_carrier.num_moves_left(include_gfi=False)]
                for step in steps:
                    if game.num_tackle_zones_at(ball_carrier, step) > 0 and path.prob < .85:
                        break
                    real_steps.append(step)
                if real_steps:
                    self.queued_actions = [Action(ActionType.MOVE, position=p) for p in real_steps]
                    return Action(ActionType.START_MOVE, player=ball_carrier)
            
        action = self._mark_ball_carrier(game, allow_nonmove=True)
        if action:
            return action

        # HACK REMOVE
        return self.get_defense_action(game)

    def _get_offense_action_no_posession(self, game):
        # If we do not possess the ball, find safest player to move toward ball
        ball_position = game.get_ball_position()
        free_players = [p for p in self.my_team.players if p.position and not p.state.used]
        #free_players = [p for p in free_players if game.num_tackle_zones_in(p) == 0]

        best_player = None
        for player in free_players:
            path = pf.get_safest_path(game, player, ball_position)
            if not path:
                continue
            pickup_prob = game.get_pickup_prob(player, ball_position)
            success_prob = path.prob * pickup_prob
            if not best_player or success_prob > best_player[0]:
                best_player = (success_prob, player, path)

        if best_player:
            player,path = best_player[1:]
            # TODO: refactor below
            if not player.state.up:
                self.queued_actions.append(Action(ActionType.STAND_UP))
            self.queued_actions.extend([Action(ActionType.MOVE, position=p) for p in path.steps])
            return Action(ActionType.START_MOVE, player=player)
        return self._end_turn()

# Register the bot to the framework
ffai.register_bot('terriblebot', TerribleBot)


if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = ffai.load_config("bot-bowl-ii")
    config.competition_mode = False
    ruleset = ffai.load_rule_set(config.ruleset, all_rules=False)
    arena = ffai.load_arena(config.arena)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    import test_random
    import grodbot

    # Play 10 games
    game_times = []
    for i in range(10):
        away_agent = ffai.make_bot("terriblebot")
        home_agent = ffai.make_bot("grodbot")

        game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")
