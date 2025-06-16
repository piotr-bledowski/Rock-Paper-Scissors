import time
import threading
import cv2
import numpy as np

class GameState:
    WAITING_FOR_PLAYERS = 0
    ROUND_SETUP = 1
    COUNTDOWN = 2
    ROUND_ACTIVE = 3
    ROUND_END = 4
    ROUND_BREAK = 5
    GAME_END = 6

class RPSLSGame:
    def __init__(self):
        self.state = GameState.WAITING_FOR_PLAYERS
        self.rounds_to_play = 0
        self.current_round = 0
        self.player1_score = 0
        self.player2_score = 0
        self.countdown_value = 3
        self.countdown_thread = None
        self.countdown_active = False
        self.round_result = None
        self.game_result = None
        self.round_break_timer = None
        self.round_break_duration = 3  # seconds between rounds
        self.use_countdown = True
        
    def set_countdown_enabled(self, enabled):
        """Enable or disable countdown before rounds"""
        self.use_countdown = enabled
        
    def start_game(self, rounds):
        """Start a new game with specified number of rounds"""
        self.rounds_to_play = rounds
        self.current_round = 0
        self.player1_score = 0
        self.player2_score = 0
        self.state = GameState.ROUND_SETUP
        self.start_round()
        
    def start_round(self):
        """Start a new round with optional countdown"""
        if self.current_round >= self.rounds_to_play:
            self.end_game()
            return
            
        self.current_round += 1
        self.round_result = None
        
        if self.use_countdown:
            self.state = GameState.COUNTDOWN
            self.countdown_value = 3
            self.countdown_active = True
            
            # Start countdown in a separate thread
            self.countdown_thread = threading.Thread(target=self._countdown)
            self.countdown_thread.daemon = True
            self.countdown_thread.start()
        else:
            # Skip countdown and go directly to active round
            self.state = GameState.ROUND_ACTIVE
        
    def _countdown(self):
        """Countdown thread function"""
        while self.countdown_value > 0 and self.countdown_active:
            time.sleep(1)
            self.countdown_value -= 1
            
        if self.countdown_active:  # Only proceed if countdown wasn't interrupted
            self.state = GameState.ROUND_ACTIVE
            
    def process_round_result(self, player1_gesture, player2_gesture):
        """Process the result of a round"""
        if self.state != GameState.ROUND_ACTIVE:
            return
            
        # Determine round winner
        if player1_gesture == player2_gesture:
            self.round_result = "Draw"
            # No score change for draws
        elif player2_gesture in self._get_winning_gestures(player1_gesture):
            self.round_result = "Player 1 Wins Round"
            self.player1_score += 1
        else:
            self.round_result = "Player 2 Wins Round"
            self.player2_score += 1
            
        self.state = GameState.ROUND_END
        
        # Start round break timer
        self.round_break_timer = threading.Timer(2.0, self._start_round_break)
        self.round_break_timer.start()
        
    def _start_round_break(self):
        """Start the break between rounds"""
        if self.current_round >= self.rounds_to_play:
            self.end_game()
        else:
            self.state = GameState.ROUND_BREAK
            # Start next round after break duration
            self.round_break_timer = threading.Timer(self.round_break_duration, self.start_round)
            self.round_break_timer.start()
        
    def end_game(self):
        """End the game and determine the winner"""
        self.state = GameState.GAME_END
        self.countdown_active = False
        
        if self.player1_score > self.player2_score:
            self.game_result = "Player 1 Wins Game"
        elif self.player2_score > self.player1_score:
            self.game_result = "Player 2 Wins Game"
        else:
            self.game_result = "Game Draw"
            
    def _get_winning_gestures(self, gesture):
        """Get the gestures that the given gesture beats"""
        rules = {
            'rock': ['scissors', 'lizard'],
            'paper': ['rock', 'spock'],
            'scissors': ['paper', 'lizard'],
            'lizard': ['spock', 'paper'],
            'spock': ['scissors', 'rock']
        }
        return rules.get(gesture, [])
        
    def get_score_display(self):
        """Get the current score as a string"""
        return f"{self.player1_score}:{self.player2_score}"
        
    def get_round_display(self):
        """Get the current round display string"""
        return f"Round {self.current_round}/{self.rounds_to_play}"
        
    def get_state_display(self):
        """Get the display text for the current game state"""
        if self.state == GameState.WAITING_FOR_PLAYERS:
            return "Waiting for players..."
        elif self.state == GameState.ROUND_SETUP:
            return "Get ready!"
        elif self.state == GameState.COUNTDOWN:
            return f"Round {self.current_round} starting in {self.countdown_value}..."
        elif self.state == GameState.ROUND_ACTIVE:
            return f"Round {self.current_round} - Make your gesture!"
        elif self.state == GameState.ROUND_END:
            return f"Round {self.current_round} - {self.round_result}"
        elif self.state == GameState.ROUND_BREAK:
            return f"Preparing for next round..."
        elif self.state == GameState.GAME_END:
            return f"Game Over - {self.game_result} ({self.get_score_display()})"
        return ""
        
    def is_recognition_active(self):
        """Check if gesture recognition should be active"""
        return self.state == GameState.ROUND_ACTIVE
        
    def stop_timers(self):
        """Stop all active timers"""
        self.countdown_active = False
        if self.round_break_timer:
            self.round_break_timer.cancel()
        if self.countdown_thread and self.countdown_thread.is_alive():
            self.countdown_thread.join(timeout=1) 