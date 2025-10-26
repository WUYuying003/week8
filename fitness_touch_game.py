#!/usr/bin/env python3
"""
笑容收集之旅 (Smile Collection Journey)
Inspired by NEX HomeCourt - Touch numbered circles in order using hand gestures
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import random
from camera_utils import setup_camera

class FitnessTouchGame:
    def __init__(self, frame_width=1280, frame_height=720):
        # MediaPipe Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game settings
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.score = 0
        self.level = 1
        self.current_target = 1
        self.targets = []
        self.target_radius = 50
        self.touch_threshold = 60
        
        # Level progression - number of targets increases with level
        self.targets_per_level = min(2 + self.level - 1, 8)  # Start with 2, max 8
        
        # Game states
        self.game_state = "playing"  # playing, level_complete, waiting_next_level, game_over, wrong_touch
        self.level_complete_time = None
        self.level_complete_duration = 2.0  # seconds
        self.wrong_touch_time = None
        self.wrong_touch_duration = 2.0  # seconds
        
        # Star animation for completed targets
        self.stars = []  # List of stars flying to score area
        
        # Next level button (for touch interaction)
        self.next_button = {
            'x': frame_width // 2,
            'y': frame_height // 2 + 180,
            'width': 300,
            'height': 60,
            'being_touched': False
        }
        
        # Countdown timer per target
        self.target_countdown_duration = 10.0  # 10 seconds per target
        self.current_target_start_time = time.time()
        
        # Level timer (overall)
        self.level_timer_enabled = False  # Disabled for now
        
        # Colors - uniform blue theme
        self.color_current = (255, 255, 255)  # White (current target - very obvious)
        self.color_waiting = (180, 100, 0)  # Darker blue (waiting - hollow)
        self.color_completed = (255, 150, 0)  # Blue (completed - will be filled)
        self.color_touch = (100, 255, 100)  # Green (when touching)
        self.color_progress = (0, 255, 255)  # Yellow progress bar
        
        # Initialize first level
        self.generate_targets()
        
    def generate_targets(self):
        """Generate random positions for targets"""
        self.targets = []
        margin = 80  # Reduced margin for wider distribution
        
        for i in range(self.targets_per_level):
            # Avoid overlapping circles - wider distribution
            while True:
                x = random.randint(margin, self.frame_width - margin)
                y = random.randint(margin + 100, self.frame_height - margin - 100)
                
                # Check if too close to existing targets
                valid = True
                for target in self.targets:
                    dist = np.sqrt((x - target['x'])**2 + (y - target['y'])**2)
                    if dist < self.target_radius * 2.5:  # Reduced minimum distance
                        valid = False
                        break
                
                if valid:
                    break
            
            self.targets.append({
                'number': i + 1,
                'x': x,
                'y': y,
                'completed': False,
                'being_touched': False,
                'wrong': False,  # Track if this was touched incorrectly
                'start_time': None  # Will be set when this becomes current target
            })
        
        self.current_target = 1
        self.current_target_start_time = time.time()
        self.targets[0]['start_time'] = self.current_target_start_time
        self.stars = []  # Clear stars when generating new targets
    
    def get_finger_tip_position(self, hand_landmarks, frame_shape):
        """Get index finger tip position in pixel coordinates"""
        # Index finger tip is landmark 8
        index_tip = hand_landmarks.landmark[8]
        h, w = frame_shape[:2]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        return x, y
    
    def check_touch(self, finger_x, finger_y):
        """Check if finger is touching any target - return target number or None"""
        for target in self.targets:
            if not target['completed']:
                dist = np.sqrt((finger_x - target['x'])**2 + (finger_y - target['y'])**2)
                
                if dist < self.touch_threshold:
                    target['being_touched'] = True
                    return target['number']  # Return which target was touched
                else:
                    target['being_touched'] = False
        
        return None
    
    def check_button_touch(self, finger_x, finger_y):
        """Check if finger is touching the next level button"""
        button_x = self.next_button['x'] - self.next_button['width'] // 2
        button_y = self.next_button['y'] - self.next_button['height'] // 2
        
        if (button_x <= finger_x <= button_x + self.next_button['width'] and
            button_y <= finger_y <= button_y + self.next_button['height']):
            self.next_button['being_touched'] = True
            return True
        else:
            self.next_button['being_touched'] = False
            return False
    
    def complete_target(self):
        """Mark current target as completed and move to next"""
        for target in self.targets:
            if target['number'] == self.current_target:
                target['completed'] = True
                target['being_touched'] = False
                
                # Create a star that flies to score area
                self.stars.append({
                    'x': target['x'],
                    'y': target['y'],
                    'target_x': 100,  # Score area position
                    'target_y': 45,
                    'progress': 0.0  # 0 to 1
                })
                
                self.score += 1
                self.current_target += 1
                
                # Start timer for next target
                if self.current_target <= self.targets_per_level:
                    self.current_target_start_time = time.time()
                    self.targets[self.current_target - 1]['start_time'] = self.current_target_start_time
                
                # Check if level is complete
                if self.current_target > self.targets_per_level:
                    # Don't complete level immediately, wait for star animation
                    pass
                
                return
    
    def wrong_touch(self, touched_number):
        """Handle touching the wrong target"""
        self.game_state = "wrong_touch"
        self.wrong_touch_time = time.time()
        
        # Mark the wrong target as red
        for target in self.targets:
            if target['number'] == touched_number:
                target['wrong'] = True
    
    def complete_level(self):
        """Complete current level and prepare next one"""
        self.game_state = "waiting_next_level"
        self.level_complete_time = time.time()
        self.level += 1
        self.targets_per_level = min(2 + self.level - 1, 8)  # Increase targets
    
    def start_next_level(self):
        """Start the next level"""
        self.game_state = "playing"
        self.generate_targets()
    
    def reset_game(self):
        """Reset game to initial state"""
        self.score = 0
        self.level = 1
        self.current_target = 1
        self.targets_per_level = 2
        self.game_state = "playing"
        self.level_complete_time = None
        self.generate_targets()
    
    def update(self):
        """Update game state"""
        if self.game_state == "waiting_next_level":
            # Just wait for user input to continue
            pass
        
        elif self.game_state == "wrong_touch":
            # Check if wrong touch message duration is over
            if time.time() - self.wrong_touch_time > self.wrong_touch_duration:
                # Restart current level
                self.generate_targets()
                self.game_state = "playing"
        
        elif self.game_state == "playing":
            # Check if all targets are completed and stars animation is done
            if self.current_target > self.targets_per_level and len(self.stars) == 0:
                self.complete_level()
            
            # Check current target countdown
            if self.current_target <= self.targets_per_level:
                elapsed = time.time() - self.current_target_start_time
                if elapsed > self.target_countdown_duration:
                    # Time up for current target, end level
                    self.game_state = "time_up"
            
            # Update star animations
            for star in self.stars[:]:
                star['progress'] += 0.05  # Animation speed
                if star['progress'] >= 1.0:
                    self.stars.remove(star)
    
    def draw_targets(self, frame):
        """Draw all targets on frame"""
        for target in self.targets:
            x, y = target['x'], target['y']
            number = target['number']
            
            # Draw countdown progress bar for current target
            if number == self.current_target and not target['completed'] and target['start_time']:
                elapsed = time.time() - target['start_time']
                remaining_ratio = max(0, 1 - (elapsed / self.target_countdown_duration))
                
                # Draw progress circle (arc)
                angle = int(360 * remaining_ratio)
                if angle > 0:
                    # Draw remaining progress as arc
                    cv2.ellipse(frame, (x, y), (self.target_radius + 10, self.target_radius + 10),
                               0, -90, -90 + angle, self.color_progress, 5)
            
            # Choose color and style based on state
            if target.get('wrong', False):
                # Wrong touch: Red filled circle
                color = (0, 0, 255)
                cv2.circle(frame, (x, y), self.target_radius, color, -1)  # Filled
                # Draw X mark
                offset = self.target_radius // 2
                cv2.line(frame, (x - offset, y - offset), (x + offset, y + offset), (255, 255, 255), 4)
                cv2.line(frame, (x - offset, y + offset), (x + offset, y - offset), (255, 255, 255), 4)
            elif target['completed']:
                # Completed: Blue filled circle (no number, will show star)
                color = self.color_completed
                cv2.circle(frame, (x, y), self.target_radius, color, -1)  # Filled
            elif number == self.current_target:
                # Current target: WHITE hollow circle with thicker border
                if target['being_touched']:
                    color = self.color_touch
                else:
                    color = self.color_current
                cv2.circle(frame, (x, y), self.target_radius, color, 5)  # Hollow, thicker
                # Draw number in white
                text_size = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(frame, str(number), (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
            else:
                # Waiting targets: hollow circle with normal border (no countdown yet)
                color = self.color_waiting
                cv2.circle(frame, (x, y), self.target_radius, color, 3)  # Hollow
                # Draw number
                text_size = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(frame, str(number), (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    def draw_stars(self, frame):
        """Draw flying stars animation"""
        for star in self.stars:
            # Interpolate position
            progress = star['progress']
            x = int(star['x'] + (star['target_x'] - star['x']) * progress)
            y = int(star['y'] + (star['target_y'] - star['y']) * progress)
            
            # Draw yellow star with white outline
            # Scale gets smaller as it flies
            scale = 1.5 - (progress * 0.5)
            size = int(20 * scale)
            
            # Draw a filled circle (like a coin/smiley)
            cv2.circle(frame, (x, y), size, (0, 215, 255), -1)  # Yellow filled circle
            cv2.circle(frame, (x, y), size, (255, 255, 255), 2)  # White outline
            
            # Draw simple smiley face
            if size > 10:
                # Eyes
                eye_offset_x = int(size * 0.3)
                eye_offset_y = int(size * 0.2)
                eye_size = max(2, int(size * 0.15))
                cv2.circle(frame, (x - eye_offset_x, y - eye_offset_y), eye_size, (0, 0, 0), -1)
                cv2.circle(frame, (x + eye_offset_x, y - eye_offset_y), eye_size, (0, 0, 0), -1)
                
                # Smile
                smile_y = int(y + size * 0.2)
                cv2.ellipse(frame, (x, smile_y), (int(size*0.4), int(size*0.2)), 0, 0, 180, (0, 0, 0), 2)
    
    def draw_ui(self, frame):
        """Draw game UI elements"""
        # Title bar background
        cv2.rectangle(frame, (0, 0), (self.frame_width, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (self.frame_width, 70), (255, 255, 255), 2)
        
        # Score with stars ⭐
        # Draw smiley icon
        cv2.circle(frame, (30, 30), 15, (0, 215, 255), -1)  # Yellow circle
        cv2.circle(frame, (30, 30), 15, (255, 255, 255), 2)  # White outline
        # Eyes
        cv2.circle(frame, (25, 26), 2, (0, 0, 0), -1)
        cv2.circle(frame, (35, 26), 2, (0, 0, 0), -1)
        # Smile
        cv2.ellipse(frame, (30, 33), (6, 3), 0, 0, 180, (0, 0, 0), 2)
        
        score_text = f"x {self.score}"
        cv2.putText(frame, score_text, (50, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"Level: {self.level}", (250, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3)
        
        # Countdown timer for current target
        if self.game_state == "playing" and self.current_target <= self.targets_per_level:
            elapsed = time.time() - self.current_target_start_time
            remaining = max(0, self.target_countdown_duration - elapsed)
            timer_color = (100, 255, 255) if remaining > 5 else (0, 100, 255)
            
            cv2.putText(frame, f"Target Time: {int(remaining)}s", (500, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, timer_color, 2)
        
        # Instructions at bottom
        bottom_y = self.frame_height - 60
        cv2.rectangle(frame, (0, bottom_y), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        
        if self.game_state == "playing":
            instruction = f"Touch circles in order: 1 → 2 → 3 → ... → {self.targets_per_level}"
            cv2.putText(frame, instruction, (20, bottom_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "ESC: Quit | R: Restart", (20, bottom_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Level complete overlay - waiting for next level
        if self.game_state == "waiting_next_level":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 50, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            text = "LEVEL COMPLETE!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
            text_x = (self.frame_width - text_size[0]) // 2
            text_y = self.frame_height // 2 - 80
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (100, 255, 100), 5)
            
            next_level_text = f"Next Level: {self.level}"
            text_size2 = cv2.getTextSize(next_level_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x2 = (self.frame_width - text_size2[0]) // 2
            cv2.putText(frame, next_level_text, (text_x2, text_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 100), 3)
            
            targets_text = f"Targets: {self.targets_per_level}"
            text_size3 = cv2.getTextSize(targets_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x3 = (self.frame_width - text_size3[0]) // 2
            cv2.putText(frame, targets_text, (text_x3, text_y + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 255), 2)
            
            # Draw interactive button for hand touch
            button_x = self.next_button['x'] - self.next_button['width'] // 2
            button_y = self.next_button['y'] - self.next_button['height'] // 2
            button_width = self.next_button['width']
            button_height = self.next_button['height']
            
            # Button color changes when touched
            if self.next_button['being_touched']:
                button_color = (50, 200, 50)  # Darker green when touched
            else:
                button_color = (100, 255, 100)  # Bright green
            
            # Button background
            cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height),
                         button_color, -1)
            cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height),
                         (255, 255, 255), 3)
            
            # Button text - just "NEXT"
            button_text = "NEXT"
            text_size4 = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x4 = button_x + (button_width - text_size4[0]) // 2
            text_y4 = button_y + (button_height + text_size4[1]) // 2
            cv2.putText(frame, button_text, (text_x4, text_y4),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        # Time up overlay
        elif self.game_state == "time_up":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 50), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            text = "TIME'S UP!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
            text_x = (self.frame_width - text_size[0]) // 2
            text_y = self.frame_height // 2
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (100, 100, 255), 5)
            
            score_text = f"Final Score: {self.score} | Level: {self.level}"
            text_size2 = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x2 = (self.frame_width - text_size2[0]) // 2
            cv2.putText(frame, score_text, (text_x2, text_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 100), 2)
            
            restart_text = "Press R to Restart"
            text_size3 = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x3 = (self.frame_width - text_size3[0]) // 2
            cv2.putText(frame, restart_text, (text_x3, text_y + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # Wrong touch overlay
        elif self.game_state == "wrong_touch":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            text = "WRONG ORDER!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
            text_x = (self.frame_width - text_size[0]) // 2
            text_y = self.frame_height // 2
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
            
            retry_text = "Restarting level..."
            text_size2 = cv2.getTextSize(retry_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x2 = (self.frame_width - text_size2[0]) // 2
            cv2.putText(frame, retry_text, (text_x2, text_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            instruction_text = f"Touch circles in order: 1 → {self.targets_per_level}"
            text_size3 = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x3 = (self.frame_width - text_size3[0]) // 2
            cv2.putText(frame, instruction_text, (text_x3, text_y + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

def main():
    print("😊 笑容收集之旅 (Smile Collection Journey)")
    print("=" * 40)
    print("Inspired by NEX HomeCourt")
    print()
    
    try:
        # Setup camera using configuration
        cap, camera_id = setup_camera()
        print("✅ Camera initialized successfully")
        
    except RuntimeError as e:
        print(f"❌ Camera Error: {e}")
        print("💡 Run 'python setup_camera.py' to configure your camera")
        return
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read from camera")
        return
    
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize game
    game = FitnessTouchGame(frame_width, frame_height)
    
    print("🚀 Game started!")
    print("📖 Instructions:")
    print("   - Touch numbered circles in order (1→2→3→...)")
    print("   - Collect smiles by touching targets!")
    print("   - Each correct touch = 1 smile 😊")
    print("   - Complete all circles to advance to next level")
    print("   - Touch the NEXT button to continue")
    print("   - Press ESC to quit, R to restart")
    print()
    
    finger_touched_last_frame = False
    finger_positions = []  # Track all finger positions from both hands
    button_touched_last_frame = False  # Track button touch state
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = game.hands.process(rgb_frame)
        
        finger_positions = []  # Reset finger positions for this frame
        
        # Draw hand landmarks and get finger positions from both hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                game.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    game.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                
                # Get index finger tip position
                finger_x, finger_y = game.get_finger_tip_position(hand_landmarks, frame.shape)
                finger_positions.append((finger_x, finger_y))
                
                # Draw finger tip indicator
                cv2.circle(frame, (finger_x, finger_y), 15, (255, 255, 0), -1)
                cv2.circle(frame, (finger_x, finger_y), 18, (255, 255, 255), 2)
        
        # Update game state
        game.update()
        
        # Check for touches based on game state
        if game.game_state == "playing" and len(finger_positions) > 0:
            touched_target = None
            for finger_x, finger_y in finger_positions:
                target_num = game.check_touch(finger_x, finger_y)
                if target_num is not None:
                    touched_target = target_num
                    break
            
            # Handle touch
            if touched_target is not None and not finger_touched_last_frame:
                if touched_target == game.current_target:
                    # Correct target touched
                    game.complete_target()
                else:
                    # Wrong target touched
                    game.wrong_touch(touched_target)
                
                finger_touched_last_frame = True
            elif touched_target is None:
                finger_touched_last_frame = False
        
        elif game.game_state == "waiting_next_level" and len(finger_positions) > 0:
            # Check if button is touched
            button_touched = False
            for finger_x, finger_y in finger_positions:
                if game.check_button_touch(finger_x, finger_y):
                    button_touched = True
                    break
            
            # Start next level on button touch
            if button_touched and not button_touched_last_frame:
                game.start_next_level()
                print(f"🎯 Starting Level {game.level - 1}!")
                button_touched_last_frame = True
            elif not button_touched:
                button_touched_last_frame = False
        
        else:
            finger_touched_last_frame = False
            button_touched_last_frame = False
        
        # Draw game elements
        game.draw_targets(frame)
        game.draw_stars(frame)  # Draw flying stars
        game.draw_ui(frame)
        
        # Display frame
        cv2.imshow('笑容收集之旅 - Smile Collection Journey', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            game.reset_game()
            print("🔄 Game restarted!")
        elif key == ord(' '):  # SPACE
            if game.game_state == "waiting_next_level":
                game.start_next_level()
                print(f"🎯 Starting Level {game.level - 1}!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Final Stats:")
    print(f"   Smiles Collected: {game.score} 😊")
    print(f"   Level Reached: {game.level}")
    print("👋 Thanks for playing 笑容收集之旅!")

if __name__ == "__main__":
    main()
