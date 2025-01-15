import math
from typing import List, Tuple

class Position:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def move(self, vector: Tuple[float, float]):
        self.x += vector[0]
        self.y += vector[1]
    
    def __str__(self):
        return f"({self.x}, {self.y})"

class Enemy:
    def __init__(self, label: str, position: Tuple[float, float], move_vector: Tuple[float, float]):
        self.label = label
        self.position = Position(*position)
        self.move_vector = move_vector
        self.life_points = 10
        self.is_dead = False
    
    def move(self):
        if not self.is_dead:
            self.position.move(self.move_vector)
    
    def take_damage(self, damage: int):
        if not self.is_dead:
            self.life_points -= damage
            if self.life_points <= 0:
                self.is_dead = True
                self.life_points = 0

    def __str__(self):
        return f"{self.label} at {self.position}, HP: {self.life_points}"

class Tower:
    def __init__(self, label: str, position: Tuple[float, float], attack_points: int, attack_range: int):
        self.label = label
        self.position = Position(*position)
        self.attack_points = attack_points
        self.attack_range = attack_range
    
    def can_attack(self, enemy: Enemy) -> bool:
        if enemy.is_dead:
            return False
        return self.position.distance_to(enemy.position) <= self.attack_range
    
    def attack(self, enemies: List[Enemy]):
        for enemy in enemies:
            if self.can_attack(enemy):
                enemy.take_damage(self.attack_points)

class Game:
    def __init__(self):
        # 初始化敵人
        self.enemies = [
            Enemy("E1", (-10, 2), (2, -1)),
            Enemy("E2", (-8, 0), (3, 1)),
            Enemy("E3", (-9, -1), (3, 0))
        ]
        
        # 初始化基礎防禦塔
        self.basic_towers = [
            Tower("T1", (-3, 2), 1, 2),
            Tower("T2", (-1, -2), 1, 2),
            Tower("T3", (4, 2), 1, 2),
            Tower("T4", (7, 0), 1, 2)
        ]
        
        # 初始化進階防禦塔
        self.advanced_towers = [
            Tower("A1", (1, 1), 2, 4),
            Tower("A2", (4, -3), 2, 4)
        ]
    
    def run_turn(self, turn_number: int):
        print(f"\n=== Turn {turn_number} ===")
        
        # 1. 移動所有敵人
        for enemy in self.enemies:
            enemy.move()
        
        # 2. 所有防禦塔進行攻擊
        for tower in self.basic_towers + self.advanced_towers:
            tower.attack(self.enemies)
        
        # 顯示當前狀態
        for enemy in self.enemies:
            print(enemy)
    
    def run_game(self, turns: int = 10):
        print("Game Start!")
        for turn in range(1, turns + 1):
            self.run_turn(turn)
        print("\nGame Over!")
        print("\nFinal Status:")
        for enemy in self.enemies:
            print(f"{enemy.label} - Final Position: {enemy.position}, Life Points: {enemy.life_points}")

# 執行遊戲
game = Game()
game.run_game()