#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from wsgiref.simple_server import make_server

# =======================
# 1. 定義遊戲相關類別
# =======================
class Position:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def move(self, vector):
        self.x += vector[0]
        self.y += vector[1]

    def to_dict(self):
        return {"x": self.x, "y": self.y}

class Enemy:
    def __init__(self, label, position, move_vector):
        self.label = label
        self.position = Position(*position)
        self.move_vector = move_vector
        self.life_points = 10
        self.is_dead = False
    
    def move(self):
        if not self.is_dead:
            self.position.move(self.move_vector)
    
    def take_damage(self, damage):
        if not self.is_dead:
            self.life_points -= damage
            if self.life_points <= 0:
                self.is_dead = True
                self.life_points = 0

    def to_dict(self):
        return {
            "label": self.label,
            "x": self.position.x,
            "y": self.position.y,
            "life_points": self.life_points,
            "is_dead": self.is_dead
        }

class Tower:
    def __init__(self, label, position, attack_points, attack_range):
        self.label = label
        self.position = Position(*position)
        self.attack_points = attack_points
        self.attack_range = attack_range
    
    def can_attack(self, enemy: Enemy) -> bool:
        if enemy.is_dead:
            return False
        return self.position.distance_to(enemy.position) <= self.attack_range
    
    def attack(self, enemies):
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
        # 初始化防禦塔
        self.basic_towers = [
            Tower("T1", (-3, 2), 1, 2),
            Tower("T2", (-1, -2), 1, 2),
            Tower("T3", (4, 2), 1, 2),
            Tower("T4", (7, 0), 1, 2)
        ]
        self.advanced_towers = [
            Tower("A1", (1, 1), 2, 4),
            Tower("A2", (4, -3), 2, 4)
        ]
        self.turn = 0
        self.max_turns = 10
    
    def run_turn(self):
        self.turn += 1
        # 1. 移動所有敵人
        for enemy in self.enemies:
            enemy.move()
        # 2. 所有防禦塔進行攻擊
        for tower in self.basic_towers + self.advanced_towers:
            tower.attack(self.enemies)

    def to_dict(self):
        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "enemies": [enemy.to_dict() for enemy in self.enemies],
            "basicTowers": [
                {
                    "label": tower.label,
                    "x": tower.position.x,
                    "y": tower.position.y,
                    "attack_points": tower.attack_points,
                    "attack_range": tower.attack_range
                }
                for tower in self.basic_towers
            ],
            "advancedTowers": [
                {
                    "label": tower.label,
                    "x": tower.position.x,
                    "y": tower.position.y,
                    "attack_points": tower.attack_points,
                    "attack_range": tower.attack_range
                }
                for tower in self.advanced_towers
            ]
        }

# 在全域放一個 game 變數，當前遊戲狀態
GAME = None

# =======================
# 2. 實作簡易 WSGI APP
# =======================

def simple_app(environ, start_response):
    """
    WSGI 應用程式入口，解析路徑與方法，回傳 JSON。
    """
    path = environ.get('PATH_INFO', '')
    method = environ.get('REQUEST_METHOD', 'GET')
    
    # 預先宣告一個「客製化的start_response」，我們要在回傳的header中加入CORS
    def custom_start_response(status, headers, exc_info=None):
        # 在原本 headers 基礎上，加上 CORS
        headers.append(('Access-Control-Allow-Origin', '*'))
        return start_response(status, headers, exc_info)
    
    # 路由判斷
    if path == '/start' and method == 'POST':
        return handle_start(environ, start_response)
    elif path == '/runTurn' and method == 'POST':
        return handle_run_turn(environ, start_response)
    elif path == '/status' and method == 'GET':
        return handle_status(environ, start_response)
    elif path == '/isGameOver' and method == 'GET':
        return handle_is_game_over(environ, start_response)
    else:
        # 其他路徑一律回傳 404
        custom_start_response('404 Not Found', [('Content-Type', 'application/json')])
        return [json.dumps({"error": "Not Found"}).encode('utf-8')]

def handle_start(environ, start_response):
    global GAME
    GAME = Game()  # 重新建立一個新遊戲
    data = {
        "message": "Game Started!",
        "gameState": GAME.to_dict()
    }
    # ★ 這裡多加一個 Access-Control-Allow-Origin
    start_response('200 OK', [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*')
    ])
    return [json.dumps(data).encode('utf-8')]


def handle_run_turn(environ, start_response):
    if GAME is None:
        start_response('400 Bad Request', [('Content-Type', 'application/json'), ('Access-Control-Allow-Origin', '*')])
        return [json.dumps({"error": "Game not started"}).encode('utf-8')]
    
    if GAME.turn >= GAME.max_turns:
        start_response('400 Bad Request', [('Content-Type', 'application/json'), ('Access-Control-Allow-Origin', '*')])
        return [json.dumps({"error": "Game is already over"}).encode('utf-8')]
    
    GAME.run_turn()
    data = {
        "message": f"Turn {GAME.turn} finished",
        "gameState": GAME.to_dict()
    }
    start_response('200 OK', [('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*')
                              ])
    return [json.dumps(data).encode('utf-8')]

def handle_status(environ, start_response):
    if GAME is None:
        start_response('400 Bad Request', [('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*')
                                           ])
        return [json.dumps({"error": "Game not started"}).encode('utf-8')]

    data = {
        "gameState": GAME.to_dict()
    }
    start_response('200 OK', [('Content-Type', 'application/json'), ('Access-Control-Allow-Origin', '*')])
    return [json.dumps(data).encode('utf-8')]

def handle_is_game_over(environ, start_response):
    if GAME is None:
        start_response('400 Bad Request', [('Content-Type', 'application/json')])
        return [json.dumps({"error": "Game not started"}).encode('utf-8')]
    
    is_over = (GAME.turn >= GAME.max_turns)
    data = {"isOver": is_over}
    start_response('200 OK', [('Content-Type', 'application/json')])
    return [json.dumps(data).encode('utf-8')]


# ======================
# 3. 啟動簡易的HTTP伺服器
# ======================
if __name__ == "__main__":
    httpd = make_server('127.0.0.1', 5000, simple_app)
    print("Serving on http://127.0.0.1:5000 ...")
    httpd.serve_forever()
