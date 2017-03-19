import socket
import json
import time

class Client:
    msg_config_speed = json.dumps({'config': {'speed': '%s'}})
    msg_config_divisor = json.dumps({'config': {'divisor': '%d'}})

    msg_press_right = json.dumps({'key': {'value': 'Right'}})
    msg_press_left = json.dumps({'key': {'value': 'Left'}})
    msg_press_up = json.dumps({'key': {'value': 'Up'}})
    msg_press_down = json.dumps({'key': {'value': 'Down'}})
    msg_press_start = json.dumps({'key': {'value': 'Start'}})
    msg_press_A = json.dumps({'key': {'value': 'A'}})
    msg_press_B = json.dumps({'key': {'value': 'B'}})

    msg_game_reset = json.dumps({'game': {'value': 'Reset'}})
    msg_game_info = json.dumps({'game': {'value': 'Info'}})
    msg_game_tiles = json.dumps({'game': {'value': 'Tiles'}})

    buffer_size = 1024

    def __init__(self, addr='127.0.0.1', port=4561):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((addr, port))

    def send(self, msg):
        self.client.send(msg + '\r\n\r\n')

    def config_speed(self, speed='maximum'):
        return self.msg_config_speed % speed

    def config_divisor(self, divisior=2):
        return self.msg_config_divisor % divisior

    def reset(self):
        self.send(self.config_speed())
        self.send(self.config_divisor())
        self.send(self.msg_press_right)
        self.send(self.msg_game_reset)

    def info(self):
        self.send(self.msg_game_info)
        data_raw = self.client.recv(self.buffer_size)
        game_info = json.loads(data_raw)
        game_info['tiles'] = self.parse_tiles(game_info['tiles'])
        return game_info

    def parse_tiles(self, tiles_info):
        tiles = []
        for i in range(-6, 7, 1):
            tiles.extend(tiles_info[str(i)])
        return tiles
