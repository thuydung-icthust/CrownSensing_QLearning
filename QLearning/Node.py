import math
from scipy.spatial import distance
import Parameter as para
from Node_Method import to_string, find_receiver, update_postion


class Node:
    def __init__(self, car_name=None, location_list=None, cover_radius=para.cover_radius, cover_time=para.cover_time, current_velocity=None,
                 energy=100.0, prob=para.initial_prob, avg_energy=0.0, id=None, is_active=True):
        self.location_list = location_list
        self.car_name = car_name
        self.longitude = 0.0
        self.latitude = 0.0
        self.cover_radius = cover_radius
        self.cover_time = cover_time
        self.energy = energy
        self.prob = prob  # freq to sent message
        self.total_sending = 0  # total messages sent to gnb from last checkpoint
        self.used_energy = 0.0  # energy was used from last check point to now
        self.check_point = [{"E_current": self.energy,
                             "total_sending": self.total_sending}]
        # self.len_cp = len_cp  # length of check point list
        self.id = id  # identify of sensor
        self.neighbor = []  # neighborhood of sensor
        self.is_active = is_active  # statement of sensor. If sensor dead, state is False
        self.min_energy = 0.02
        self.len_cp = 1000
        self.current_row = 0
        self.moving_step = (0.0, 0.0)
        update_postion(self, 0)

    def set_check_point(self, t):
        if len(self.check_point) >= self.len_cp:
            self.check_point.pop(0)
        self.check_point.append(
            {"E_current": self.energy, "time": t, "total_sending": self.total_sending - self.check_point[-1]["total_sending"]})
        # self.avg_energy = self.check_point[-1]["avg_e"]
        self.used_energy = 0.0
        self.total_sending = 0

    def send(self, net=None, package=None, receiver=find_receiver, is_energy_info=False):
        # if not self.is_active:
        #     return
        # d0 = math.sqrt(para.EFS / para.EMP)
        receiver_id = receiver(self, net)
        # d = distance.euclidean(self.location, net.node[receiver_id].location)
        package.update_path(self.id)
        package.is_success = True
        self.total_sending += 1
        package.update_path(receiver_id)

    def get_prob(self):
        return self.prob

    def receive(self, package):
        self.energy -= para.ER * package.size
        self.used_energy += para.ER * package.size

    def check_active(self, net):
        if self.energy < self.min_energy:
            self.is_active = False

    def print_node(self, func=to_string):
        func(self)

    def update_prob(self, new_prob=None):
        self.prob = new_prob

    # def update_node_location(self, new_location = None):
    #     self.location = new_location

    def update_loc(self, long=None, lat=None):
        self.longitude = long
        self.latitude = lat

    def update_position(self, time_s=None):
        update_postion(self, time_s)

    def description(self):
        to_string(self)
