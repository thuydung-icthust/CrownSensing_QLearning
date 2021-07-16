import Parameter as para


class Package:
    def __init__(self, is_energy_info=False):
        self.path = []  # the path include nodes in path from node to base
        self.is_success = False  # is_success = True if package can go to base

    def get_size(self):
        return para.b if not self.is_energy_info else para.b_energy

    def update_path(self, node_id):
        self.path.append(node_id)
