
class BattlerTrainConfig:
    def __init__(self, num_servers: int):
        self.num_servers = num_servers

    def get_train_config(self):
        config = {
            "num_servers": self.num_servers,
        }
        return config