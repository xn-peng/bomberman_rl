def log_debug(self, content):
    # print("DEBUG: " + str(content))
    self.logger.debug(content)

def log_info(self, content):
    # print("INFO: " + str(content))
    self.logger.info(content)

ACTION_INDEX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "WAIT": 4, "BOMB": 5}
