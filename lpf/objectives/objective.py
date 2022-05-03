

class Objective:
    
    def __init__(self, device=None):
        if not device:
            device = "cpu"
        
        self.device = device
        
    def compute(self, x, targets):
        raise NotImplementedError()
