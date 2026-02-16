

class Objective:
    
    def __init__(self, device=None):
        if device is None:
            device = "cpu"
        
        self.device = device
        
    def compute(self, x, targets):
        raise NotImplementedError()
