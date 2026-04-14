from .hessian import Hessian

class Filter:
    
    def __init__(self):
        self.filter = Hessian()
    
    def forward(self, data):

        return self.filter.process(data)
    

    