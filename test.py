class GFG: 
    a = 4 
    def __init__(self): 
        self.x=1
        self.y=self.func()
        self.z = 3

    def func(self):
        self.z = 2
        return self.z

    def setA(self,val):
        GFG.a = val

    @classmethod
    def clsFunc(cls):
        cls.a = 5

      
if __name__ == "__main__": 
    print(GFG.a)
    GFG.clsFunc()
    print(GFG.a)