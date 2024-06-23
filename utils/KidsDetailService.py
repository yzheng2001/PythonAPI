class Kid:
    def __init__(self, n, a):
        self._name = n
        self._age = a
    @property
    def Name(self):
        return self._name
    @Name.setter
    def Name(self, value):
        self._name = value

    @property
    def Age(self):
        return self._age
    @Age.setter
    def Age(self, value):
        self._age = value;

class KidDetail:
    def __init__(self,n, a):
        self.Name  = n
        self.Age = a
    def details(self):
        print("kid with name " + self.Name +" is of age " + self.Age)

class KidsDetailsService:
    async def get_KidsDetails(self):
        kid_details = KidDetail("Charlotte", 10)
        kid_details.Age = 1
        kid_details.Name ="Cheryl"
        kid_details.details()
        kids = []        
        kids.append(kid_details)
        kids.append(KidDetail("charlotte", 10))
        kids.append(KidDetail("chloe", 15))
        return kids
