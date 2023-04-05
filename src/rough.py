# import sys

# try:
#     a+b
# except Exception as e:
#     a,b,c=sys.exc_info()
#     print(sys.exc_info())
#     print(sys.exc_info()[0])
#     print(sys.exc_info()[1])
#     print(sys.exc_info()[2])
#     print(f"Error: {sys.exc_info()[0]}. Error Msg: {sys.exc_info()[1]}. Line: {sys.exc_info()[2].tb_lineno}")
#     print(sys.exc_info()[2].tb_frame)
#     print(sys.exc_info()[2].tb_frame.f_code)
#     print(sys.exc_info()[2].tb_frame.f_code.co_filename)


#Creating a class
class myClass:
    x=5

# creating an object that belongs to myClass

object1 = myClass()

print(object1.x)

class person:
    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age
        
p1 = person("Ramakanth", "28")


print(p1.name)
print(p1.age)

'''
Create a class named Person, 
with firstname and lastname properties, and a printname method:
'''

class Person():
    def __init__(self, fname, lname) -> None:
        self.firstname = fname
        self.lastname = lname
        pass

    def printname(self):
        print(self.firstname, self.lastname)

p1 = Person("Ramakanth", "Sharma")

p1.printname()

class student(Person):
    pass

s1 = student("Ashish", "Sharma")
s1.printname()


e = Exception

e.error