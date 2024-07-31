## Unit 6: Introduction to Object-Oriented Programming

### 1. Basics of Object-Oriented Programming

**What is OOP?**
- **Definition:**
  - Object-Oriented Programming (OOP) is a programming paradigm that uses objects and classes to structure software. It promotes encapsulation, inheritance, and polymorphism.
- **Key Concepts:**
  - **Classes:** Blueprints for creating objects.
  - **Objects:** Instances of classes.
  - **Attributes:** Data stored in an object.
  - **Methods:** Functions defined in a class that operate on objects.

### 2. Classes and Objects

**Defining a Class**

**Code Sample:**
```python
class Person:
    # Constructor method
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # Method to display person details
    def display_info(self):
        print(f"Name: {self.name}, Age: {self.age}")

# Creating an object of the Person class
person1 = Person("Alice", 30)
person1.display_info()
```

**Creating and Using Objects**

**Code Sample:**
```python
# Creating another object
person2 = Person("Bob", 25)
person2.display_info()
```

### 3. Inheritance

**What is Inheritance?**
- **Definition:**
  - Inheritance allows one class (child class) to inherit attributes and methods from another class (parent class), promoting code reuse.

**Code Sample:**
```python
class Employee(Person):
    def __init__(self, name, age, position):
        super().__init__(name, age)  # Call the constructor of the parent class
        self.position = position

    def display_info(self):
        super().display_info()
        print(f"Position: {self.position}")

# Creating an object of the Employee class
employee1 = Employee("Charlie", 28, "Software Engineer")
employee1.display_info()
```

### 4. Method Overriding

**What is Method Overriding?**
- **Definition:**
  - Method overriding allows a subclass to provide a specific implementation of a method that is already defined in its superclass.

**Code Sample:**
```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Dog barks")

# Creating objects
animal = Animal()
dog = Dog()

animal.speak()  # Output: Animal speaks
dog.speak()     # Output: Dog barks
```

### 5. Encapsulation

**What is Encapsulation?**
- **Definition:**
  - Encapsulation is the concept of restricting access to certain details of an object and only exposing necessary functionality. It is achieved using private and public access modifiers.

**Code Sample:**
```python
class BankAccount:
    def __init__(self, account_number, balance):
        self.__account_number = account_number  # Private attribute
        self.__balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew ${amount}")
        else:
            print("Insufficient balance")

    def get_balance(self):
        return self.__balance

# Creating an object
account = BankAccount("123456", 1000)
account.deposit(500)
account.withdraw(200)
print("Balance:", account.get_balance())
```

### 6. Abstraction

**What is Abstraction?**
- **Definition:**
  - Abstraction involves hiding complex implementation details and showing only the necessary features of an object. It is achieved using abstract classes and methods.

**Code Sample:**
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

# Creating an object
rectangle = Rectangle(10, 5)
print("Area of rectangle:", rectangle.area())
```

### 7. Exception Handling in OOP

**Handling Exceptions**

**Code Sample:**
```python
class CustomError(Exception):
    pass

class Calculator:
    def divide(self, a, b):
        if b == 0:
            raise CustomError("Division by zero is not allowed")
        return a / b

# Using the Calculator class
calc = Calculator()
try:
    result = calc.divide(10, 0)
except CustomError as e:
    print(e)
else:
    print("Result:", result)
```

### Hands-On Exercises and Examples

**Exercise 1: Create a Class and Object**
- Define a `Car` class with attributes `make`, `model`, and `year`. Implement a method `display_info` to print these details.

**Code Sample:**
```python
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def display_info(self):
        print(f"Make: {self.make}, Model: {self.model}, Year: {self.year}")

# Creating an object
car1 = Car("Toyota", "Corolla", 2022)
car1.display_info()
```

**Exercise 2: Implement Inheritance**
- Create a base class `Vehicle` and a derived class `Bike`. Implement the `display_info` method in both classes, and show how the `Bike` class inherits and extends functionality from `Vehicle`.

**Code Sample:**
```python
class Vehicle:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year

    def display_info(self):
        print(f"Brand: {self.brand}, Year: {self.year}")

class Bike(Vehicle):
    def __init__(self, brand, year, type_of_bike):
        super().__init__(brand, year)
        self.type_of_bike = type_of_bike

    def display_info(self):
        super().display_info()
        print(f"Type of Bike: {self.type_of_bike}")

# Creating an object
bike1 = Bike("Yamaha", 2021, "Sport")
bike1.display_info()
```

**Exercise 3: Demonstrate Encapsulation**
- Create a `Student` class with private attributes `name` and `grade`. Implement methods to get and set these attributes while ensuring data integrity.

**Code Sample:**
```python
class Student:
    def __init__(self, name, grade):
        self.__name = name
        self.__grade = grade

    def set_name(self, name):
        if name:
            self.__name = name

    def get_name(self):
        return self.__name

    def set_grade(self, grade):
        if 0 <= grade <= 100:
            self.__grade = grade

    def get_grade(self):
        return self.__grade

# Creating an object
student1 = Student("John", 85)
student1.set_name("Jane")
student1.set_grade(90)
print("Name:", student1.get_name())
print("Grade:", student1.get_grade())
```

**Exercise 4: Abstract Class and Method**
- Define an abstract class `Animal` with an abstract method `make_sound`. Create derived classes `Dog` and `Cat` that implement this method.

**Code Sample:**
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

# Creating objects
dog = Dog()
cat = Cat()
print("Dog sound:", dog.make_sound())
print("Cat sound:", cat.make_sound())
```
