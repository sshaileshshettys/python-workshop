

## Unit 1: Introduction to Programming in Python

### 1. Programming Basics

#### What is Programming?
- Definition and purpose: Writing instructions for a computer to perform tasks.
- Examples of programming in daily life: Automating repetitive tasks, creating applications, etc.

#### Brief History of Python
- Origins and development: Created by Guido van Rossum in 1991.
- Key features and philosophy: Readability, simplicity.
- Popular use cases: Web development, data analysis, AI, automation.

### 2. Variables, Expressions, and Statements

#### Variables
```python
# Defining variables
name = "John"
age = 25
height = 5.9

# Using variables
print(name)
print(age)
print(height)
```

#### Expressions
```python
# Arithmetic expressions
a = 5
b = 10
sum = a + b
product = a * b
print("Sum:", sum)
print("Product:", product)

# Combining variables and constants
c = a + 2
print("a + 2 =", c)
```

#### Statements
```python
# Assignment statement
x = 10

# Print statement
print(x)
```

### 3. Conditional Executions and Iterations

#### Conditional Executions

**`if` Statements**
```python
# Syntax and examples
number = 10
if number > 0:
    print("Positive number")
```

**`else` Statements**
```python
# Syntax and examples
number = -5
if number > 0:
    print("Positive number")
else:
    print("Negative number")
```

**`elif` Statements**
```python
# Syntax and examples
number = 0
if number > 0:
    print("Positive number")
elif number == 0:
    print("Zero")
else:
    print("Negative number")
```

#### Iterations

**`for` Loops**
```python
# Syntax and examples
# Looping through lists
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Looping through ranges
for i in range(5):
    print(i)
```

**`while` Loops**
```python
# Syntax and examples
# Common use cases and pitfalls
count = 0
while count < 5:
    print(count)
    count += 1
```

### 4. Basic Data Types and Operators

#### Data Types

**Integers**
```python
# Definition and examples
a = 10
b = -5
print(a)
print(b)
```

**Strings**
```python
# Definition, examples, and basic operations
text = "Hello, World!"
print(text)

# Concatenation
greeting = "Hello"
name = "Alice"
message = greeting + " " + name
print(message)

# Slicing
print(text[0:5])
```

**Booleans**
```python
# Definition and examples
is_python_fun = True
is_sky_blue = False
print(is_python_fun)
print(is_sky_blue)
```

#### Operators

**Arithmetic Operators**
```python
# Addition
a = 10
b = 5
print(a + b)

# Subtraction
print(a - b)

# Multiplication
print(a * b)

# Division
print(a / b)

# Modulus
print(a % b)

# Exponentiation
print(a ** 2)
```

**Comparison (Relational) Operators**
```python
# Equal to
print(a == b)

# Not equal to
print(a != b)

# Greater than
print(a > b)

# Less than
print(a < b)
```

**Assignment Operators**
```python
# Basic assignment
c = 10

# Compound assignment
c += 5
print(c)
```

**Logical Operators**
```python
# And
print(True and False)

# Or
print(True or False)

# Not
print(not True)
```

**Bitwise Operators**
```python
# And
print(5 & 3)

# Or
print(5 | 3)

# Xor
print(5 ^ 3)

# Not
print(~5)

# Shift operators
print(5 << 1)
print(5 >> 1)
```

**Membership Operators**
```python
# in
fruits = ["apple", "banana", "cherry"]
print("banana" in fruits)

# not in
print("grape" not in fruits)
```

**Identity Operators**
```python
# is
x = [1, 2, 3]
y = [1, 2, 3]
print(x is y)

# is not
print(x is not y)
```

### 5. Expressions

#### Defining Expressions
```python
# Combining variables and operators
x = 10
y = 5
result = x + y * 2
print(result)
```

#### Evaluating Expressions
```python
# Order of operations (precedence)
result = (x + y) * 2
print(result)
```

### Hands-On Exercises and Examples

#### Exercise 1: Variable Declaration and Basic Arithmetic
```python
# Define variables
a = 5
b = 10

# Perform arithmetic operations
sum = a + b
product = a * b

# Print results
print("Sum:", sum)
print("Product:", product)
```

#### Exercise 2: Conditional Statements
```python
# Check if a number is positive, negative, or zero
number = int(input("Enter a number: "))
if number > 0:
    print("Positive number")
elif number == 0:
    print("Zero")
else:
    print("Negative number")

# Categorize ages
age = int(input("Enter your age: "))
if age < 13:
    print("Child")
elif age < 20:
    print("Teenager")
else:
    print("Adult")
```

#### Exercise 3: Loops
```python
# Print numbers 1 to 10
for i in range(1, 11):
    print(i)

# Sum numbers until a condition is met
total = 0
count = 1
while count <= 10:
    total += count
    count += 1
print("Total sum:", total)
```

#### Exercise 4: Data Types and Operators
```python
# Demonstrate usage of different data types and operators
name = "Alice"
age = 30
is_student = False

# String concatenation
message = "Hello, my name is " + name + "."
print(message)

# Arithmetic operations
year_of_birth = 2023 - age
print("Year of Birth:", year_of_birth)

# Logical operation
print("Is Alice a student?", is_student)
```
