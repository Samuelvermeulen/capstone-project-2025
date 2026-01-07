"""
Basic Python Concepts Demonstration.
Covers control flow, functions, lists, dictionaries, and file I/O.
"""

# -----------------------------------------------------------------
# 1. Control Flow (if/else, loops)
# -----------------------------------------------------------------
print("=" * 50)
print("1. CONTROL FLOW DEMONSTRATION")
print("=" * 50)

# If-elif-else
def categorize_number(n):
    """Categorize a number."""
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n > 0 and n <= 10:
        return "small positive"
    else:
        return "large positive"

# Test
test_numbers = [-5, 0, 3, 15]
for num in test_numbers:
    print(f"{num}: {categorize_number(num)}")

# While loop with break
print("\nCounting with while loop:")
count = 0
while True:
    print(count, end=" ")
    count += 1
    if count >= 5:
        break
print()

# For loop with continue
print("\nEven numbers (0-9):")
for i in range(10):
    if i % 2 != 0:
        continue  # Skip odd numbers
    print(i, end=" ")
print()

# -----------------------------------------------------------------
# 2. Functions and Recursion
# -----------------------------------------------------------------
print("\n\n" + "=" * 50)
print("2. FUNCTIONS AND RECURSION")
print("=" * 50)

def factorial(n):
    """Calculate factorial using recursion."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(f"Factorial of 5: {factorial(5)}")
print(f"Factorial of 7: {factorial(7)}")

# Lambda function
square = lambda x: x ** 2
print(f"\nLambda function - square of 4: {square(4)}")

# Function as argument
def apply_function(func, value):
    """Apply a function to a value."""
    return func(value)

print(f"Apply square to 3: {apply_function(square, 3)}")

# -----------------------------------------------------------------
# 3. Lists, Tuples, Dictionaries
# -----------------------------------------------------------------
print("\n\n" + "=" * 50)
print("3. DATA STRUCTURES")
print("=" * 50)

# List operations
fruits = ["apple", "banana", "cherry", "date", "elderberry"]
print(f"Original list: {fruits}")

# List comprehension
fruit_lengths = [len(fruit) for fruit in fruits]
print(f"Length of each fruit: {fruit_lengths}")

# Slicing
print(f"First 3: {fruits[:3]}")
print(f"Last 2: {fruits[-2:]}")
print(f"Every other: {fruits[::2]}")

# Tuple
coordinates = (10, 20)
x, y = coordinates  # Tuple unpacking
print(f"\nCoordinates: {coordinates}")
print(f"Unpacked: x={x}, y={y}")

# Dictionary
student = {
    "name": "Alice",
    "age": 22,
    "courses": ["Math", "Physics", "CS"],
    "grades": {"Math": 85, "Physics": 90, "CS": 95}
}

print(f"\nStudent dictionary:")
for key, value in student.items():
    print(f"  {key}: {value}")

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(1, 6)}
print(f"\nSquares dictionary: {squares_dict}")

# -----------------------------------------------------------------
# 4. String Manipulation
# -----------------------------------------------------------------
print("\n\n" + "=" * 50)
print("4. STRING MANIPULATION")
print("=" * 50)

text = "  Data Science is amazing!  "
print(f"Original: '{text}'")
print(f"Stripped: '{text.strip()}'")
print(f"Upper: '{text.upper()}'")
print(f"Lower: '{text.lower()}'")
print(f"Replace: '{text.replace('amazing', 'awesome')}'")
print(f"Contains 'Science': {'Science' in text}")
print(f"Words: {text.strip().split()}")

# String formatting
name = "Samuel"
age = 25
print(f"\nFormatted: My name is {name} and I am {age} years old.")

# -----------------------------------------------------------------
# 5. File I/O Demonstration
# -----------------------------------------------------------------
print("\n\n" + "=" * 50)
print("5. FILE I/O OPERATIONS")
print("=" * 50)

# Write to a file
with open("results/demo_output.txt", "w") as f:
    f.write("This is a demonstration of file I/O.\n")
    f.write(f"Student: {student['name']}\n")
    f.write(f"Fruits: {', '.join(fruits)}\n")
    f.write("=" * 30 + "\n")
    for i in range(1, 6):
        f.write(f"Square of {i}: {i**2}\n")

print("File 'results/demo_output.txt' created successfully!")

# Read from the file
print("\nContents of the file:")
with open("results/demo_output.txt", "r") as f:
    for line in f:
        print(line.strip())

# -----------------------------------------------------------------
# 6. Error Handling
# -----------------------------------------------------------------
print("\n\n" + "=" * 50)
print("6. ERROR HANDLING")
print("=" * 50)

def safe_division(a, b):
    """Perform division with error handling."""
    try:
        result = a / b
        print(f"{a} / {b} = {result}")
        return result
    except ZeroDivisionError:
        print(f"Error: Cannot divide {a} by zero!")
        return None
    except TypeError as e:
        print(f"Type error: {e}")
        return None

# Test error handling
safe_division(10, 2)
safe_division(10, 0)
safe_division(10, "2")

print("\n" + "=" * 50)
print("DEMONSTRATION COMPLETE!")
print("=" * 50)
