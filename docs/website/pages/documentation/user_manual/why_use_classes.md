# Why use Classes?

## What is a Python Class?

A Python `class` is like a template or blueprint for creating objects — in this example, our class 
creates tools to check if the condensation is likely to occur based on temperature and dew point at
a certain location.

Think of it like this: <br>
- A `function` does one task. <br>
- A `class` bundles together data and functions (called methods) that relate to a single concept — 
like checking condensation.

For example:
```py
class CondensationChecker:
    def __init__(self, location, temperature_c, dew_point_c):
        self.location = location
        self.temperature_c = temperature_c
        self.dew_point_c = dew_point_c

    def check_condensation(self):
        return self.temperature_c <= self.dew_point_c

    def report(self):
        # Prints the current condition and whether condensation is likely
```

Here: <br>
The data: temperature, dew point, location <br>
The behavior: check_condensation(), report() <br>

## What are Classes used for?

In this example, we use a `class` to create objects that represent weather conditions that will be
used to check if condensation is likely to occur.

```py
station = CondensationChecker("Berlin", 12.0, 12.5)
station.report()
```

The object `station` holds its own data and can run its own analysis on the data provided.


## Why use a Class instead of just functions?

Let's compare using a class vs. just using functions.

With just functions:

```py
def check_condensation(temp, dew_point):
    return temp <= dew_point
```

You’d need to manually pass temperature and dew point every single time, and it doesn't naturally 
group this data together. You also can't easily add more features like location, time, or logging.

With a `class`:
```py
station = CondensationChecker("Berlin", 12.0, 12.5)
station.report()
```
This is a much cleaner option — all related data is stored inside the object, and methods operate 
on that data.

## Benefits of using a Class

**Organization**

A class brings data and the functions that operate on it together in one place.

In CondensationChecker: <br>
The data: temperature, dew point, location <br>
The behavior: check_condensation(), report() <br>

These things belong together logically. Instead of keeping temperature and dew point as separate 
variables and writing separate functions, the class keeps them bundled as one logical unit.


**Reusability**

You can reuse the same class to create multiple objects representing different conditions — 
without repeating code.

Example:
```py
station1 = CondensationChecker("London", 15, 12)
station2 = CondensationChecker("Barcelona", 28, 27)

station1.report()
station2.report()
```

Each station is independent, and the logic to check condensation is shared and reusable.

Without a class, you'd have to manage multiple sets of variables manually and pass them into 
functions every time — more error-prone and harder to manage.


**Easy to Add New Features**

When your code grows in complexity, classes make it easy to add new functionality without breaking 
existing logic.

For example, you can add a method to estimate relative humidity based on existing data:
```py
def estimate_relative_humidity(self):
    # Use formula here
```

This method now becomes part of the condensation checker — you don’t have to touch outside code.


**Modularity**

Classes act like building blocks for larger systems. You can isolate pieces of your program into 
logical units.


**Maintainability and Cleanliness: Easier to Read, Debug, and Scale**

When your project grows, keeping code maintainable is very important.

With a class: <br>
- Each part of the program has a clear purpose. <br>
- You can fix or update just one class without affecting others. <br>
- You don’t have to trace global variables across multiple files. <br>

If someone new joins your team, they can understand what CondensationChecker does just by reading 
that one class.

