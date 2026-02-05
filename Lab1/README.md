# ML4Cyber Labs

**EEP 567: Machine Learning for Cybersecurity**

Welcome! This repository contains lab materials for EEP 567, covering foundational Python programming skills essential for machine learning applications in cybersecurity.

---

## 📋 Table of Contents

- [Quick Start (5 minutes)](#-quick-start-5-minutes)
- [Lab 1 Overview](#-lab-1-python-fundamentals)
- [Getting Started](#-getting-started)
- [How to Approach Exercises](#-how-to-approach-exercises)
- [Quick Reference Cheat Sheets](#-quick-reference-cheat-sheets)
- [Troubleshooting Common Errors](#-troubleshooting-common-errors)
- [Tips for Success](#-tips-for-success)
- [Getting Help](#-need-help)
- [Additional Resources](#-additional-resources)

---

## ⚡ Quick Start (5 minutes)

**Already know Python?** Jump right in:

```bash
# 1. Install packages (if needed)
pip install numpy scipy pandas matplotlib

# 2. Launch Jupyter
jupyter notebook

# 3. Open Lab1/lab-1-student-20260107.ipynb

# 4. Look for ## [ TODO ] markers for exercises
```

**New to Python?** Read on for a gentler introduction! 👇

---

## 🧪 Lab 1: Python Fundamentals

### Overview

Lab 1 introduces the Python programming language and essential libraries for data science and machine learning. This foundational lab prepares you for more advanced topics in subsequent labs.

### What You'll Learn

#### Section 1: Python Basics
| Topic | Description | Why It Matters |
|-------|-------------|----------------|
| **Variables** | Dynamic typing, assignment | Store and manipulate data |
| **Data Types** | `int`, `float`, `str`, `bool`, `None` | Understand what kind of data you're working with |
| **Collections** | `list`, `tuple`, `set`, `dict` | Organize multiple pieces of data |
| **Control Flow** | `if-else`, `for`, `while` | Make decisions and repeat actions |
| **Functions** | Definition, arguments, returns | Reuse code and stay organized |

#### Section 2: Libraries
| Library | Purpose | Real-World Use |
|---------|---------|----------------|
| **NumPy** | Fast array operations | Processing large datasets |
| **Pandas** | Data manipulation | Analyzing CSV files, databases |
| **Matplotlib** | Visualization | Creating charts and graphs |

### The 4 Exercises

| # | Exercise | Difficulty | Time Est. |
|:-:|----------|:----------:|:---------:|
| 1 | NumPy Slicing | ⭐ Easy | 5 min |
| 2 | NumPy Broadcasting | ⭐⭐ Medium | 10 min |
| 3 | Pandas Filtering | ⭐⭐ Medium | 10 min |
| 4 | Matplotlib 3D Plot | ⭐ Easy | 5 min |

Each exercise includes:
- ✅ Clear instructions
- ✅ Step-by-step hints
- ✅ Expected output examples

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8 or higher** (Check: `python --version`)
- **Jupyter Notebook** or JupyterLab

### Step-by-Step Installation

#### Step 1: Check Python Version
```bash
python --version
# Should show Python 3.8 or higher
```

If you don't have Python, download from [python.org](https://www.python.org/downloads/) or install [Anaconda](https://www.anaconda.com/download).

#### Step 2: Install Required Packages
```bash
pip install numpy scipy pandas matplotlib jupyter
```

**Using Anaconda?** These are pre-installed! Just run:
```bash
conda activate base  # or your environment name
```

#### Step 3: Download Lab Materials
- Download from your course website, OR
- Clone from repository (if provided)

#### Step 4: Launch Jupyter
```bash
cd path/to/ML4Cyber-Labs
jupyter notebook
```

Your browser should open automatically. If not, go to `http://localhost:8888`

#### Step 5: Open the Lab
Navigate to: `Lab1/` → `lab-1-student-20260107.ipynb`

---

## 🎯 How to Approach Exercises

### The 4-Step Method

#### Step 1: Read the TODO Carefully
```python
## [ TODO ] ============================================================
# Slice a sub-matrix of `a`, such that it only contains elements
# from the first two rows and the second and third columns
#
# Hint: Use the slicing syntax a[row_start:row_end, col_start:col_end]
#       Remember: end indices are exclusive!
#
# Expected output:
# [[2 3]
#  [6 7]]
# ======================================================================
```

**Identify:**
- What are you being asked to do?
- What hints are provided?
- What should the output look like?

#### Step 2: Look Back at Examples
Scroll up in the notebook! There's always an example showing similar concepts.

#### Step 3: Try Something
Don't be afraid to experiment! You can always re-run the cell.

```python
# Even a wrong attempt helps you learn:
a_sub = a[0:2]  # Try this, see what happens
print(a_sub)    # Check the result
```

#### Step 4: Check Your Output
Compare your output to the "Expected output" in the TODO.

### If You're Stuck

| Try This | Why It Helps |
|----------|--------------|
| `print(variable)` | See what's inside a variable |
| `type(variable)` | Check what kind of data it is |
| `variable.shape` | For NumPy arrays, see dimensions |
| `help(function)` | Get documentation |
| Read the error message | Python often tells you exactly what's wrong! |

---

## 📋 Quick Reference Cheat Sheets

### Python Indexing (SUPER IMPORTANT!)

```
String/List:  "abcdefg"
              
Index:         0  1  2  3  4  5  6
              [a][b][c][d][e][f][g]
Negative:     -7 -6 -5 -4 -3 -2 -1

Slicing: [start:end]  ← end is EXCLUSIVE!

Examples:
  "abcdefg"[0]     → 'a'     (first element)
  "abcdefg"[-1]    → 'g'     (last element)
  "abcdefg"[1:4]   → 'bcd'   (indices 1,2,3)
  "abcdefg"[:3]    → 'abc'   (first 3)
  "abcdefg"[3:]    → 'defg'  (from index 3 to end)
```

### NumPy 2D Array Indexing

```
Array a:
        Col 0  Col 1  Col 2  Col 3
Row 0  [  1      2      3      4  ]
Row 1  [  5      6      7      8  ]
Row 2  [  9     10     11     12  ]

a[row, col]           → single element
a[row_start:row_end, col_start:col_end] → sub-matrix
a[:, col]             → entire column (all rows)
a[row, :]             → entire row (all columns)

Examples:
  a[0, 1]      → 2        (row 0, col 1)
  a[:, -1]     → [4,8,12] (last column)
  a[:2, 1:3]   → [[2,3], [6,7]] (first 2 rows, cols 1-2)
```

### Common Pandas Operations

```python
df["column"]              # Select one column
df[["col1", "col2"]]      # Select multiple columns
df[df["col"] > value]     # Filter rows by condition
df.loc[condition, "col"]  # Filter + select column
df.head(n)                # First n rows
df.describe()             # Summary statistics
```

### Matplotlib Basics

```python
# 2D Plot
plt.plot(x, y)
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.title("Title")
plt.legend()
plt.show()

# 3D Plot (use ax instead of plt)
ax.scatter(x, y, z, label="Name")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Title")
ax.legend()
```

---

## 🔧 Troubleshooting Common Errors

### "ModuleNotFoundError: No module named 'numpy'"

**Problem:** Package not installed

**Solution:**
```bash
pip install numpy pandas matplotlib scipy
```

---

### "NameError: name 'x' is not defined"

**Problem:** Variable doesn't exist (yet)

**Solutions:**
1. Run the cells above first! (Shift+Enter)
2. Check spelling - Python is case-sensitive (`X` ≠ `x`)
3. Make sure you defined the variable

---

### "SyntaxError: invalid syntax"

**Problem:** Python can't understand your code

**Common causes:**
- Missing colon after `if`, `for`, `while`, `def`
- Mismatched parentheses `(` `)` or brackets `[` `]`
- Using `=` instead of `==` for comparison

---

### "IndexError: index out of bounds"

**Problem:** Trying to access an element that doesn't exist

**Solution:** Check your indices! Remember:
- Indexing starts at 0
- `arr[5]` means the 6th element
- Use `len(arr)` or `arr.shape` to check size

---

### "TypeError: 'int' object is not subscriptable"

**Problem:** Using `[]` on something that's not a list/array

**Example of wrong code:**
```python
x = 5
print(x[0])  # Error! x is just a number
```

---

### Nothing happens when I run a cell

**Possible reasons:**
1. Cell is still running (see `[*]` on the left)
2. Code runs but produces no output (add `print()`)
3. Kernel died - Restart: Kernel → Restart

---

### Plot doesn't show up

**Solution:** Add this to a cell and run it:
```python
%matplotlib inline
```

---

## 💡 Tips for Success

### For Everyone

| Tip | Why |
|-----|-----|
| **Run cells in order** | Variables from earlier cells are needed later |
| **Read the comments** | They explain what the code does |
| **Don't skip examples** | They teach the concepts you need |
| **Save frequently** | Ctrl+S or Cmd+S |

### For Beginners

| Tip | Why |
|-----|-----|
| **Type code yourself** | Don't just read - muscle memory helps |
| **Experiment** | Change numbers and see what happens |
| **It's okay to fail** | Errors are part of learning! |
| **Go slow** | Understanding > Speed |

### For Experienced Programmers

| Tip | Why |
|-----|-----|
| **Note Python differences** | Indexing, no semicolons, whitespace matters |
| **Learn NumPy broadcasting** | It's powerful but different |
| **Check Pandas syntax** | It has its own conventions |

---

## ❓ Need Help?

### Self-Help (Try These First!)

1. **Re-read the TODO** - Hints are there for a reason
2. **Look at similar examples** - Earlier cells show the patterns
3. **Check the cheat sheets above** - Quick syntax reference
4. **Read the error message** - Python usually explains the problem
5. **Google the error** - Someone else has had the same issue!

### Ask for Help

- **Office Hours** - Best for conceptual questions
- **Lab Sessions** - TAs can help with specific issues
- **Classmates** - Collaboration is encouraged (but write your own code!)
- **Email** - For issues that need instructor attention

### Useful Search Tips

When Googling, try:
- `"python numpy" + your error message`
- `"pandas how to" + what you want to do`
- `"matplotlib" + type of plot you need`

---

## 📚 Additional Resources

### Official Documentation
- [Python Docs](https://docs.python.org/3/)
- [NumPy Docs](https://numpy.org/doc/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Matplotlib Docs](https://matplotlib.org/stable/contents.html)

### Tutorials
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Interactive Learning
- [Python Tutor](https://pythontutor.com/) - Visualize code execution
- [W3Schools Python](https://www.w3schools.com/python/) - Quick tutorials
- [Real Python](https://realpython.com/) - In-depth articles

---

## 🎉 You've Got This!

Remember:
- Every expert was once a beginner
- Errors are learning opportunities
- Programming is a skill - it gets easier with practice

Good luck with the lab! 🚀

---

*EEP 567 - Machine Learning for Cybersecurity*
