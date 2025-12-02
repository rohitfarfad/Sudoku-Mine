# Sudoku Mine CSP Solver

Backtracking + CSP solver for the **Sudoku Mine** puzzle.  
Implements MRV + Degree heuristics and Forward Checking to place mines on a 9×9 board so that:

- Each row, column, and 3×3 block has exactly 3 mines.
- Numbered cells (1–8) equal the number of mines in their 8-neighborhood.
- Mines can only be on cells that are 0 in the input.

## Run

```bash
# From repo root
python3 sudoku_mine.py input.txt output.txt
