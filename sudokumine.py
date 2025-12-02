#!/usr/bin/env python3
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

GRID_SIZE = 9
BLOCK_SIZE = 3
MINES_PER_ROW = 3
MINES_PER_COL = 3
MINES_PER_BLOCK = 3


@dataclass
class SumConstraint:
    scope: list  # list of variable ids
    target: int  # required sum

    def is_consistent(self, assignment):
        """
        Check this constraint against current (partial) assignment.

        - sum of assigned values must not exceed target
        - even if all remaining variables become 1, must still be able
          to reach target
        """
        s_assigned = 0
        remaining_unassigned = 0
        for v in self.scope:
            if v in assignment:
                s_assigned += assignment[v]
            else:
                remaining_unassigned += 1

        # Too many mines already
        if s_assigned > self.target:
            return False

        # Even if we put a mine in every remaining variable,
        # we still must be able to reach target
        if s_assigned + remaining_unassigned < self.target:
            return False

        return True


class SudokuMineCSP:
    def __init__(self, clues_grid):
        """
        clues_grid: 9x9 int matrix.
        0 = blank (potential mine cell, variable),
        1..8 = numbered cell (not a variable).
        """
        self.clues = clues_grid

        # Maps variable id -> (row, col)
        self.var_pos = {}
        # Reverse: (row, col) -> var id
        self.pos_var = {}

        self.variables = []
        self._create_variables()

        # Domains: {var: set({0,1})}
        self.domains = {v: {0, 1} for v in self.variables}

        # Constraints & neighbor structure
        self.constraints = []           # list[SumConstraint]
        self.var_constraints = defaultdict(list)  # var -> list[SumConstraint]
        self.neighbors = {v: set() for v in self.variables}

        self._create_constraints()

    def _create_variables(self):
        """Create a variable for each blank cell (0 in clues grid)."""
        vid = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.clues[r][c] == 0:
                    self.variables.append(vid)
                    self.var_pos[vid] = (r, c)
                    self.pos_var[(r, c)] = vid
                    vid += 1

    def _add_constraint(self, scope, target):
        if not scope:  # no variables participating; skip
            return
        cons = SumConstraint(scope=scope, target=target)
        self.constraints.append(cons)
        for v in scope:
            self.var_constraints[v].append(cons)
        # For neighbors: all variables sharing this constraint are neighbors
        for i in range(len(scope)):
            for j in range(i + 1, len(scope)):
                v1, v2 = scope[i], scope[j]
                self.neighbors[v1].add(v2)
                self.neighbors[v2].add(v1)

    def _create_constraints(self):
        # Row constraints
        for r in range(GRID_SIZE):
            scope = []
            for c in range(GRID_SIZE):
                if (r, c) in self.pos_var:
                    scope.append(self.pos_var[(r, c)])
            self._add_constraint(scope, MINES_PER_ROW)

        # Column constraints
        for c in range(GRID_SIZE):
            scope = []
            for r in range(GRID_SIZE):
                if (r, c) in self.pos_var:
                    scope.append(self.pos_var[(r, c)])
            self._add_constraint(scope, MINES_PER_COL)

        # Block constraints (3x3)
        for br in range(0, GRID_SIZE, BLOCK_SIZE):
            for bc in range(0, GRID_SIZE, BLOCK_SIZE):
                scope = []
                for r in range(br, br + BLOCK_SIZE):
                    for c in range(bc, bc + BLOCK_SIZE):
                        if (r, c) in self.pos_var:
                            scope.append(self.pos_var[(r, c)])
                self._add_constraint(scope, MINES_PER_BLOCK)

        # Number (neighbor) constraints
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = self.clues[r][c]
                if val > 0:
                    scope = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                                if (nr, nc) in self.pos_var:
                                    scope.append(self.pos_var[(nr, nc)])
                    self._add_constraint(scope, val)


def select_unassigned_variable(csp, assignment, domains):
    """
    MRV + Degree heuristic.
    """
    unassigned = [v for v in csp.variables if v not in assignment]
    # MRV
    min_domain_size = min(len(domains[v]) for v in unassigned)
    mrv_candidates = [v for v in unassigned
                      if len(domains[v]) == min_domain_size]

    if len(mrv_candidates) == 1:
        return mrv_candidates[0]

    # Degree heuristic: max number of unassigned neighbors
    best = None
    best_degree = -1
    for v in mrv_candidates:
        degree = sum(1 for n in csp.neighbors[v] if n not in assignment)
        if degree > best_degree:
            best_degree = degree
            best = v
    return best


def order_domain_values(var, domains):
    # The assignment says: just use domain in order {0, 1}.
    return sorted(domains[var])


def is_consistent(csp, var, value, assignment):
    """
    Check whether assigning var=value keeps all constraints consistent.
    """
    assignment[var] = value
    # Only need to check constraints involving var
    for cons in csp.var_constraints[var]:
        if not cons.is_consistent(assignment):
            del assignment[var]
            return False
    del assignment[var]
    return True


def forward_check(csp, var, assignment, domains):
    """
    Forward Checking:

    After assigning var, revise domains of its neighbors.
    If any neighbor domain becomes empty, return False.
    """
    for nbr in csp.neighbors[var]:
        if nbr in assignment:
            continue
        to_remove = set()
        for d in domains[nbr]:
            assignment[nbr] = d
            # Check constraints of nbr
            ok = True
            for cons in csp.var_constraints[nbr]:
                if not cons.is_consistent(assignment):
                    ok = False
                    break
            del assignment[nbr]
            if not ok:
                to_remove.add(d)
        if to_remove:
            domains[nbr] = domains[nbr] - to_remove
        if not domains[nbr]:
            return False
    return True


def backtrack(csp, assignment, domains, depth, stats):
    """
    Backtracking with Forward Checking.
    stats["nodes"] counts total nodes generated.
    Returns (solution_assignment, goal_depth) or (None, None).
    """
    stats["nodes"] += 1

    # If assignment complete, we are at a goal
    if len(assignment) == len(csp.variables):
        return assignment, depth

    var = select_unassigned_variable(csp, assignment, domains)

    for value in order_domain_values(var, domains):
        if is_consistent(csp, var, value, assignment):
            # Copy domains for recursive call
            new_domains = deepcopy(domains)
            assignment[var] = value

            if forward_check(csp, var, assignment, new_domains):
                result, goal_depth = backtrack(
                    csp, assignment, new_domains, depth + 1, stats
                )
                if result is not None:
                    return result, goal_depth

            # undo assignment
            del assignment[var]

    # failure
    return None, None


def solve_sudoku_mine(clues_grid):
    csp = SudokuMineCSP(clues_grid)
    assignment = {}
    domains = deepcopy(csp.domains)
    stats = {"nodes": 0}

    solution, goal_depth = backtrack(csp, assignment, domains, depth=0, stats=stats)

    if solution is None:
        raise ValueError("No solution found")

    return csp, solution, goal_depth, stats["nodes"]


def read_input(filename):
    grid = []
    with open(filename, "r") as f:
        for _ in range(GRID_SIZE):
            line = f.readline()
            if not line:
                break
            row = [int(x) for x in line.strip().split()]
            if len(row) != GRID_SIZE:
                raise ValueError("Each row must have 9 integers.")
            grid.append(row)
    if len(grid) != GRID_SIZE:
        raise ValueError("Input must have 9 rows.")
    return grid


def write_output(filename, csp, assignment, goal_depth, nodes_generated):
    # Create 9x9 mine grid of 0/1
    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    # numbered cells stay 0 (no mine)
    for v, val in assignment.items():
        r, c = csp.var_pos[v]
        board[r][c] = val

    with open(filename, "w") as f:
        f.write(str(goal_depth) + "\n")
        f.write(str(nodes_generated) + "\n")
        for r in range(GRID_SIZE):
            line = " ".join(str(board[r][c]) for c in range(GRID_SIZE))
            f.write(line + "\n")


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python sudoku_mine.py input.txt output.txt")
        return

    in_file, out_file = sys.argv[1], sys.argv[2]
    clues_grid = read_input(in_file)
    csp, assignment, depth, nodes = solve_sudoku_mine(clues_grid)
    write_output(out_file, csp, assignment, depth, nodes)


if __name__ == "__main__":
    main()
