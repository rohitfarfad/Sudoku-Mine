import sys

class SudokuMineCSP:
    def __init__(self, input_file):
        # Read the board
        self.board = []
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.board.append(list(map(int, line.strip().split())))
        except FileNotFoundError:
            print(f"Error: File {input_file} not found.")
            sys.exit(1)

        self.rows = 9
        self.cols = 9
        self.total_mines_per_unit = 3
        
        # CSP Components
        self.variables = [] # List of (r, c) tuples for empty cells (0)
        self.constraints = [] # List of numbered cells ((r,c), val)
        
        # Statistics
        self.nodes_generated = 0
        
        # Initialize Variables and Static Constraints
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.board[r][c]
                if val == 0:
                    self.variables.append((r, c))
                else:
                    self.constraints.append(((r, c), val))

    def get_neighbors_8(self, r, c):
        """Returns valid 8-neighbor coordinates for a cell (r,c)."""
        nbs = []
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    if (i, j) != (r, c):
                        nbs.append((i, j))
        return nbs

    def is_complete(self, assignment):
        return len(assignment) == len(self.variables)

    def is_consistent(self, var, value, assignment):
        """
        Checks if assigning value to var is consistent with current assignment.
        Constraints:
        1. Row/Col/Block cannot exceed 3 mines.
        2. Numbered cells cannot exceed their specific count in neighbors.
        """
        r, c = var
        
        # Temporarily add to assignment to check totals
        # (We treat the potential assignment as valid for calculation)
        current_val = value
        
        # 1. Check Row Constraint
        row_mines = sum(1 for j in range(self.cols) 
                        if assignment.get((r, j), 0) == 1) + (1 if current_val == 1 else 0)
        # Note: We only check existing assignment. 
        # Since we are iterating, we check if we *exceed*. 
        # We verify 'exact' match only at the end or if domains are exhausted, 
        # but for consistency, we just ensure we don't go OVER 3.
        if row_mines > 3: 
            return False

        # 2. Check Column Constraint
        col_mines = sum(1 for i in range(self.rows) 
                        if assignment.get((i, c), 0) == 1) + (1 if current_val == 1 else 0)
        if col_mines > 3: 
            return False

        # 3. Check Block Constraint
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_mines = 0
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if (i, j) in assignment and assignment[(i, j)] == 1:
                    block_mines += 1
        if current_val == 1:
            block_mines += 1
        if block_mines > 3: 
            return False

        # 4. Check Numbered Neighbors Constraints
        # We need to check neighbors of 'var' that are Numbered Cells.
        # If 'var' is placed near a '4', we ensure that '4' doesn't exceed 4 mines.
        neighbors = self.get_neighbors_8(r, c)
        for nr, nc in neighbors:
            # If neighbor is a numbered cell (constraint)
            # Find it in our constraint list or just check board > 0 (since vars are 0)
            # But wait, original board has numbers. 
            # Board values: 0 = variable, 1-8 = constraint.
            # We must verify against the ORIGINAL board value, not the assignment.
            n_val = self.board[nr][nc]
            if n_val > 0: # It is a constraint
                # Count mines around this numbered cell
                # Existing mines in assignment + this new one
                n_neighbors = self.get_neighbors_8(nr, nc)
                current_mines_around_n = 0
                for nnr, nnc in n_neighbors:
                    if (nnr, nnc) == (r, c):
                        if current_val == 1:
                            current_mines_around_n += 1
                    elif (nnr, nnc) in assignment and assignment[(nnr, nnc)] == 1:
                        current_mines_around_n += 1
                
                if current_mines_around_n > n_val:
                    return False
                    
        return True

    def select_unassigned_variable(self, assignment):
        """
        Selects next variable using MRV, breaking ties with Degree Heuristic.
        """
        unassigned = [v for v in self.variables if v not in assignment]
        
        # 1. MRV: Minimum Remaining Values
        # For binary domain, domain size is either 1 (forced) or 2.
        # We calculate valid values (0, 1) consistent with assignment.
        scored_vars = []
        
        for var in unassigned:
            valid_values = 0
            for val in [0, 1]:
                if self.is_consistent(var, val, assignment):
                    valid_values += 1
            
            # Degree Heuristic Calculation
            # "Two variables are neighbors if they share one or more common constraints"
            # Degree = number of constraints involving 'var' that contain OTHER unassigned variables.
            degree = 0
            r, c = var
            
            # Constraints: Row, Col, Block, Numbered Neighbors
            
            # Row
            if any((r, j) in unassigned and (r, j) != var for j in range(self.cols)):
                degree += 1
            # Col
            if any((i, c) in unassigned and (i, c) != var for i in range(self.rows)):
                degree += 1
            # Block
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_has_unassigned = False
            for i in range(br, br + 3):
                for j in range(bc, bc + 3):
                    if (i, j) in unassigned and (i, j) != var:
                        block_has_unassigned = True
                        break
            if block_has_unassigned:
                degree += 1
            
            # Numbered Neighbors
            # If var is adjacent to a number, and that number is adjacent to OTHER unassigned vars
            ns = self.get_neighbors_8(r, c)
            for nr, nc in ns:
                if self.board[nr][nc] > 0: # It's a constraint
                    # Check if this constraint has other unassigned vars
                    n_neighbors = self.get_neighbors_8(nr, nc)
                    if any(x in unassigned and x != var for x in n_neighbors):
                        degree += 1
            
            scored_vars.append({'var': var, 'mrv': valid_values, 'degree': degree})
            
        # Sort: Primary = MRV (Ascending), Secondary = Degree (Descending)
        scored_vars.sort(key=lambda x: (x['mrv'], -x['degree']))
        
        return scored_vars[0]['var']

    def inference(self, var, assignment):
        """
        Forward Checking.
        Returns a dict of forced assignments {var: val} or 'failure'.
        """
        inferences = {}
        
        # We need to check constraints affected by 'var' to see if they force neighbors
        r, c = var
        affected_constraints = []
        
        # 1. Global Constraints (Row, Col, Block)
        affected_constraints.append(('row', r))
        affected_constraints.append(('col', c))
        affected_constraints.append(('block', (r//3, c//3)))
        
        # 2. Local Constraints (Numbered neighbors of var)
        ns = self.get_neighbors_8(r, c)
        for nr, nc in ns:
            if self.board[nr][nc] > 0:
                affected_constraints.append(('cell', (nr, nc)))
        
        # Check all constraints
        # Note: In a full AC-3 we would queue these, but for FC we check immediate impact.
        # We check ALL global constraints and numbered cells to be safe and maximize inference,
        # or strictly just the affected ones. Checking affected is standard for FC.
        
        # To be robust, we iterate affected constraints and check if they are "tight"
        
        for c_type, c_val in affected_constraints:
            
            # Get variables in this constraint
            cons_vars = []
            target = 0
            
            if c_type == 'row':
                cons_vars = [(c_val, j) for j in range(self.cols) if self.board[c_val][j] == 0]
                target = 3
            elif c_type == 'col':
                cons_vars = [(i, c_val) for i in range(self.rows) if self.board[i][c_val] == 0]
                target = 3
            elif c_type == 'block':
                br, bc = c_val[0]*3, c_val[1]*3
                for i in range(br, br+3):
                    for j in range(bc, bc+3):
                        if self.board[i][j] == 0:
                            cons_vars.append((i, j))
                target = 3
            elif c_type == 'cell':
                target = self.board[c_val[0]][c_val[1]]
                cons_vars = [x for x in self.get_neighbors_8(c_val[0], c_val[1]) if self.board[x[0]][x[1]] == 0]

            # Calculate current status
            current_mines = 0
            unassigned_vars = []
            
            for cv in cons_vars:
                if cv in assignment:
                    if assignment[cv] == 1:
                        current_mines += 1
                elif cv in inferences:
                     if inferences[cv] == 1:
                         current_mines += 1
                else:
                    unassigned_vars.append(cv)
            
            remaining_needed = target - current_mines
            
            # Failure check
            if remaining_needed < 0:
                return "failure"
            if remaining_needed > len(unassigned_vars):
                return "failure"
            
            # Inference check
            if remaining_needed == 0:
                # All unassigned must be 0
                for uv in unassigned_vars:
                    inferences[uv] = 0
            elif remaining_needed == len(unassigned_vars):
                # All unassigned must be 1
                for uv in unassigned_vars:
                    inferences[uv] = 1
                    
        # Double check consistency of inferences (e.g. one constraint says 0, another says 1)
        # and re-validate against other constraints? 
        # Standard FC just returns. The recursive step will validate.
        # However, we should ensure we didn't infer a conflict immediately.
        
        return inferences

    def backtrack(self, assignment):
        self.nodes_generated += 1
        
        if self.is_complete(assignment):
            return assignment
            
        var = self.select_unassigned_variable(assignment)
        
        # Order Domain Values: 0 then 1
        for value in [0, 1]:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                
                # Inference
                inferences = self.inference(var, assignment)
                
                if inferences != "failure":
                    # Add inferences to assignment
                    # Must be careful not to overwrite valid assignments or conflict
                    valid_inf = True
                    added_vars = []
                    
                    for invar, inval in inferences.items():
                        if invar in assignment:
                            if assignment[invar] != inval:
                                valid_inf = False
                                break
                        else:
                            # Check consistency of inference before adding?
                            # Optimisation: Assume inference is consistent with triggering constraint,
                            # but check basic consistency.
                            if not self.is_consistent(invar, inval, assignment):
                                valid_inf = False
                                break
                            assignment[invar] = inval
                            added_vars.append(invar)
                    
                    if valid_inf:
                        result = self.backtrack(assignment)
                        if result != "failure":
                            return result
                    
                    # Backtrack: Remove inferences
                    for v_ in added_vars:
                        del assignment[v_]
                
                # Remove var from assignment
                del assignment[var]
                
        return "failure"

    def solve(self):
        # Reset counter
        self.nodes_generated = 0 # Root is call 1
        
        # Pre-check: Are any rows/cols already violated by input? 
        # (Assuming valid input as per project spec, we start search)
        
        # We account for the initial call as the first node/state check.
        # Actually logic varies, but usually entering the recursive function is a node.
        # self.nodes_generated starts at 0, first call becomes 1.
        
        result = self.backtrack({})
        return result

def write_output(filename, solution, nodes, board_template):
    with open(filename, 'w') as f:
        if solution == "failure":
            f.write("No solution found.\n")
        else:
            # d (depth): number of assigned variables. For a full solution, this is total variables.
            # Project spec Fig 4 says "d is the level of the goal node... root node is at level 0".
            # If we assign N variables, depth is N.
            d = len(solution)
            
            f.write(f"{d}\n")
            f.write(f"{nodes}\n")
            
            # Construct final grid
            # "No need to copy the initial cell values... n equals 0 or 1"
            # This implies the output grid is FULL 9x9 but numbers are replaced by solution?
            # Or just the variables? 
            # Fig 4 shows 1s and 0s. 
            # "Rows 3 to 11 contain your solution with n equals to 0 (no mine) or 1 (has mine)."
            # It also says "No need to copy the initial cell values...".
            # This usually means where there was a number, we put the solution for that cell?
            # BUT: Numbered cells are NOT variables. They cannot have mines.
            # So they are effectively 0 (no mine) in the mine map.
            # Mines can only be placed on empty cells.
            
            rows = 9
            cols = 9
            for r in range(rows):
                line = []
                for c in range(cols):
                    if (r,c) in solution:
                        line.append(str(solution[(r,c)]))
                    else:
                        # It was a pre-filled number. No mine there.
                        line.append("0") 
                f.write(" ".join(line) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sudoku_mine.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    solver = SudokuMineCSP(input_file)
    solution = solver.solve()
    
    write_output(output_file, solution, solver.nodes_generated, solver.board)