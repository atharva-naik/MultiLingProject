# do semantics preserving code transformations for Python source code with AST module.
import ast
import builtins
from typing import *
from ast import fix_missing_locations

TEST_CODE = """def do_something(z): print(z)
x = 3
y = False
z = x+y
if x == 3 and y == True:
    do_something(x*y)"""

class FunctionRenameTransform(ast.NodeTransformer):
    def reset(self):
        self.func_count = 0

        self.func_mapping = {}
        self.excluded_functions = set(dir(builtins))
        
    def __init__(self):
        self.func_count = 0

        self.func_mapping = {}
        self.excluded_functions = set(dir(builtins))

    def generic_func_name(self):
        self.func_count += 1

        return f'FUNC_{self.func_count}'

    def visit_Name(self, node):
        node.id = self.func_mapping.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node):
        # Exclude standard Python functions and imported library functions
        if node.name not in self.excluded_functions:
            # Rename the user-defined function
            generic_name = self.generic_func_name()
            self.func_mapping[node.name] = generic_name
            node.name = generic_name
        # Visit the body of the function
        self.generic_visit(node)
        return node

class VariableRenameTransform(ast.NodeTransformer):
    """Stateful transformation that replaces all encountered variable names with
    generic names of the form VAR_i"""
    def __init__(self):
        self.variable_count = 0
        self.variable_mapping = {}

    def reset(self):
        self.variable_count = 0
        self.variable_mapping = {}

    def generic_variable_name(self):
        self.variable_count += 1
        return f'VAR_{self.variable_count}'

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            # If it's a variable being assigned, replace its name
            if node.id not in self.variable_mapping:
                self.variable_mapping[node.id] = self.generic_variable_name()
            node.id = self.variable_mapping[node.id]
        elif isinstance(node.ctx, ast.Load):
            # If it's a variable being loaded, replace its name if available
            node.id = self.variable_mapping.get(node.id, node.id)
        return node

class BoolConditionModifier(ast.NodeTransformer):
    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        # add not on top to negate the effect.
        for i in range(len(node.values)):
            node.values[i] = ast.UnaryOp(
                op=ast.Not(),
                operand=node.values[i],
            )
        if isinstance(node.op, ast.Or):
            node.op = ast.And()
        elif isinstance(node.op, ast.And):
            node.op = ast.Or()
        node = ast.UnaryOp(
            op=ast.Not(),
            operand=node,
        )
        return node# super().generic_visit(node)

class BoolConstantModifier(ast.NodeTransformer):
    def visit_Compare(self, node: ast.Compare) -> Any:
        if (len(node.ops) == 1 and isinstance(node.ops[0], (ast.NotEq, ast.Eq)) and 
            len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value in [True, False]):
            
            if isinstance(node.ops[0], ast.NotEq):
                node.ops[0] = ast.Eq()
            elif isinstance(node.ops[0], ast.Eq):
                node.ops[0] = ast.NotEq()
            if node.comparators[0].value == True:
                node.comparators[0].value = False
            elif node.comparators[0].value == False:
                node.comparators[0].value = True
            return node
        else:
            return super().generic_visit(node)

class CompConditionModifier(ast.NodeTransformer):
    def visit_Compare(self, node: ast.Compare) -> Any:
        # for simple conditions with just one operator:
        if len(node.ops) == 1:
            # flip the comparison operator.
            # print(ast.unparse(node), node.ops)
            if isinstance(node.ops[0], ast.Lt):
                node.ops[0] = ast.GtE()
            elif isinstance(node.ops[0], ast.GtE):
                node.ops[0] = ast.Lt()
            elif isinstance(node.ops[0], ast.Gt):
                node.ops[0] = ast.LtE()
            elif isinstance(node.ops[0], ast.LtE):
                node.ops[0] = ast.Gt()
            elif isinstance(node.ops[0], ast.Eq):
                node.ops[0] = ast.NotEq()
            elif isinstance(node.ops[0], ast.NotEq):
                node.ops[0] = ast.Eq()
            elif isinstance(node.ops[0], ast.In):
                node.ops[0] = ast.NotIn()
            elif isinstance(node.ops[0], ast.NotIn):
                node.ops[0] = ast.In()
            elif isinstance(node.ops[0], ast.Is):
                node.ops[0] = ast.IsNot()
            elif isinstance(node.ops[0], ast.IsNot):
                node.ops[0] = ast.Is()
            # add not on top to negate the effect.
            node = ast.UnaryOp(
                op=ast.Not(),
                operand=node,
            )
            return node
        else: 
            return super().generic_visit(node)

    # def visit_BoolOp(self, node: ast.BoolOp) -> Any:
    #     print(node.values)
    #     return super().generic_visit(node)

class VarDecPermuteTransform(ast.NodeTransformer):
    def __init__(self):
        self.variables_to_move = set()
        self.permuted_statements = []
    
    def reset(self):
        self.variables_to_move = set()
        self.permuted_statements = []

    def check_assign_deps(self, node):
        # Check if the assignment depends on other variables
        dependencies = set()
        for value_node in ast.walk(node.value):
            if isinstance(value_node, ast.Name):
                dependencies.add(value_node.id)
        # If no dependencies, return true
        if not dependencies: return True
        return False

    def transform(self, tree):
        permuted_assigns = []
        other_stmts = []
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                # print(self.check_assign_deps(stmt))
                if self.check_assign_deps(stmt): 
                    permuted_assigns.append(stmt)
                else: other_stmts.append(stmt)
            else: other_stmts.append(stmt)
        tree.body = permuted_assigns+other_stmts

        return tree

class CodeTransformAugmenter:
    def __init__(self):
        self.transformers = {
            "compare_conditions": CompConditionModifier(),                 
            "bool_constants": BoolConstantModifier(),                 
            "bool_conditions": BoolConditionModifier(),
            "variable_rename": VariableRenameTransform(),
            "function_rename": FunctionRenameTransform(),
            "permute_statemtns": VarDecPermuteTransform()
        }

    def apply(self, code: str):
        new_codes = []
        for rule, transformer in self.transformers.items():
            unparsed_code = ast.unparse(ast.parse(code))
            if hasattr(transformer, 'reset'):
                transformer.reset() # reset the transformers that requrie resetting.
            if hasattr(transformer, 'transform'):
                new_code = ast.unparse(fix_missing_locations(transformer.transform(ast.parse(code))))
                print(new_code)
            else: new_code = ast.unparse(fix_missing_locations(transformer.visit(ast.parse(code))))
            if new_code != unparsed_code:
                new_codes.append((rule, new_code))
        # new_tree = fix_missing_locations(BoolConditionModifier().visit(copy.deepcopy(tree)))
        # new_codes.append(ast.unparse(new_tree))
        # new_tree = fix_missing_locations(BoolConstantModifier().visit(copy.deepcopy(tree)))
        # new_codes.append(ast.unparse(new_tree))
        return new_codes

    def __call__(self, code: str) -> Any:
        return self.apply(code)

# main
if __name__ == "__main__":
    aug = CodeTransformAugmenter()
    aug.apply(TEST_CODE)