# do semantics preserving code transformations for Python source code with AST module.
import ast
from typing import *
from ast import fix_missing_locations

TEST_CODE = """if x == 3 and y == True:
    print(x*y)"""

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

class CodeTransformAugmenter:
    def __init__(self):
        self.transformers = {
            "compare_conditions": CompConditionModifier(),                 
            "bool_constants": BoolConstantModifier(),                 
            "bool_conditions": BoolConditionModifier(),
        }

    def apply(self, code: str):
        new_codes = []
        for rule, transformer in self.transformers.items():
            new_code = ast.unparse(fix_missing_locations(transformer.visit(ast.parse(code))))
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