# do semantics preserving code transformations for Python source code with AST module.
import ast
from typing import *
from ast import fix_missing_locations

TEST_CODE = """if x == 3 and y == True:
    print(x*y)"""

class CompConditionModifier(ast.NodeTransformer):
    # def visit_If(self, node: ast.If) -> Any:
    #     print(node.test, ast.unparse(node.test))
    #     return super().generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> Any:
        # for simple conditions with just one operator:
        if len(node.ops) == 1:
            # flip the comparison operator.
            print(ast.unparse(node), node.ops)
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
            print(ast.unparse(node), node.ops)
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
        pass

    def apply(self, code: str):
        tree = ast.parse(code)
        new_tree = fix_missing_locations(CompConditionModifier().visit(tree))
        new_codes = []
        new_codes.append(ast.unparse(new_tree))

        return new_codes

    def __call__(self, code: str) -> Any:
        return self.apply(code)

# main
if __name__ == "__main__":
    aug = CodeTransformAugmenter()
    aug.apply(TEST_CODE)