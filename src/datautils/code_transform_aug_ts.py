# code for augmenting the programming language data by applying semantics preserving transforms (wit tree-sitter).
import tree_sitter
from tree_sitter import Language

# Load the Java grammar for tree-sitter
java_lang = Language('src/datautils/my-languages.so', 'java')

# Sample Java code
java_code = """
class Example {
    public static void main(String[] args) {
        boolean var = true;

        if (var == true) {
            System.out.println("Condition 1");
        } else if (var == false) {
            System.out.println("Condition 2");
        }

        if (var == false) {
            System.out.println("Condition 3");
        }
    }
}
"""

# Parse the Java code
parser = tree_sitter.Parser()
parser.set_language(java_lang)
tree = parser.parse(bytes(java_code, 'utf-8'))

# Helper function to replace conditions
def replace_conditions(node):
    if node.type == 'binary_expression' and node.child_by_field_name('operator').type == 'EQEQ':
        left_child = node.child_by_field_name('left')
        right_child = node.child_by_field_name('right')

        if left_child.type == 'identifier' and right_child.type == 'boolean':
            if right_child.text == 'true':
                left_child.set_field('text', '!(%s == false)' % left_child.text)
            elif right_child.text == 'false':
                left_child.set_field('text', '%s == false' % left_child.text)

# def iterate_over_tree(tree):
#     cursor = tree.walk()
#     while cursor.goto_first_child() or cursor.goto_next_sibling():
#         print(cursor.node)

def walk_tree(root_node):
    for child in root_node.children:
        print(child)
        walk_tree(child)

# Helper function to unparse the tree
def unparse(node):
    result = ""

    if node.is_named:
        result += node.type
    for child in node.children:
        result += unparse(child)
    if node.is_named:
        result += node.end_byte - node.start_byte

    return result

# main
if __name__ == "__main__":
    # Traverse the tree and apply transformations
    for node in tree.root_node.walk():
        replace_conditions(node)

    # Print the modified code
    print(tree.root_node.to_sexp())
