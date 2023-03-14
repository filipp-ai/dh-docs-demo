import ast
"""
prepare for docs building (merge inherited members into classes)
"""
members_list = []


class MemberCollector(ast.NodeVisitor):
    current_class_node = None

    def visit_ClassDef(self, node):
        for member in [n for n in  node.body if isinstance(n, ast.FunctionDef)]:
            members_list.append({"class_name": node.name, "member": member})
        return node


class MemberInjector(ast.NodeVisitor):
    def visit_ClassDef(self, node):
        for base_class_name in [b.id for b in node.bases]:
            node.body += [m['member'] for m in members_list
                          if m['class_name'] == base_class_name
                          and m['member'].name not in [b.name for b in  node.body if isinstance(b, ast.FunctionDef)]
                          ]


def collect_members(module_path):
    with open(module_path) as f:
        source = f.read()
    tree = ast.parse(source)
    MemberCollector().visit(tree)

def inject_members(module_path):
    with open(module_path) as f:
        source = f.read()
    tree = ast.parse(source)
    MemberInjector().visit(tree)
    with open(module_path, "w") as f:
        lines = ast.unparse(tree).split('\n')
        for line in lines:
            f.write(line + '\n')


collect_members("../dataheroes/services/coreset_tree/_base.py")
collect_members("../dataheroes/services/coreset_tree/_mixin.py")

for result_module in  [
                        "../dataheroes/services/coreset_tree/kmeans.py",
                        "../dataheroes/services/coreset_tree/lg.py",
                        "../dataheroes/services/coreset_tree/lr.py",
                        "../dataheroes/services/coreset_tree/pca.py",
                        "../dataheroes/services/coreset_tree/svd.py",
                       ]:
    inject_members(result_module)
