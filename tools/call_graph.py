import ast
import os
import json
from typing import Any

def get_call_graph(repo_dir: str, target_lib: str) -> dict[str, Any]:
    """
    Parses Python files in repo_dir using AST.
    Returns a mapping of {filepath: [functions/classes that use target_lib]}.
    This helps the agent locate exactly which functions need migration.
    """
    results = {}
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, repo_dir)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    tree = ast.parse(source, filename=rel_path)
                except Exception:
                    continue

                usages = _find_usages(tree, target_lib)
                if usages:
                    results[rel_path] = usages

    if not results:
        return {"status": "success", "usages": {}, "message": f"No usages of '{target_lib}' found."}
    
    return {"status": "success", "usages": results}


def _find_usages(tree: ast.Module, target_lib: str) -> list[str]:
    # Track the aliases used for the target_lib
    aliases = {target_lib}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == target_lib:
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] == target_lib:
                for alias in node.names:
                    aliases.add(alias.asname or alias.name)
    
    usages = set()
    
    class UsageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_context = []
        
        def visit_FunctionDef(self, node):
            self.current_context.append(f"def {node.name}")
            self.generic_visit(node)
            self.current_context.pop()
            
        def visit_AsyncFunctionDef(self, node):
            self.current_context.append(f"async def {node.name}")
            self.generic_visit(node)
            self.current_context.pop()

        def visit_ClassDef(self, node):
            self.current_context.append(f"class {node.name}")
            self.generic_visit(node)
            self.current_context.pop()

        def visit_Name(self, node):
            if node.id in aliases:
                if self.current_context:
                    usages.add(self.current_context[-1])
                else:
                    usages.add("<module level>")
            self.generic_visit(node)

    visitor = UsageVisitor()
    visitor.visit(tree)
    
    # Return as sorted list for deterministic output
    return sorted(list(usages))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        res = get_call_graph(sys.argv[1], sys.argv[2])
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python call_graph.py <repo_dir> <target_lib>")
