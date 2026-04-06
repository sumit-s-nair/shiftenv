import ast
import os
import json
import networkx as nx

class MigrationAnalyzer(ast.NodeVisitor):
    def __init__(self, filepath, library_name):
        self.filepath = filepath
        self.library_name = library_name
        self.current_function = None
        
        self.calls = []                
        self.defined_functions = set() 
        self.import_aliases = {}       
        self.library_calls = {}       

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.import_aliases[asname] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                asname = alias.asname or alias.name
              
                self.import_aliases[asname] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        prev = self.current_function
     
        self.current_function = f"{self.filepath}:{node.name}"

        self.defined_functions.add(self.current_function)
        if self.current_function not in self.library_calls:
            self.library_calls[self.current_function] = set()

        self.generic_visit(node)
        self.current_function = prev


    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self.current_function:
            func_name = self._get_call_name(node)
            if func_name:
                self.calls.append((self.current_function, func_name))

               
                if self._is_library_call(func_name):
                    self.library_calls[self.current_function].add(func_name)

        self.generic_visit(node)

    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            curr = node.func
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            return ".".join(reversed(parts))
        return None

    def _is_library_call(self, func_name):
        base = func_name.split(".")[0]
        
        resolved = self.import_aliases.get(base, base)
        return resolved.startswith(self.library_name)



def analyze_project(folder_path, library_name):
    G = nx.DiGraph()
    all_defined_functions = set()
    raw_calls = []
    all_library_calls = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
              
                filepath = os.path.relpath(os.path.join(root, file), folder_path)

                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=file)
                    except SyntaxError:
                        continue

                analyzer = MigrationAnalyzer(filepath, library_name)
                analyzer.visit(tree)

                all_defined_functions.update(analyzer.defined_functions)
                raw_calls.extend(analyzer.calls)
                
              
                for func, calls in analyzer.library_calls.items():
                    if calls:
                        all_library_calls.setdefault(func, set()).update(calls)

   
    G.add_nodes_from(all_defined_functions)

   
    for caller, callee in raw_calls:

        matches = [f for f in all_defined_functions if f.endswith(f":{callee}")]
        for match in matches:
            G.add_edge(caller, match)

    return G, all_library_calls



def export_graph(G, output_file="dependency_graph.graphml"):
    """
    Saves the graph to a GraphML file which can be opened in Gephi, 
    Cytoscape, or any standard graph visualization tool.
    """
    nx.write_graphml(G, output_file)
    print(f"Graph successfully written to {output_file}")


def export_migration_json(G, library_calls_map, output_file="migration_plan.json"):
    """
    Exports a sorted JSON file. Sorts by outgoing edges (out_degree) so 
    leaf nodes (functions that call no other internal functions) appear first.
    """
    result = []


    target_nodes = list(library_calls_map.keys())


    target_nodes.sort(key=lambda node: G.out_degree(node) if node in G else 0)

    for node in target_nodes:
    
        filepath, func_name = node.rsplit(":", 1) 
        
        
        internal_deps = list(G.successors(node)) if node in G else []
        
        entry = {
            "file_name": filepath,
            "function_name": func_name,
            "internal_dependency_count": len(internal_deps),
            "internal_dependencies": internal_deps,  
            "library_functions_used": list(library_calls_map[node])
        }
        result.append(entry)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Migration JSON successfully written to {output_file}")



def grapher(TARGET_FOLDER, TARGET_LIBRARY):


    print(f"Analyzing codebase in '{TARGET_FOLDER}' for library '{TARGET_LIBRARY}'...")
    
    G, library_calls_map = analyze_project(TARGET_FOLDER, TARGET_LIBRARY)

  
    export_graph(G, "dependency_graph.graphml")


    export_migration_json(G, library_calls_map, "migration_plan.json")


TARGET_FOLDER = "./test_folder" 
TARGET_LIBRARY = "requests"

grapher(TARGET_FOLDER,TARGET_LIBRARY)