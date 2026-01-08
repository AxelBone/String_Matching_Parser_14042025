import networkx as nx
import obonet



#### HPO TERMS for EHR #######

def reading_hpo_terms(path: str) -> dict:
    """
    Reads HPO terms from a file and creates a dictionary mapping term IDs to term names.

    Args:
        path (str): The path to the file containing HPO terms. The file should have each line formatted
                    as "Term Name <space or tab> Term ID".

    Returns:
        dict: A dictionary where the keys are HPO term IDs and the values are HPO term names.

    Notes:
        - This function skips empty lines and lines that do not match the expected format.
        - Only the first occurrence of a term ID will be stored in the dictionary.
    """
    d = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()  # Remove any trailing newline and spaces
            if not line:  
                continue  # Skip empty lines
            
            # Try to split the line into name and ID (handle both space and tab separated lines)
            try:
                val, key = line.rsplit(maxsplit=1)  # Split only on the last space/tab
            except ValueError:
                print(f"Skipping invalid line: {line}")  # Handle line with no split
                continue
            
            # Keep only the first occurrence of each ID
            if key not in d:
                d[key] = val 

    return d


def idToHpo(id: str, ontology: dict) -> str:
    """
    Retrieves the HPO term name corresponding to the given term ID from the ontology.

    Args:
        id (str): The HPO term ID to look up.
        ontology (dict): A dictionary where the keys are HPO term IDs and the values are HPO term names.

    Returns:
        str: The HPO term name associated with the given term ID. If the ID is not found, 
             it may raise a KeyError depending on the dictionary behavior.
    """
    return ontology[id]


def HpoToId(hpo_name: str, ontology: dict) -> str:
    """
    Retrieves the HPO term ID corresponding to the given HPO term name from the ontology.

    Args:
        hpo_name (str): The HPO term name to look up.
        ontology (dict): A dictionary where the keys are HPO term IDs and the values are HPO term names.

    Returns:
        str: The HPO term ID associated with the given term name. If the name is not found, 
             returns a string "HPO term not found".
    """
    for key, value in ontology.items():  # Iterate through the dictionary
        if value == hpo_name:  # Check if the value (name) matches the hpo_name
            return key  # Return the corresponding ID (key)
    return "HPO term not found"  # Return a message if the term name is not found


##### HPO GRAPH 

def load_ontology(source: str) -> nx.MultiDiGraph:
    """
    Load the Gene Ontology from a URL or a file and return a networkx MultiDiGraph.
    
    Parameters:
    - source (str): URL or file path to the Gene Ontology OBO file.
    
    Returns:
    - nx.MultiDiGraph: The loaded ontology as a network graph.
    """
    return obonet.read_obo(source)



def get_graph_statistics(graph: nx.MultiDiGraph):
    """
    Get the number of nodes, edges, and check if the graph is a DAG.
    
    Parameters:
    - graph (nx.MultiDiGraph): The ontology graph.
    
    Returns:
    - dict: A dictionary containing statistics (nodes, edges, is_dag).
    """
    stats = {
        'nodes': len(graph),
        'edges': graph.number_of_edges(),
        'is_dag': nx.is_directed_acyclic_graph(graph)
    }
    return stats



def get_node_properties(graph: nx.MultiDiGraph, node_id: str):
    """
    Retrieve the properties of a node in the ontology graph by its ID.
    
    Parameters:
    - graph (nx.MultiDiGraph): The ontology graph.
    - node_id (str): The ID of the node to retrieve properties for.
    
    Returns:
    - dict: The properties of the node.
    """
    return graph.nodes.get(node_id, {})

def create_id_name_mappings(graph: nx.MultiDiGraph):
    """
    Create mappings between node IDs and names (and vice versa) in the ontology graph.
    
    Parameters:
    - graph (nx.MultiDiGraph): The ontology graph.
    
    Returns:
    - tuple: A tuple containing two dictionaries (id_to_name, name_to_id).
    """
    id_to_name = {id_: data.get("name") for id_, data in graph.nodes(data=True)}
    name_to_id = {data["name"]: id_ for id_, data in graph.nodes(data=True) if "name" in data}
    return id_to_name, name_to_id


def find_parent_child_relationships(graph: nx.MultiDiGraph, node_id: str, id_to_name: dict, direction: str):
    """
    Find the parent-child relationships of a node.
    Parameters:
    - graph (nx.MultiDiGraph): The ontology graph.
    - node_id (str): The ID of the node to find relationships for.
    - id_to_name (dict): A dictionary mapping node IDs to names.
    - direction (str): 'superterms' to find relationships to parents, 'subterms' to find relationships to children.
    Returns:
    - list: A list of strings describing the relationships.
    """
    relationships = []
    
    if direction == "superterms":
        # Find relationships to parent terms (node_id -> parent)
        for child, parent, key in graph.out_edges(node_id, keys=True):
            relationships.append(f"• {id_to_name.get(child, child)} ⟶ {key} ⟶ {id_to_name.get(parent, parent)}")
    
    elif direction == "subterms":
        # Find relationships to child terms (child -> node_id)
        for child, parent, key in graph.in_edges(node_id, keys=True):
            relationships.append(f"• {id_to_name.get(child, child)} ⟶ {key} ⟶ {id_to_name.get(parent, parent)}")
    
    else:
        raise ValueError("Direction must be 'superterms' or 'subterms'.")
    
    return relationships


def find_superterms_or_subterms(graph: nx.MultiDiGraph, node_id: str, direction: str = 'superterms', id_to_name=dict) -> list:
    """
    Find all superterms or subterms for a given node.
    
    Parameters:
    - graph (nx.MultiDiGraph): The ontology graph.
    - node_id (str): The ID of the node to find terms for.
    - direction (str): 'superterms' to find ancestors, 'subterms' to find descendants.
    
    Returns:
    - list: A sorted list of superterms or subterms.
    """

    if direction == 'subterms':
        return sorted(id_to_name[term] for term in nx.ancestors(graph, node_id))

    elif direction == 'superterms':
        return sorted(id_to_name[term] for term in nx.descendants(graph, node_id))
    else:
        raise ValueError("Direction must be 'superterms' or 'subterms'.")


def find_paths_to_root(graph: nx.MultiDiGraph, start_term: str, end_term: str, id_to_name: dict):
    """
    Find all paths from a start term to the end term (typically the root).
    
    Parameters:
    - graph (nx.MultiDiGraph): The ontology graph.
    - start_term (str): The ID of the starting term.
    - end_term (str): The ID of the ending term (typically the root).
    - id_to_name (dict): A dictionary mapping node IDs to names.
    
    Returns:
    - list: A list of strings representing the paths.
    """
    paths = nx.all_simple_paths(graph, source=start_term, target=end_term)
    return [" ⟶ ".join(id_to_name.get(node, node) for node in path) for path in paths]


def create_node_to_root_distance_dict(subgraph: nx.MultiDiGraph, root_id: str = "HP:0000118") -> dict:
    """
    Create a dictionary mapping each node to its shortest distance from the root.
    
    Parameters:
    - subgraph (nx.MultiDiGraph): The subgraph containing the root and its descendants
    - root_id (str): The ID of the root node (default: "HP:0000118" for phenotypic abnormality)
    
    Returns:
    - dict: A dictionary mapping each node ID to its shortest distance from the root
    """
    if root_id not in subgraph:
        raise ValueError(f"Root node {root_id} not found in the graph")
    

    # Initialize distances dictionary with infinity for all nodes
    distances = {node: float('inf') for node in subgraph.nodes()}

    distances[root_id] = 0
    queue = [root_id]
    visited = set([root_id])

    while queue:
        current = queue.pop(0)
        current_distance = distances[current]

        # Find all children of the current node
        for child, parent, key in subgraph.in_edges(current, keys=True):
            if key == 'is_a' and child not in visited:
                # Distance to child is one more than distance to current
                distances[child] = current_distance + 1
                visited.add(child)
                queue.append(child)
                
    return distances


def find_deepest_node(graph: nx.MultiDiGraph, root_id: str = "HP:0000118"):
    """
    Trouve le nœud le plus profond à partir d'un nœud racine donné.
    
    Parameters:
    - graph (nx.MultiDiGraph): Le graphe d'ontologie
    - root_id (str): L'identifiant du nœud racine (par défaut: "HP:0000118" pour phenotypic abnormality)
    
    Returns:
    - tuple: (nœud_le_plus_profond, profondeur_maximale, chemin_le_plus_long)
    """
    # Créer les mappages pour les noms
    id_to_name, name_to_id = create_id_name_mappings(graph)
    
    # Initialiser les distances et les chemins
    distances = {root_id: 0}
    paths = {root_id: [root_id]}
    
    # File pour le BFS
    queue = [root_id]
    
    # Nœud le plus profond et sa profondeur
    deepest_node = root_id
    max_depth = 0
    
    while queue:
        current = queue.pop(0)
        current_depth = distances[current]
        
        # Si ce nœud est plus profond que le précédent max, mettre à jour
        if current_depth > max_depth:
            max_depth = current_depth
            deepest_node = current
        
        # Trouver tous les enfants du nœud courant
        for child, parent, key in graph.in_edges(current, keys=True):
            if key == 'is_a' and child not in distances:
                # Distance au child est un de plus que la distance au parent
                distances[child] = current_depth + 1
                # Mémoriser le chemin
                paths[child] = paths[current] + [child]
                queue.append(child)
    
    # Convertir les IDs du chemin en noms pour plus de lisibilité
    path_names = [id_to_name.get(node_id, node_id) for node_id in paths[deepest_node]]
    
    return deepest_node, max_depth, path_names

def analyze_branch_depths(graph: nx.MultiDiGraph, root_id: str = "HP:0000118"):
    """
    Analyse les profondeurs de toutes les branches immédiates d'un nœud racine.
    
    Parameters:
    - graph (nx.MultiDiGraph): Le graphe d'ontologie
    - root_id (str): L'identifiant du nœud racine
    
    Returns:
    - dict: Dictionnaire mappant chaque enfant direct à sa profondeur maximale
    """
    id_to_name, name_to_id = create_id_name_mappings(graph)
    branch_depths = {}
    
    # Trouver tous les enfants directs
    direct_children = []
    for child, parent, key in graph.in_edges(root_id, keys=True):
        if key == 'is_a':
            direct_children.append(child)
    
    # Pour chaque enfant direct, trouver sa profondeur maximale
    for child in direct_children:
        deepest_node, depth, _ = find_deepest_node(graph, child)
        child_name = id_to_name.get(child, child)
        branch_depths[child_name] = {
            'max_depth': depth + 1,  # +1 car on compte à partir du nœud racine
            'deepest_node': id_to_name.get(deepest_node, deepest_node)
        }
    
    return branch_depths


def subset_ontology_by_term(ontology: nx.MultiDiGraph, term: str) -> nx.MultiDiGraph:
    """
    Extract a subgraph of the ontology starting from a specific term.
    
    Parameters:
    - ontology (nx.MultiDiGraph): The ontology graph.
    - term (str): The HPO term identifier (e.g., "HP:0000001").
    
    Returns:
    - nx.MultiDiGraph: The subgraph of the ontology containing the specified term and its descendants.
    """
    # Vérifie si le terme existe dans le graphe
    if term not in ontology:
        raise ValueError(f"Le terme {term} n'existe pas dans l'ontologie.")
    
    # Trouver tous les ancêtres et descendants du terme
    descendants = nx.ancestors(ontology, term)
    
    descendants_set = set(descendants)
    # Ajouter le terme lui-même à l'ensemble des descendants
    descendants_set.add(term)
    
    # Extraire le sous-graphe des descendants
    subgraph = ontology.subgraph(descendants_set).copy()
    
    return subgraph