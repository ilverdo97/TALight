import ast
import re
import networkx as nx
from sys import stderr
from typing import Dict
from RO_verify_submission_gen_prob_lib import verify_submission_gen

# specifiche dell'istanza del problema
instance_objects_spec = {
    ('n', int),                 # numero di nodi
    ('m', int),                 # numero di archi
    ('edges', str),             # lista degli archi
    ('forbidden_edges', str),   # lista indici degli archi da escludere
    ('forced_edges', str),      # lista indici degli archi obbligati
    ('query_edge', str),        # indice arco da esaminare
}

# specifiche delle soluzioni
answer_objects_spec = {
    'opt_sol': str,         # soluzione ottimale (lista indici archi)
    'opt_val': int,         # Valore ottimale (peso di questa opt_sol)
    'num_opt_sols': int,    # numero di soluzioni ottimali totali
    'list_opt_sols': str,   # lista soluzioni ottimali
    'edge_profile': str     # il query_edge appartiene a nessuna ("in_no"), a tutte ("in_all") o ad alcune soluzioni
                            # ottime ("in_some_but_not_in_all")
}

# lista delle soluzioni implementate
answer_objects_implemented = [
    'opt_sol',
    'opt_val',
    'num_opt_sols',
    'list_opt_sols',
    'edge_profile'
]


def check_isolated_nodes(n: int, edges: list) -> bool:
    """
    Verifica se esistono nodi isolati (nodi non connessi da archi).
    """
    nodes_found = set()
    for u, v, w in edges:
        nodes_found.add(u)
        nodes_found.add(v)
    return len(nodes_found) != n


def check_isolated_nodes_by_forbidden(n: int, edges: list, forbidden_edges: list) -> bool:
    """
    Verifica se esistono nodi isolati, anche a causa di archi esclusi.
    """
    nodes_found = set()
    for i, (u, v, w) in enumerate(edges):
        if i not in forbidden_edges:
            nodes_found.add(u)
            nodes_found.add(v)
    return len(nodes_found) != n


def check_tree(tree: list, n: int, edges: list) -> bool:
    """
    Verifica se la lista di archi corrisponde a un albero.
    """
    graph = nx.MultiGraph(n)
    for i in tree:
        u, v = list(edges[i][0])    # edges[i] == ({u, v}, w)
        graph.add_edge(u, v)
    return nx.is_tree(graph)


def check_spanning(tree: list, n: int, edges: list) -> bool:
    """
    Verifica se la lista di archi copre tutto il grafo.
    """
    nodes_found = set()
    for i in tree:
        nodes_found = nodes_found.union(edges[i][0])    # edges[i] == ({u, v}, w)
    return len(nodes_found) == n


def check_weight_in_range(input_weight: float, edges: list, nodes: int) -> int:
    """
    Verifica il peso totale dell'albero rientra nel range possibile.
    """
    weights = sorted([w for _, w in edges])     # ordina gli archi per peso crescente
    min_weight = sum(weights[:nodes - 1])       # somma i pesi dei primi |V| - 1 archi
    max_weight = sum(weights[-(nodes-1):])      # somma i pesi degli ultimi |V| - 1 archi
    return -1 if input_weight < min_weight else 1 if input_weight > max_weight else 0


def check_instance_consistency(instance: dict):
    print(f"instance={instance}", file=stderr)
    n = instance['n']
    m = instance['m']
    edges = instance['edges']
    forbidden_edges = instance['forbidden_edges']
    forced_edges = instance['forced_edges']
    query_edge = instance['query_edge']

    edges = ast.literal_eval(re.sub(r"[{}]", "", edges))
    forbidden_edges = ast.literal_eval(forbidden_edges)
    forced_edges = ast.literal_eval(forced_edges)

    if n <= 0:
        print(f"Errore: il numero di nodi è minore o uguale a 0")
        exit(0)
    if m <= 0:
        print(f"Errore: il numero di archi è minore o uguale a   0")
        exit(0)
    if m != len(edges):
        print(f"Errore: il numero di archi non corrisponde con il numero di archi della lista data")
        exit(0)
    if not all(w >= 0 for _, _, w in edges):
        print(f"Errore: alcuni pesi sono minori di 0")
    if not all(0 <= u < n and 0 <= v < n for u, v, _ in edges):
        print(f"Errore: alcuni nodi dati negli archi non esistono")
        exit(0)
    if not all(u != v for u, v, _ in edges):
        print(f"Errore: il grafo non può contenere auto-loop")
        exit(0)
    if not all(0 <= e < m for e in forbidden_edges):
        print(f"Errore: alcuni archi dichiarati in forbidden_edges non esistono")
        exit(0)
    if not all(0 <= e < m for e in forced_edges):
        print(f"Errore: alcuni archi dichiarati in forced_edges non esistono")
        exit(0)
    if not 0 <= query_edge < m:
        print(f"Errore: il query_edge non esiste")
        exit(0)
    if len(set(forbidden_edges).intersection(set(forced_edges))) != 0:
        print(f"Errore: alcuni forced_edges sono anche forbidden_edges")
        exit(0)
    if query_edge in forbidden_edges:
        print(f"Errore: il query_edge è nei forbidden_edges, la soluzione può solo che essere in_no")
        exit(0)
    if query_edge in forced_edges:
        print(f"Errore: il query_edge è nei forbidden_edges, la soluzione può solo che essere in_all")
        exit(0)
    if check_isolated_nodes(n, edges):
        print(f"Errore: sono presenti dei nodi isolati, non connessi da archi")
        exit(0)
    if check_isolated_nodes_by_forbidden(n, edges, forbidden_edges):
        print(f"Errore: sono presenti dei nodi isolati, dovuti ad archi eliminati da forbidden_edges")
        exit(0)


class Graph:
    """
    Classe per grafi pesati indiretti, con possibili archi paralleli.
    """

    def __init__(self, vertices: int):
        self.V = vertices   # numero di vertici
        self.edges = []     # lista degli archi, [(u, v, weight, label), ...]
        self.adjacency = [[[] for _ in range(vertices)] for _ in range(vertices)]   # matrice di adiacenza

    def add_edge(self, u: int, v: int, weight: float, label: int) -> None:
        """
        Aggiungi un arco al grafo.
        """
        self.edges.append((u, v, weight, label))
        self.adjacency[u][v].append({'weight': weight, 'label': label})
        self.adjacency[v][u].append({'weight': weight, 'label': label})

    def add_all_edges(self, edges: list) -> None:
        """
        Aggiungi una lista di archi pesati, nel formato [({u,v}, w), ...], al grafo attuale.
        """
        for label, edge in enumerate(edges):
            u, v = list(edge[0])
            weight = float(edge[1])
            self.add_edge(u, v, weight, label)

    def __search_root(self, parent: list, i: int) -> int:
        """
        Cerca il nodo radice del sotto albero a cui appartiene il nodo i
        """
        return i if parent[i] == i else self.__search_root(parent, parent[i])

    def __apply_union(self, parent: list, rank: list, u: int, v: int):
        """
        Unisci i sotto-alberi a cui appartengono u e v
        """
        u_root = self.__search_root(parent, u)
        v_root = self.__search_root(parent, v)
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        elif rank[u_root] > rank[v_root]:
            parent[v_root] = u_root
        else:
            parent[v_root] = u_root
            rank[u_root] += 1

    def kruskal_constrained(self, forced: list, excluded: list) -> (list, int):
        """
        Trova un MST per il grafo attuale, considerando archi forzati e archi esclusi. Nota: funziona solo se il grafo
        è connesso.
        """
        mst = []
        i, e, tot_weight = 0, 0, 0
        self.edges = sorted(self.edges, key=lambda item: item[2])   # ordina gli archi in ordine crescente di peso
        parent = list(range(self.V))    # list del nodo genitore, per ogni nodo
        rank = [0] * self.V             # dimensione del sotto-albero di appartenenza per ogni nodo

        # aggiungo gli archi forzati alla soluzione (obbligati)
        for u, v, weight, label in self.edges:
            if label in forced:
                e += 1
                mst.append(label)
                tot_weight += weight
                u_root = self.__search_root(parent, u)              # ricerca della radice di u
                v_root = self.__search_root(parent, v)              # ricerca della radice di v
                self.__apply_union(parent, rank, u_root, v_root)    # unisci i due sotto-alberi

        # itera finché non completi l'albero (n°archi = |V| - 1)
        while e < self.V - 1:
            u, v, weight, label = self.edges[i]
            i += 1
            u_root = self.__search_root(parent, u)
            v_root = self.__search_root(parent, v)
            # se: le radici sono diverse, l'arco non è in quelli esclusi e non è già stato aggiunto (forzato)
            if u_root != v_root and label not in excluded and label not in forced:
                e += 1
                mst.append(label)   # aggiungi arco a mst
                tot_weight += weight
                self.__apply_union(parent, rank, u_root, v_root)    # unisci i due sotto-alberi

        return mst, tot_weight  # ritorno lista indice archi es.([0, 2]), peso totale

    def __find_substitute(self, cut: int, tree: set, excluded: set) -> int | None:
        """
        Trova un sostituto ideale per l'arco cut
        """
        # ricerca arco tagliato nella lista degli archi del grafo
        cut_u, cut_v, cut_w, cut_l = list(filter(lambda x: x[3] == cut, self.edges))[0]
        subtree = {cut_u}       # sotto-abero di partenza (l'altro sotto-albero corrisponde a V\subtree)
        tmp_list = [cut_u]      # lista dei nodi dei quali bisogna esplorare gli archi

        # costruzione dei due sotto-alberi generati dal taglio
        while tmp_list:
            u = tmp_list.pop()
            for v in range(self.V):
                # se il nodo v non fa parte di questo sotto-albero ed esistono archi che collegano u e v
                if v not in subtree and (edges := self.adjacency[u][v]):
                    # se il primo arco che collega u e v non fa parte dell'albero e non è l'arco tagliato
                    if edges[0]['label'] in tree and edges[0]['label'] != cut_l:
                        tmp_list.append(v)  # aggiungi v ai nodi di cui esplorare gli archi
                        subtree.add(v)      # aggiungi v al sotto-albero

        # ricerca di un sostituito per l'arco tagliato, ovvero un arco che riconnette i due sotto-alberi,
        # con peso uguale a quello tagliato (non può essere minore)
        for u, v, weight, label in self.edges:
            # se l'arco, non è quello tagliato
            # non è nella lista degli archi esclusi
            # i nodi che collega appartengono a due sotto-alberi diversi
            # il peso è equivalente a quello dell'arco tagliato
            if label != cut_l and \
                    label not in excluded and \
                    ((u in subtree) ^ (v in subtree)) and \
                    weight == cut_w:
                return label

        # non esiste un sostituto per l'arco tagliato
        return None

    def __all_mst(self, tree: set, forced: set, excluded: set) -> list:
        """
        Trova tutti gli MST del grafo attuale, a partire da una soluzione ottima
        """
        search_set = tree.difference(forced)    # si esclude dalla ricerca gli archi forzati
        msts = []
        for edge in search_set:
            sub = self.__find_substitute(edge, tree, excluded)  # ricerca del sostituto per edge
            if sub is not None:
                new_tree = tree                 # nuovo mst trovato in cui...
                new_tree.remove(edge)           # sostituisco l'arco tagliato (edge)...
                new_tree.add(sub)               # ...con l'arco sostitutivo (sub)
                msts.append(list(new_tree))     # aggiungi il nuovo mst alla lista degli mst trovati
                # ricerca ricorsiva di nuovi mst a partire da quello nuovo trovato, inserendo tra gli esclusi edge
                msts += self.__all_mst(new_tree, forced, excluded.union({edge}))
        return msts

    def all_mst(self, forced: list, excluded: list) -> list:
        """
        Trova tutti gli MST del grafo attuale
        """
        # trova un mst di partenza
        first, _ = self.kruskal_constrained(forced, excluded)
        # trova tutti gli altri mst a partire dalla soluzione appena trovata
        return [first] + self.__all_mst(set(first), set(forced), set(excluded))


def solver(input_to_oracle: dict) -> dict:
    instance = input_to_oracle['input_data_assigned']
    n = instance['n']
    m = instance['m']
    edges = instance['edges']
    forbidden_edges = instance['forbidden_edges']
    forced_edges = instance['forced_edges']
    query_edge = instance['query_edge']

    edges = ast.literal_eval(edges)
    forbidden_edges = ast.literal_eval(forbidden_edges)
    forced_edges = ast.literal_eval(forced_edges)
    graph = Graph(n)
    graph.add_all_edges(edges)

    opt_sol, opt_val = graph.kruskal_constrained(forced_edges, forbidden_edges)
    list_opt_sols = graph.all_mst(forced_edges, forbidden_edges)
    num_opt_sols = len(list_opt_sols)
    count = [query_edge in sol for sol in list_opt_sols].count(True)
    if count == len(list_opt_sols):
        edge_profile = 'in_all'
    elif count > 0:
        edge_profile = 'in_some_but_not_in_all'
    else:
        edge_profile = 'in_no'

    print(f"input_to_oracle={input_to_oracle}", file=stderr)
    input_data = input_to_oracle["input_data_assigned"]
    print(f"Instance={input_data}", file=stderr)
    oracle_answers = {}
    for std_name, ad_hoc_name in input_to_oracle["request"].items():
        oracle_answers[ad_hoc_name] = locals()[std_name]
    print(oracle_answers)
    return oracle_answers


class verify_submission_problem_specific(verify_submission_gen): #verifica soluzioni sottoposte dall'utente
    def __init__(self, SEF, input_data_assigned: Dict, long_answer_dict: Dict, oracle_response: Dict = None):
        super().__init__(SEF, input_data_assigned, long_answer_dict, oracle_response)

    def verify_format(self, SEF): #formato risposta
        if not super().verify_format(SEF):
            return False

        if 'opt_sol' in self.goals:
            g = self.goals['opt_sol']
            if type(g.answ) != str:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere una stringa di archi. Una lista di archi è data dagli è "
                                        f"costituita da una lista di indici riferiti alla lista degli archi "
                                        f"nell'istanza del problema")
            try:
                answ = ast.literal_eval(g.answ)
                if type(answ) != list:
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di indici riferiti alla lista degli archi "
                                            f"nell'istanza del problema")
                if not all([isinstance(edge, int) for edge in answ]):
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di indici riferiti alla lista degli archi "
                                            f"nell'istanza del problema")
            except SyntaxError:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere una lista di archi. Impossibile effettuare il parsing dell'input: "
                                        f"errore nella sintassi.")
            SEF.format_OK(g, f"come `{g.alias}` hai immesso una stringa come richiesto",
                          f"ovviamente durante lo svolgimento dell'esame non posso dirti se la stringa inserita sia "
                          f"poi la risposta corretta, ma il formato è corretto")

        if 'opt_val' in self.goals:
            g = self.goals['opt_val']
            if type(g.answ) != int:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere un intero.")
            SEF.format_OK(g, f"come `{g.alias}` hai immesso un intero come richiesto",
                          f"ovviamente durante lo svolgimento dell'esame non posso dirti se l'intero immesso sia poi "
                          f"la risposta corretta, ma il formato è corretto")

        if 'num_opt_sols' in self.goals:
            g = self.goals['num_opt_sols']
            if type(g.answ) != int:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere un intero.")
            SEF.format_OK(g, f"come `{g.alias}` hai immesso un intero come richiesto",
                          f"ovviamente durante lo svolgimento dell'esame non posso dirti se l'intero immesso sia poi "
                          f"la risposta corretta, ma il formato è corretto")

        if 'list_opt_sols' in self.goals:
            g = self.goals['list_opt_sols']
            if type(g.answ) != str:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere una stringa di archi. Una lista di archi è data dagli è "
                                        f"costituita da una lista di indici riferiti alla lista degli archi "
                                        f"nell'istanza del problema")
            try:
                answ = ast.literal_eval(g.answ)
                if type(answ) != list:
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di indici riferiti alla lista degli archi "
                                            f"nell'istanza del problema")
                if not all([isinstance(tree, list) for tree in answ]):
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di indici riferiti alla lista degli archi "
                                            f"nell'istanza del problema")
                for tree in answ:
                    if not all([isinstance(edge, int) for edge in tree]):
                        return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                                f"immettere una lista di archi. Una lista di archi è costituita da una "
                                                f"lista di indici (interi) riferiti alla lista degli archi "
                                                f"nell'istanza del problema")
            except SyntaxError:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere una lista di archi. Impossibile effettuare il parsing dell'input: "
                                        f"errore nella sintassi.")
            SEF.format_OK(g, f"come `{g.alias}` hai immesso una stringa come richiesto",
                          f"ovviamente durante lo svolgimento dell'esame non posso dirti se la stringa inserita sia "
                          f"poi la risposta corretta, ma il formato è corretto")

        if 'edge_profile' in self.goals:
            g = self.goals['edge_profile']
            if type(g.answ) != str:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                        f"immettere una stringa.")
            if g.answ not in ['in_all', 'in_no', 'in_some_but_not_in_all']:
                return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', Risposte valide: in_all, in_no, "
                                        f"in_some_but_not_in_all ")
            SEF.format_OK(g, f"come '{g.alias}' hai immesso una stringa come richiesto",
                          f"ovviamente durante lo svolgimento dell'esame non posso dirti se la stringa inserita sia "
                          f"poi la risposta corretta, ma il formato è corretto")

        return True

    def verify_feasibility(self, SEF): #verificare che quello che l'utente ha inserito sia sensato rispetto all'istanza del problema
        if not super().verify_feasibility(SEF):
            return False

        if 'opt_sol' in self.goals:
            g = self.goals['opt_sol']
            answ = ast.literal_eval(g.answ) #risposta utente
            edges = ast.literal_eval(self.I.edges) #istanza problame in self.I
            forbidden_edges = ast.literal_eval(self.I.forbidden_edges)
            forced_edges = ast.literal_eval(self.I.forced_edges)
            if not all(0 <= e < self.I.m for e in answ):
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma al suo interno sono presenti archi che non esistono")
            if len(answ) != len(set(answ)):
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma al suo interno sono presenti degli archi ripetuti.")
            if not check_tree(answ, edges, self.I.n):
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma esso non rappresenta un albero")
            if not check_spanning(answ, self.I.n, edges):
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma esso non rappresenta uno spanning tree")
            if len(set(answ).intersection(set(forbidden_edges))) != 0:
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma al suo interno sono presenti dei forbidden_edges")
            if len(set(answ).intersection(set(forced_edges))) != len(forced_edges):
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma al suo interno non sono presenti tutti i forced_edges")
            SEF.feasibility_OK(g, f"Come '{g.alias}' hai immesso un sottoinsieme degli oggetti dell'istanza originale", f"Ora resta da stabilire l'ottimalità di '{g.alias}'")

        if 'opt_val' in self.goals:
            g = self.goals['opt_val']
            edges = ast.literal_eval(self.I.edges)
            if (res := check_weight_in_range(g.answ, edges, self.I.n)) != 0:
                return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma esso sfora la somma {'minima' if res < 0 else 'massima'} possibile dei pesi")
            SEF.feasibility_OK(g, f"Come '{g.alias}' hai immesso un sottoinsieme degli oggetti dell'istanza originale", f"Ora resta da stabilire l'ottimalità di '{g.alias}'")

        if 'list_opt_sols' in self.goals:
            g = self.goals['list_opt_sols']
            answ = ast.literal_eval(g.answ)
            edges = ast.literal_eval(self.I.edges)
            forbidden_edges = ast.literal_eval(self.I.forbidden_edges)
            forced_edges = ast.literal_eval(self.I.forced_edges)
            for tree in answ:
                if not all(0 <= e < self.I.m for e in tree):
                    return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma alcune di esse presentano degli archi che non esistono.")
                if len(tree) != len(set(tree)):
                    return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma alcune di esse presentano degli archi ripetuti.")
                if not check_tree(tree, edges, self.I.n):
                    return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma alcune di esse non rappresentano un albero")
                if not check_spanning(answ, self.I.n, edges):
                    return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma alcune di esse non rappresentano uno spanning tree")
                if len(set(tree).intersection(set(forbidden_edges))) != 0:
                    return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma alcune di esse contengono dei forbidden_edges")
                if len(set(tree).intersection(set(forced_edges))) != len(forced_edges):
                    return SEF.feasibility_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}', ma alcune di esse non contengono tutti i forced_edges")
            SEF.feasibility_OK(g, f"Come '{g.alias}' hai immesso un sottoinsieme degli oggetti dell'istanza originale", f"Ora resta da stabilire l'ottimalità di '{g.alias}'")

        if 'num_opt_sols' in self.goals:
            g = self.goals['num_opt_sols']
            if g.answ <= 0:
                return SEF.feasibility_NO(g, f"Come '{g.alias} hai immesso '{g.answ}', ma il numero di soluzioni ottime deve essere maggiore di 0")
            if g.answ > self.I.n ** (self.I.n - 2):
                return SEF.feasibility_NO(g, f"Come '{g.alias} hai immesso '{g.answ}', ma in qualsiasi grafo non possono esistere più di n^(n-2) spanning trees, dove n è il numero di archi")
            SEF.feasibility_OK(g, f"Come {g.alias} hai immesso un sottoinsieme degli oggetti dell'istanza originale", f"Ora resta da stabilire l'ottimalità di `{g.alias}`")

        return True

    def verify_consistency(self, SEF): #verificare che l'utente sia consistente con se stesso rispetto a quello che inserisce lui
        if not super().verify_consistency(SEF):
            return False

        if 'opt_sol' in self.goals and 'opt_val' in self.goals:
            opt_sol_g = self.goals['opt_sol']
            opt_val_g = self.goals['opt_val']
            opt_sol_answ = ast.literal_eval(opt_sol_g.answ)
            if sum([self.I.edges[i][1] for i in opt_sol_answ]) != opt_val_g.answ: #soluzione ottimale corrisponde come peso al valore ottimale
                return SEF.consistency_NO(['opt_val', 'opt_sol'], f"Il peso totale di '{opt_sol_g.alias}' e il valore '{opt_val_g.alias}' non corrispondono")
            SEF.consistency_OK(['opt_sol', 'opt_val'], f"Il peso totale di '{opt_sol_g.alias}' e il valore '{opt_val_g.alias}' corrispondono", f"Ora resta da verificare l'ottimalità")

        if 'list_opt_sols' in self.goals:
            g = self.goals['list_opt_sols']
            answ = ast.literal_eval(g.answ)
            sols_weights = [sum([self.I.edges[i][1] for i in tree]) for tree in answ]
            if len(set(sols_weights)) != 1:
                return SEF.consistency_NO(['list_opt_sols'], f"Non tutte le soluzioni in '{g.alias}' hanno lo stesso peso")
            SEF.consistency_OK(['list_opt_sols'], f"Tutte le soluzioni in '{self.goals}'", f"Ora resta da verificare l'ottimalità")
        # FIXME: E' giusto fare questo controllo oppure è obsoleto ?
        if 'list_opt_sols' in self.goals and 'num_opt_sols' in self.goals:
            list_opt_sols_g = self.goals['list_opt_sols']
            num_opt_sols_g = self.goals['num_opt_sols']
            list_opt_sols_answ = ast.literal_eval(list_opt_sols_g.answ)
            if num_opt_sols_g.answ != len(list_opt_sols_answ):
                return SEF.consistency_NO(['list_opt_sols', 'num_opt_sols'], f"Come '{list_opt_sols_g.alias}' hai inserito '{list_opt_sols_g.answ}', ma essa presenta un numero di soluzioni diverso dal valore '{num_opt_sols_g.alias}' immesso")
            SEF.consistency_OK(['list_opt_sols', 'opt_val'], f"Il numero di soluzioni di '{list_opt_sols_g.alias}' corrisponde con il valore '{num_opt_sols_g.alias}'", f"Ora resta da verificare l'ottimalità")

        if 'list_opt_sols' in self.goals and 'opt_val' in self.goals:
            list_opt_sols_g = self.goals['list_opt_sols']
            opt_val_g = self.goals['opt_val']
            list_opt_sols_answ = ast.literal_eval(list_opt_sols_g.answ)
            sols_weights = [sum([self.I.edges[i][1] for i in tree]) for tree in list_opt_sols_answ] #lista dei pesi delle soluzioni ottime
            if not all(weight == opt_val_g.answ for weight in sols_weights):
                return SEF.consistency_NO(['list_opt_sols', 'opt_val'], f"Il peso totale di alcune delle soluzioni in '{list_opt_sols_g.alias}' e il valore '{opt_val_g.alias}' non corrispondono")
            SEF.consistency_OK(['list_opt_sols', 'opt_val'], f"Il peso totale di ogni soluzione in '{list_opt_sols_g.alias}' corrisponde con il valore '{opt_val_g.alias}'", f"Ora resta da verificare l'ottimalità")

        return True

    def verify_optimality(self, SEF):
        if not super().verify_optimality(SEF):
            return False

        if 'opt_val' in self.goals:
            g = self.goals['opt_val']
            true_answ = SEF.oracle_dict['opt_val']
            if g.answ != true_answ:
                return SEF.optimality_NO(g, f"Come '{g.alias}' ha inserito '{g.answ}', tuttavia esso non è il valore minimo possibile.")
            SEF.optimality_OK(g, f"{g.alias} = {true_answ} è effettivamente il valore ottimo.", "")

        if 'opt_sol' in self.goals:
            g = self.goals['opt_sol']
            answ = ast.literal_eval(g.answ)
            list_opt_sols = sum([set(tree) for tree in SEF.oracle_dict['opt_sol']]) #non la soluzione ottima perchè l'utente potrebbe trovare una soluzione diversa da quella dell'oracolo
            if set(answ) not in list_opt_sols:
                return SEF.optimality_NO(g, f"Come '{g.alias}' hai inserito '{g.answ}', ma essa non è tra le soluzioni ottime")
            SEF.optimality_OK(g, f"{g.alias} = {g.answ} é effettivamente una possibile soluzione ottima", "")

        if 'num_opt_sols' in self.goals:
            g = self.goals['num_opt_sols']
            true_answ = SEF.oracle_dict['num_opt_sols']
            if g.answ != true_answ:
                return SEF.optimality_NO(g, f"Come '{g.alias}' hai inserito '{g.answ}', ma questo non corrisponde al numero di soluzioni ottime corretto")
            SEF.optimality_OK(g, f"{g.alias} = {g.answ} è effettivamente il numero corretto di soluzioni ottime", "")

        if 'list_opt_sols' in self.goals:
            g = self.goals['list_opt_sols']
            answ = [set(tree) for tree in ast.literal_eval(g.answ)]
            true_answ = [set(tree) for tree in SEF.oracle_dict['list_opt_sols']] #ogni soluzione sia una sottolista di tutte le soluzioni ottime #FIXME: es. utente inserisce una lista di 5 soluzioni, mentre l'oracolo ne possiede 10.
            if not all(tree in true_answ for tree in answ):
                return SEF.optimality_NO(g, f"Come '{g.alias}' hai inserito '{g.answ}', ma non tutte le soluzioni sono ottimali")
            if len(answ) < len(true_answ):
                return SEF.optimality_NO(g, f"Come '{g.alias}' hai inserito '{g.answ}', ma mancano alcune soluzioni")
            SEF.optimality_OK(g, f"{g.alias} = {g.answ} è effettivamente la lista completa di soluzioni ottime", "")

        if 'edge_profile' in self.goals:
            g = self.goals['edge_profile']
            true_answ = SEF.oracle_dict['edge_profile']
            if g.answ != true_answ:
                return SEF.optimality_NO(g, f"Come '{g.alias}' hai inserito '{g.answ}', ma la risposta non risulta corretta")
            SEF.optimality_OK(g, f"{g.alias} = {g.answ} è effettivamente la risposta corretta per l'arco richiesto", "")

        return True
