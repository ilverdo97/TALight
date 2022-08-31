import ast
import re
import networkx as nx
from sys import stderr
from typing import Dict

from RO_verify_submission_gen_prob_lib import verify_submission_gen

instance_objects_spec = {
    ('n', int),
    ('m', int),
    ('edges', str),
    ('forbidden_edges', str),
    ('forced_edges', str),
    ('query_edge', str),
}

answer_objects_spec = {
    'opt_sol': str,
    'opt_val': int,
    'num_opt_sols': int,
    'list_opt_sols': str,
    'edge_profile': str
}

answer_objects_implemented = [
    'opt_sol',
    'opt_val',
    'num_opt_sols',
    'list_opt_sols',
    'edge_profile'
]


def check_isolated_nodes(n: int, edges: list) -> bool:
    nodes_found = set()
    for u, v, w in edges:
        nodes_found.add(u)
        nodes_found.add(v)
    return len(nodes_found) != n


def check_isolated_nodes_by_forbidden(n: int, edges: list, forbidden_edges: list) -> bool:
    nodes_found = set()
    for i, (u, v, w) in enumerate(edges):
        if i not in forbidden_edges:
            nodes_found.add(u)
            nodes_found.add(v)
    return len(nodes_found) != n


def check_tree(tree: list, n: int, edges: list) -> bool:
    graph = nx.MultiGraph(n)
    for i in tree:
        u, v = list(edges[i][0])
        graph.add_edge(u, v)
    return nx.is_tree(graph)


def check_spanning(tree: list, n: int, edges: list) -> bool:
    nodes_found = set()
    for i in tree:
        nodes_found = nodes_found.union(edges[i][0])
    return len(nodes_found) == n


def check_weight_in_range(input_weight: float, edges: list, nodes: int) -> int:
    weights = sorted([w for _, w in edges])
    min_weight = sum(weights[:nodes - 1])
    max_weight = sum(weights[-(nodes-1):])
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
    def __init__(self, vertices: int):
        self.V = vertices
        self.edges = []
        self.adjacency = [[[] for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u: int, v: int, weight: float, label: int) -> None:
        self.edges.append((u, v, weight, label))
        self.adjacency[u][v].append({'weight': weight, 'label': label})
        self.adjacency[v][u].append({'weight': weight, 'label': label})

    def _search_root(self, parent: list, i: int) -> int:
        return i if parent[i] == i else self._search_root(parent, parent[i])

    def _apply_union(self, parent: list, rank: list, u: int, v: int):
        u_root = self._search_root(parent, u)
        v_root = self._search_root(parent, v)
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        elif rank[u_root] > rank[v_root]:
            parent[v_root] = u_root
        else:
            parent[v_root] = u_root
            rank[u_root] += 1

    def kruskal_constrained(self, forced: list, excluded: list) -> (list, int):
        mst = []
        i, e, tot_weight = 0, 0, 0
        self.edges = sorted(self.edges, key=lambda item: item[2])
        parent = list(range(self.V))
        rank = [0] * self.V
        for u, v, weight, label in self.edges:
            u_root = self._search_root(parent, u)
            v_root = self._search_root(parent, v)
            if label in forced:
                e += 1
                mst.append(label)
                tot_weight += weight
                self._apply_union(parent, rank, u_root, v_root)
        while e < self.V - 1:
            u, v, weight, label = self.edges[i]
            i += 1
            u_root = self._search_root(parent, u)
            v_root = self._search_root(parent, v)
            if u_root != v_root and label not in excluded and label not in forced:
                e += 1
                mst.append(label)
                tot_weight += weight
                self._apply_union(parent, rank, u_root, v_root)
        return mst, tot_weight

    def _find_substitute(self, cut: int, tree: set, excluded: set) -> int | None:
        cut_u, cut_v, cut_w, cut_l = list(filter(lambda x: x[3] == cut, self.edges))[0]
        partition = {cut_u}
        tmp_list = [cut_u]

        while tmp_list:
            u = tmp_list.pop()
            for v in range(self.V):
                for edge in self.adjacency[u][v]:
                    if edge['label'] in tree and edge['label'] != cut_l and v not in partition:
                        tmp_list.append(v)
                        partition.add(v)
                        break

        for u, v, weight, label in self.edges:
            if label != cut_l and \
                    label not in tree and \
                    label not in excluded and \
                    (u in partition ^ v in partition) and \
                    weight == cut_w:
                return label
        return None

    def _all_mst(self, tree: set, forced: set, excluded: set) -> list:
        search_set = tree.difference(forced)
        msts = []
        for edge in search_set:
            sub = self._find_substitute(edge, tree, excluded)
            if sub is not None:
                new_tree = tree
                new_tree.remove(edge)
                new_tree.add(sub)
                msts.append(list(new_tree))
                others = self._all_mst(new_tree, forced, excluded.union({edge}))
                if others:
                    msts += others
        return msts

    def all_mst(self, forced: list, excluded: list) -> list:
        first, _ = self.kruskal_constrained(forced, excluded)
        return [first] + self._all_mst(set(first), set(forced), set(excluded))


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
    for label, edge in enumerate(edges):
        u, v = list(edge[0])
        weight = float(edge[1])
        graph.add_edge(u, v, weight, label)

    opt_sol, opt_val = graph.kruskal_constrained(forced_edges, forbidden_edges)
    list_opt_sols = graph.all_mst(forced_edges, forbidden_edges)
    num_opt_sols = len(list_opt_sols)
    edge_in = 0
    edge_profile = 'in_no'
    for mst in list_opt_sols:
        if query_edge in mst:
            edge_in += 1
    if edge_in == len(list_opt_sols):
        edge_profile = 'in_all'
    elif edge_in > 0:
        edge_profile = 'in_some_but_not_in_all'

    print(f"input_to_oracle={input_to_oracle}", file=stderr)
    input_data = input_to_oracle["input_data_assigned"]
    print(f"Instance={input_data}", file=stderr)
    oracle_answers = {}
    for std_name, ad_hoc_name in input_to_oracle["request"].items():
        oracle_answers[ad_hoc_name] = locals()[std_name]
    print(oracle_answers)
    return oracle_answers


class verify_submission_problem_specific(verify_submission_gen):
    def __init__(self, SEF, input_data_assigned: Dict, long_answer_dict: Dict, oracle_response: Dict = None):
        super().__init__(SEF, input_data_assigned, long_answer_dict, oracle_response)

    def verify_format(self, SEF):
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

    def verify_feasibility(self, SEF):
        if not super().verify_feasibility(SEF):
            return False

        if 'opt_sol' in self.goals:
            g = self.goals['opt_sol']
            answ = ast.literal_eval(g.answ)
            edges = ast.literal_eval(self.I.edges)
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
            answ = g.answ
            edges = ast.literal_eval(self.I.edges)
            if (res := check_weight_in_range(answ, edges, self.I.n)) != 0:
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

    def verify_consistency(self, SEF):
        if not super().verify_consistency(SEF):
            return False

        if 'opt_sol' in self.goals and 'opt_val' in self.goals:
            opt_sol_g = self.goals['opt_sol']
            opt_val_g = self.goals['opt_val']
            opt_sol_answ = ast.literal_eval(opt_sol_g.answ)
            if sum([self.I.edges[i][1] for i in opt_sol_answ]) != opt_val_g.answ:
                return SEF.consistency_NO(['opt_val', 'opt_sol'], f"Il peso totale di '{opt_sol_g.alias}' e il valore '{opt_val_g.alias}' non corrispondono")
            SEF.consistency_OK(['opt_sol', 'opt_val'], f"Il peso totale di '{opt_sol_g.alias}' e il valore '{opt_val_g.alias}' corrispondono", f"Ora resta da verificare l'ottimalità")

        if 'list_opt_sols' in self.goals:
            g = self.goals['list_opt_sols']
            answ = ast.literal_eval(g.answ)
            sols_weights = [sum([self.I.edges[i][1] for i in tree]) for tree in answ]
            if len(set(sols_weights)) != 1:
                return SEF.consistency_NO(['list_opt_sols'], f"Non tutte le soluzioni in '{g.alias}' hanno lo stesso peso")
            SEF.consistency_OK(['list_opt_sols'], f"Tutte le soluzioni in '{self.goals}'", f"Ora resta da verificare l'ottimalità")

        if 'list_opt_sols' in self.goals and 'num_opt_sols' in self.goals:
            list_opt_sols_g = self.goals['opt_sol']
            num_opt_sols_g = self.goals['opt_val']
            list_opt_sols_answ = ast.literal_eval(list_opt_sols_g.answ)
            if num_opt_sols_g.answ != len(list_opt_sols_answ):
                return SEF.consistency_NO(['list_opt_sols', 'num_opt_sols'], f"Come '{list_opt_sols_g.alias}' hai inserito '{list_opt_sols_g.answ}', ma essa presenta un numero di soluzioni diverso dal valore '{num_opt_sols_g.alias}' immesso")
            SEF.consistency_OK(['list_opt_sols', 'opt_val'], f"Il numero di soluzioni di '{list_opt_sols_g.alias}' corrisponde con il valore '{num_opt_sols_g.alias}'", f"Ora resta da verificare l'ottimalità")

        if 'list_opt_sols' in self.goals and 'opt_val' in self.goals:
            list_opt_sols_g = self.goals['opt_sol']
            opt_val_g = self.goals['opt_val']
            list_opt_sols_answ = ast.literal_eval(list_opt_sols_g.answ)
            sols_weights = [sum([self.I.edges[i][1] for i in tree]) for tree in list_opt_sols_answ]
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
            list_opt_sols = sum([set(tree) for tree in SEF.oracle_dict['opt_sol']])
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
            true_answ = [set(tree) for tree in SEF.oracle_dict['list_opt_sols']]
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
