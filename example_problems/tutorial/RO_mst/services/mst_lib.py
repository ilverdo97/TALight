import ast
import re
from sys import stderr
from typing import Dict

from RO_verify_submission_gen_prob_lib import verify_submission_gen

instance_objects_spec = {
    ('n', int),
    ('s', int),
    ('t', int),
    ('edges', str),
    ('forbidden_edges', str),
    ('forced_edges', str),
    ('query_edge', str),
}

answer_objects_spec = {
    'opt_sol': str,
    'opt_val': str,
    'num_opt_sols': int,
    'list_opt_sols': str,
    'edge_profile': str,
    # 'cyc_cert': str,
    # 'edgecut_cert': str,
    # 'cutshore_cert': str,
}

answer_objects_implemented = [
    'opt_sol',
    'opt_val',
    'num_opt_sols',
    'list_opt_sols',
    'edge_profile',
    # 'cyc_cert',
    # 'edgecut_cert',
    # 'cutshore_cert',
]


def check_isolated_nodes(n, edges):
    nodes_founded = set()
    for u, v, w in edges:
        nodes_founded.add(u)
        nodes_founded.add(v)
    return len(nodes_founded) != n


def check_isolated_nodes_by_forbidden(n, edges, forbidden_edges):
    nodes_founded = set()
    for i, (u, v, w) in enumerate(edges):
        if i not in forbidden_edges:
            nodes_founded.add(u)
            nodes_founded.add(v)
    return len(nodes_founded) != n


def check_instance_consistency(instance):
    print(f"instance={instance}", file=stderr)
    n = instance['n']
    m = instance['m']
    edges = instance['edges']
    forbidden_edges = instance['forbidden_edges']
    forced_edges = instance['forced_edges']
    query_edge = instance['query_edge']

    n = int(n)
    m = int(m)
    edges = ast.literal_eval(re.sub(r"[{}]", "", edges))
    forbidden_edges = ast.literal_eval(forbidden_edges)
    forced_edges = ast.literal_eval(forced_edges)
    query_edge = int(query_edge)

    if m != len(edges):
        print(f"Errore: il numero di archi non corrisponde con la lista data")
        exit(0)
    if not all(w >= 0 for _, _, w in edges):
        print(f"Errore: alcuni pesi sono minori di 0")
    if not all(0 < u < n and 0 < u < n for u, v, _ in edges):
        print(f"Errore: alcuni nodi dati negli archi non esistono")
        exit(0)
    if not all(u != v for u, v, _ in edges):
        print(f"Errore: il grafo non può contenere autoloop")
        exit(0)
    if not all(0 < e < m for e in forbidden_edges):
        print(f"Errore: alcuni archi dichiarati in forbidden_edges non esistono")
        exit(0)
    if not all(0 < e < m for e in forced_edges):
        print(f"Errore: alcuni archi dichiarati in forced_edges non esistono")
        exit(0)
    if query_edge < 0 or query_edge >= m:
        print(f"Errore: il query_edge non esiste")
        exit(0)
    if query_edge in forbidden_edges:
        print(f"Errore: il query_edge è nei forbidden_edges")
    if check_isolated_nodes(n, edges):
        print(f"Errore: sono presenti dei nodi isolati")
        exit(0)
    if check_isolated_nodes_by_forbidden(n, edges, forbidden_edges):
        print(f"Errore: sono presenti dei nodi isolati dovuti ad archi eliminati da forbidden_edges")
        exit(0)


class Graph:
    def __init__(self, vertices: int):
        self.V = vertices
        self.edges = []
        self.adjacency = [[[] for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u: int, v: int, w: float, l: int):
        self.edges.append((u, v, w, l))
        self.adjacency[u][v].append({'weight': w, 'label': l})
        self.adjacency[v][u].append({'weight': w, 'label': l})

    def _search(self, parent: list, i: int):
        return i if parent[i] == i else self._search(parent, parent[i])

    def _apply_union(self, parent: list, rank: list, x: int, y: int):
        x_root = self._search(parent, x)
        y_root = self._search(parent, y)
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

    def kruskal_constrained(self, fixed: list, excluded: list):
        result = []
        tot_w = 0
        i, e = 0, 0
        self.edges = sorted(self.edges, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        for u, v, w, l in self.edges:
            x = self._search(parent, u)
            y = self._search(parent, v)
            if l in fixed:
                e += 1
                result.append(l)
                tot_w += w
                self._apply_union(parent, rank, x, y)
        while e < self.V - 1:
            u, v, w, l = self.edges[i]
            i += 1
            x = self._search(parent, u)
            y = self._search(parent, v)
            if x != y and l not in excluded and l not in fixed:
                e += 1
                result.append(l)
                tot_w += w
                self._apply_union(parent, rank, x, y)
        return result, tot_w

    def _find_substitute(self, cut: int, tree: set, excluded: set):
        cut_u, cut_v, cut_w, cut_l = list(filter(lambda x: x[3] == cut, self.edges))[0]
        V1 = {cut_u}
        tmp_list = [cut_u]

        while tmp_list:
            print(f"current list: {tmp_list}")
            u = tmp_list.pop()
            print(f"current node u: {u}")
            for v in range(self.V):
                print(f"v = {v}")
                for edge in self.adjacency[u][v]:
                    print(edge)
                    if edge['label'] in tree and edge['label'] != cut_l and v not in V1:
                        print(f"added v: {v}")
                        tmp_list.append(v)
                        V1.add(v)
                        break

        V2 = set(range(self.V)).difference(V1)

        for u, v, w, l in self.edges:
            if l != cut_l and \
                    l not in tree and \
                    l not in excluded and \
                    ((u in V1 and v in V2) or (u in V2 and v in V1)) and \
                    w == cut_w:
                return l
        return None

    def _all_mst(self, tree: set, fixed: set, excluded: set):
        search_set = tree.difference(fixed)
        msts = []
        for e in search_set:
            sub = self._find_substitute(e, tree, excluded)
            if sub is not None:
                new_tree = tree
                new_tree.remove(e)
                new_tree.add(sub)
                msts.append(list(new_tree))
                others = self._all_mst(new_tree, fixed, excluded.union({e}))
                if others:
                    msts.append(others)
        return msts

    def all_mst(self, fixed: list, excluded: list):
        first, _ = self.kruskal_constrained(fixed, excluded)
        return [first] + self._all_mst(set(first), set(fixed), set(excluded))


def solver(input_to_oracle):
    instance = input_to_oracle['instance']
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
    for l, e in enumerate(edges):
        u, v = list(e[0])
        graph.add_edge(u, v, e[1], l)

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
                                        f"costituita da una lista di inidici riferiti alla lista degli archi "
                                        f"nell'sitanza del problema")
            try:
                g_eval = ast.literal_eval(g.answ)
                if type(g_eval) != list:
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di inidici riferiti alla lista degli archi "
                                            f"nell'sitanza del problema")
                if not all([isinstance(arch, int) for arch in g_eval]):
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di inidici riferiti alla lista degli archi "
                                            f"nell'sitanza del problema")
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
                                        f"costituita da una lista di inidici riferiti alla lista degli archi "
                                        f"nell'sitanza del problema")
            try:
                g_eval = ast.literal_eval(g.answ)
                if type(g_eval) != list:
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di inidici riferiti alla lista degli archi "
                                            f"nell'sitanza del problema")
                if not all([isinstance(graph, list) for graph in g_eval]):
                    return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                            f"immettere una lista di archi. Una lista di archi è costituita da una "
                                            f"lista di inidici riferiti alla lista degli archi "
                                            f"nell'sitanza del problema")
                for tree in g_eval:
                    if not all([isinstance(arch, int) for arch in tree]):
                        return SEF.format_NO(g, f"Come '{g.alias}' hai immesso '{g.answ}' dove era invece richiesto di "
                                                f"immettere una lista di archi. Una lista di archi è costituita da una "
                                                f"lista di inidici (interi) riferiti alla lista degli archi "
                                                f"nell'sitanza del problema")
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
        # TODO

    def verify_consistency(self, SEF):
        if not super().verify_consistency(SEF):
            return False
        # TODO

    def verify_optimality(self, SEF):
        if not super().verify_optimality(SEF):
            return False
        true_opt_val = SEF.oracle_dict['opt_val']
        true_opt_sol = SEF.oracle_dict['opt_sol']
        if 'opt_val' in self.goals:
            g_val = self.goals['opt_val']
            if true_opt_val != g_val.answ:
                return SEF.optimality_NO(g_val, f"Il valore ottimo corretto è {true_opt_val} {'>' if true_opt_val != g_val.answ else '<'} {g_val.answ}, che è il valore invece immesso in `{g_val.alias}`. Una soluzione di valore ottimo è {true_opt_sol}.")
            else:
                SEF.optimality_OK(g_val, f"{g_val.alias}={g_val.answ} è effettivamente il valore ottimo.", "")
        if 'opt_sol' in self.goals:
            g_sol = self.goals['opt_sol']
            g_sol_answ = self.goals['opt_sol'].answ
            g_val_answ = sum(
                [val for ele, cost, val in zip(self.I.labels, self.I.costs, self.I.vals) if ele in g_sol_answ])
            assert g_val_answ <= true_opt_val
            if g_val_answ < true_opt_val:
                return SEF.optimality_NO(g_sol, f"Il valore totale della soluzione immessa in `{g_sol.alias}` è {g_val_answ} < {true_opt_val}, valore corretto per una soluzione ottima quale {true_opt_sol}. La soluzione (ammissibile) che hai immesso è `{g_sol.alias}`={g_sol.answ}.")
            else:
                SEF.optimality_OK(g_sol, f"Confermo l'ottimailtà della soluzione {g_sol.alias}={g_sol.answ}.", "")
        return True
