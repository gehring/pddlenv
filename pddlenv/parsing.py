from typing import FrozenSet, Tuple, Type

import pyperplan
from pddl import pddl

import pddlenv.base as base


def parse_pyperplan_predicate(pred: pddl.Predicate) -> Type["base.Predicate"]:
    types = [
        tuple(parse_pyperplan_type(pddltype) for pddltype in var_types)
        for var_name, var_types in pred.signature
    ]
    return base.predicate(pred.name, types)


def parse_pyperplan_lifted_predicate(predicate: pddl.Predicate
                                     ) -> Tuple[Type["base.Predicate"], Tuple[str, ...]]:
    pred = parse_pyperplan_predicate(predicate)
    var_names, _ = zip(*predicate.signature)
    return pred, var_names


def parse_pyperplan_action(action: pddl.Action) -> Type["base.Action"]:
    variables = [
        (var_name, tuple(parse_pyperplan_type(t) for t in types))
        for var_name, types in action.signature
    ]

    return base.define_action(
        action.name,
        variables,
        [parse_pyperplan_lifted_predicate(pred) for pred in action.precondition],
        [parse_pyperplan_lifted_predicate(pred) for pred in action.effect.addlist],
        [parse_pyperplan_lifted_predicate(pred) for pred in action.effect.dellist],
    )


def parse_pyperplan_type(pddltype: pddl.Type) -> Type["base.PDDLObject"]:
    return base._object_type_from_type_tuple(base._pyperplan_type_to_tuple(pddltype))


def parse_pyperplan_grounded_predicate(pred: pddl.Predicate) -> "base.Predicate":
    object_names = []
    object_types = []
    for name, types in pred.signature:
        object_names.append(name)
        if isinstance(types, pddl.Type):
            object_types.append(parse_pyperplan_type(types))
        else:
            if len(types) != 1:
                raise ValueError("Received more than one type for object.")
            object_types.append(parse_pyperplan_type(types[0]))

    objects = tuple(t(name) for t, name in zip(object_types, object_names))
    pred_cls = base.predicate(pred.name, tuple((t,) for t in object_types))
    return pred_cls(*objects)


def parse_pyperplan_domain(domain: pddl.Domain) -> Type["base.Problem"]:
    types = tuple(parse_pyperplan_type(t) for t in domain.types.values())
    predicates = tuple(parse_pyperplan_predicate(p) for p in domain.predicates.values())
    actions = tuple(parse_pyperplan_action(a) for a in domain.actions.values())
    constants = tuple(sorted(parse_pyperplan_type(t)(name) for name, t in domain.constants.items()))
    return base.define_problem(domain.name, types, predicates, actions, constants)


def parse_pddl_domain(domain_path: str) -> Type["base.Problem"]:
    parser = pyperplan.Parser(domain_path)
    d = parser.parse_domain()
    return parse_pyperplan_domain(d)


def parse_pyperplan_problem(problem: pddl.Problem
                            ) -> Tuple[FrozenSet["base.Predicate"], "base.Problem"]:
    Domain = parse_pyperplan_domain(problem.domain)
    init_literals = frozenset(
        parse_pyperplan_grounded_predicate(p)
        for p in problem.initial_state
    )
    return init_literals, Domain.from_pyperplan_problem(problem)


def parse_pddl_problem(domain_path: str,
                       problem_path: str) -> Tuple[FrozenSet["base.Predicate"], "base.Problem"]:
    parser = pyperplan.Parser(domain_path, problem_path)
    d = parser.parse_domain()
    p = parser.parse_problem(d)
    return parse_pyperplan_problem(p)
