import abc
import dataclasses
import functools
import itertools
from typing import AbstractSet, Dict, FrozenSet, Iterable, Optional, Protocol, Set, Tuple, TypeVar

import pyperplan
from pddl import pddl

from .base import Action, Literal, PDDLObject, PDDLType, PDDLVariable, Predicate, TypeObjectMap

T = TypeVar("T", covariant=True)


class InvalidAssignment(ValueError):
    pass


class Lifted(Protocol[T]):

    @property
    @abc.abstractmethod
    def variables(self) -> Tuple[PDDLVariable, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def assign(self,
               values: Dict[PDDLVariable, PDDLObject],
               problem: Optional["Problem"] = None) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def valid_values(self, problem: "Problem") -> Dict[PDDLVariable, Tuple[PDDLObject, ...]]:
        raise NotImplementedError

    def ground(self, problem: "Problem") -> Iterable[T]:
        valid_values = self.valid_values(problem).items()  # type: Iterable[Tuple[PDDLVariable, Tuple[PDDLObject, ...]]]  # noqa: E501
        variables, possible_values = zip(*valid_values)
        assignments = (
            dict(zip(variables, values))
            for values in itertools.product(*possible_values)
        )
        for assignment in assignments:
            try:
                yield self.assign(assignment, problem)
            except InvalidAssignment:
                continue


@dataclasses.dataclass(frozen=True, order=True)
class LiftedLiteral(Lifted[Literal]):
    predicate: Predicate
    objects: Tuple[PDDLVariable, ...]

    @property
    def variables(self) -> Tuple[PDDLVariable, ...]:
        return self.objects

    def assign(self,
               values: Dict[PDDLVariable, PDDLObject],
               problem: Optional["Problem"] = None) -> Literal:
        objects = []
        for var in self.objects:
            obj = values[var]
            # check that assignment is valid
            if not var.isvalid(obj):
                raise ValueError(
                    f"Invalid assignment. Attempted to assign object {obj} to variable {var}")
            objects.append(obj)
        literal = self.predicate(tuple(objects))

        # Return grounded literal if valid, i.e., check that if it is static that it is part of the
        # known literals.
        if (problem is not None
                and self.predicate in problem.domain.static_predicates
                and literal not in problem.static_literals):
            raise InvalidAssignment
        return literal

    def valid_values(self, problem: "Problem") -> Dict[PDDLVariable, Tuple[PDDLObject, ...]]:
        # If this lifted literal is static, only objects inside the known literal of the same type
        # are possible assignments for this literal.
        if self.predicate in problem.domain.static_predicates:
            values_gen = zip(*(
                lit.objects for lit in problem.static_literals if lit.predicate == self.predicate))
            values_sets = [set(objects) for objects in values_gen]
            possible_values = {
                var: tuple(sorted(objects))
                for var, objects in zip(self.variables, values_sets)
            }
        else:
            possible_values = {var: problem.objects[var.types] for var in self.variables}

        return possible_values

    @classmethod
    def from_predicate(cls, predicate: Predicate) -> "LiftedLiteral":
        objects = tuple(PDDLVariable(f"{i}", types) for i, types in enumerate(predicate.types))
        return cls(predicate, objects)

    @classmethod
    def from_pyperplan(cls, predicate: pddl.Predicate) -> "LiftedLiteral":
        objects = tuple(PDDLVariable.from_pyperplan(v) for v in predicate.signature)
        return cls(Predicate.from_pyperplan(predicate), objects)


@dataclasses.dataclass(frozen=True, order=True)
class LiftedAction(Lifted[Action]):
    name: str
    preconditions: FrozenSet[LiftedLiteral]
    add_effects: FrozenSet[LiftedLiteral]
    del_effects: FrozenSet[LiftedLiteral]

    @functools.cached_property
    def variables(self) -> Tuple[PDDLVariable, ...]:  # type: ignore
        lit_vars: Iterable = itertools.chain(self.precondition_variables, self.effect_variables)
        return tuple(sorted(frozenset(lit_vars)))

    @functools.cached_property
    def precondition_variables(self) -> Tuple[PDDLVariable, ...]:
        lit_vars = itertools.chain(*[lit.variables for lit in self.preconditions])
        return tuple(sorted(frozenset(lit_vars)))

    @functools.cached_property
    def effect_variables(self) -> Tuple[PDDLVariable, ...]:
        lit_vars = (itertools.chain(*[lit.variables for lit in self.add_effects]),
                    itertools.chain(*[lit.variables for lit in self.del_effects]))
        return tuple(sorted(frozenset(itertools.chain(*lit_vars))))

    def assign(self,
               values: Dict[PDDLVariable, PDDLObject],
               problem: Optional["Problem"] = None) -> Action:
        preconditions = {lit.assign(values, problem) for lit in self.preconditions}
        # If problem is given, remove known static literals. This is not done before grounding the
        # literals in order to first catch any invalid assignment that would affect both static
        # predicates and non-static predicates.
        if problem is not None:
            preconditions = {
                lit
                for lit in preconditions
                if lit.predicate not in problem.domain.static_predicates
            }

        add_effects = {lit.assign(values, problem) for lit in self.add_effects}
        del_effects = {lit.assign(values, problem) for lit in self.del_effects}

        # STRIPS convention
        del_effects -= add_effects
        # No need to add literals already in the precondition
        add_effects -= preconditions

        return Action(
            self.name,
            frozenset(preconditions),
            frozenset(add_effects),
            frozenset(del_effects),
        )

    def valid_values(self, problem: "Problem") -> Dict[PDDLVariable, Tuple[PDDLObject, ...]]:
        possible_values = {v: problem.objects[v.types] for v in self.variables}

        # Given domain and literals, we filter out impossible assignments due to static predicates
        # by gathering all the valid values for the variables in the precondition
        precond_values: Dict[PDDLVariable, Set[PDDLObject]] = {}
        for lit in self.preconditions:
            for var, values in lit.valid_values(problem).items():
                if var in precond_values:
                    precond_values[var].intersection_update(values)
                else:
                    precond_values[var] = set(values)
        # update valid values for the variables in the precondition
        for var, vals in precond_values.items():
            possible_values[var] = tuple(sorted(vals))

        return possible_values

    @classmethod
    def from_pyperplan(cls, action: pddl.Action) -> "LiftedAction":
        name = action.name
        precondition = frozenset(
            {LiftedLiteral.from_pyperplan(p) for p in action.precondition})
        add_effects = frozenset(
            {LiftedLiteral.from_pyperplan(p) for p in action.effect.addlist})
        del_effects = frozenset(
            {LiftedLiteral.from_pyperplan(p) for p in action.effect.dellist})
        return cls(name, precondition, add_effects, del_effects)


@dataclasses.dataclass(frozen=True)
class Domain:
    name: str
    types: FrozenSet[PDDLType]
    predicates: FrozenSet[Predicate]
    actions: FrozenSet[LiftedAction]
    constants: FrozenSet[PDDLObject]

    @functools.cached_property
    def static_predicates(self) -> FrozenSet[Predicate]:
        effect_lifted = itertools.chain(*(a.add_effects | a.del_effects for a in self.actions))
        effect_preds = frozenset(lit.predicate for lit in effect_lifted)
        return self.predicates - effect_preds

    def static_literals(self, literals: AbstractSet[Literal]) -> FrozenSet[Literal]:
        return frozenset(
            lit
            for lit in literals
            if lit.predicate in self.static_predicates
        )

    @classmethod
    def from_pddl_filepath(cls, filepath) -> "Domain":
        domain = pyperplan.Parser(filepath).parse_domain()
        return cls.from_pyperplan(domain)

    @classmethod
    def from_pyperplan(cls, domain: pddl.Domain) -> "Domain":
        types = frozenset({PDDLType.from_pyperplan(t) for t in domain.types.values()})
        predicates = frozenset(
            {Predicate.from_pyperplan(p) for p in domain.predicates.values()})
        actions = frozenset(
            {LiftedAction.from_pyperplan(a) for a in domain.actions.values()})
        constants = frozenset(
            {PDDLObject.from_pyperplan(o) for o in domain.constants.items()})
        return cls(domain.name, types, predicates, actions, constants)


def parse_pyperplan_literal(literal: pddl.Predicate,
                            namedpreds: Dict[str, Predicate]) -> Literal:
    objects = tuple(PDDLObject.from_pyperplan(o) for o in literal.signature)
    lifted_lit = LiftedLiteral.from_predicate(namedpreds[literal.name])
    return lifted_lit.assign(dict(zip(lifted_lit.variables, objects)))


def parse_pyperplan_literals(literals: Iterable[pddl.Predicate],
                             domain: Domain) -> FrozenSet[Literal]:
    namedpreds = {p.name: p for p in domain.predicates}
    return frozenset(parse_pyperplan_literal(lit, namedpreds) for lit in literals)


@dataclasses.dataclass(frozen=True)
class Problem:
    name: str
    domain: Domain
    objects: TypeObjectMap
    goal: FrozenSet[Literal]
    static_literals: FrozenSet[Literal]

    @classmethod
    def from_pyperplan(cls, problem: pddl.Problem):
        name = problem.name
        domain = Domain.from_pyperplan(problem.domain)
        objects = TypeObjectMap.from_dict(problem.objects)
        goal = parse_pyperplan_literals(problem.goal, domain)
        static_literals = domain.static_literals(
            parse_pyperplan_literals(problem.initial_state, domain))
        return cls(name, domain, objects, goal, static_literals)
