import collections
import dataclasses
import functools
import itertools
import operator
import types as pytypes
from typing import (AbstractSet, ClassVar, Collection, Dict, FrozenSet, Iterable, Optional,
                    Protocol, Sequence, Set, Tuple, Type, TypeVar, Union)

from pddl import pddl

from pddlenv import parsing

_TypeTuple = Tuple[str, Optional["_TypeTuple"]]  # type: ignore
_predicate_cache: Dict[str, Type] = {}
A = TypeVar("A", bound="Action")
T = TypeVar("T", bound="TypeObjectMap")
P = TypeVar("P", bound="Problem")


class InvalidAssignment(ValueError):
    pass


class PDDLObject(str):
    __slots__ = ()


class ArityObject(Protocol):
    arity: ClassVar[int]


class Predicate(str):
    __slots__ = ("objects",)
    objects: Tuple[PDDLObject, ...]
    arity: ClassVar[int] = 0
    types: ClassVar[Tuple[Tuple[Type[PDDLObject], ...], ...]] = ()

    def __init_subclass__(cls, /, types):
        cls.types = tuple(tuple(var_types) for var_types in types)
        cls.arity = len(cls.types)

    def __new__(cls, *objects, problem: "Problem" = None):
        if len(objects) != len(cls.types):
            raise ValueError(
                f"Predicate '{cls.__name__}' expects {len(cls.types)} objects but "
                f"{len(objects)} were given."
            )
        pred_str = super().__new__(cls, f"({cls.__name__} {' '.join(objects)})")  # type: ignore
        if (problem is not None
                and cls in problem.static_predicates
                and pred_str not in problem.static_literals):
            raise InvalidAssignment
        return pred_str

    def __init__(self, *objects, problem=None):
        self.objects = objects


class Action(str):
    __slots__ = ("objects", "preconditions", "add_effects", "del_effects")
    objects: Tuple[PDDLObject, ...]
    preconditions: FrozenSet[Predicate]
    add_effects: FrozenSet[Predicate]
    del_effects: FrozenSet[Predicate]

    types: ClassVar[Tuple[Type[PDDLObject], ...]] = ()
    arity: ClassVar[int] = 0
    pre_predicates: ClassVar[Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]] = ()
    add_predicates: ClassVar[Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]] = ()
    del_predicates: ClassVar[Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]] = ()

    def __init_subclass__(cls, /, variables, preconditions, add_effects, del_effects, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.types = tuple(tuple(var_types) for _, var_types in variables)
        cls.arity = len(cls.types)
        cls.pre_predicates = tuple(variable_predicate_map(variables, preconditions))
        cls.add_predicates = tuple(variable_predicate_map(variables, add_effects))
        cls.del_predicates = tuple(variable_predicate_map(variables, del_effects))

    def __new__(cls, *objects, problem=None):
        if len(cls.types) != len(objects):
            raise ValueError(
                f"Action '{cls.__name__}' expects {len(cls.types)} objects but "
                f"{len(objects)} were given."
            )
        return super().__new__(
            cls, f"{cls.__name__}({' '.join(objects)})")

    def __init__(self, *objects, problem: "Problem" = None):
        super().__init__()
        self.objects = objects
        preconditions: Set[Predicate] = set(
            pred(*(objects[i] for i in var_idx), problem=problem)
            for pred, var_idx in self.pre_predicates
        )
        add_effects: Set[Predicate] = set(
            pred(*(objects[i] for i in var_idx))
            for pred, var_idx in self.add_predicates
        )
        del_effects: Set[Predicate] = set(
            pred(*(objects[i] for i in var_idx))
            for pred, var_idx in self.del_predicates
        )

        # If problem is given, remove known static literals. This is not done before grounding the
        # literals in order to first catch any invalid assignment that would affect both static
        # predicates and non-static predicates.
        if problem is not None:
            preconditions -= problem.static_literals

        # STRIPS convention
        del_effects -= add_effects
        # No need to add literals already in the precondition
        add_effects -= preconditions

        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    @property
    def name(self) -> str:
        # to maintain compatibility with pyperplan's heuristics
        return self

    def applicable(self, literals: AbstractSet[Predicate]) -> bool:
        return self.preconditions <= literals

    def apply(self, literals: AbstractSet[Predicate]) -> AbstractSet[Predicate]:
        return (literals - self.del_effects) | self.add_effects

    def literals(self):
        return itertools.chain(self.preconditions, self.add_effects, self.del_effects)

    @classmethod
    def ground(cls: Type[A], problem: "Problem") -> Iterable[A]:
        possible_values = (problem.objectmap[var_types] for var_types in cls.types)
        for assignment in itertools.product(*possible_values):
            try:
                yield cls(*assignment, problem=problem)
            except InvalidAssignment:
                continue


def _predicates_from_effect(effect: Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]
                            ) -> Tuple[Type[Predicate], ...]:
    preds: Tuple[Type[Predicate], ...] = ()
    if effect:
        preds = tuple(zip(*effect))[0]  # type: ignore
    return preds


def find_static_predicates(actions: Collection[Type[Action]],
                           predicates: Collection[Type[Predicate]]) -> Collection[Type[Predicate]]:
    add_preds: Iterable[Type[Predicate]] = itertools.chain(
        *(_predicates_from_effect(a.add_predicates) for a in actions))
    del_preds: Iterable[Type[Predicate]] = itertools.chain(
        *(_predicates_from_effect(a.del_predicates) for a in actions))
    effect_preds = frozenset(itertools.chain(add_preds, del_preds))
    return frozenset(predicates) - effect_preds


class Problem(str):
    objectmap: "TypeObjectMap"
    goal: FrozenSet[Predicate]
    static_literals: FrozenSet[Predicate]

    types: ClassVar[Collection[Type[PDDLObject]]] = ()
    predicates: ClassVar[Collection[Type[Predicate]]] = ()
    static_predicates: ClassVar[Collection[Type[Predicate]]] = ()
    actions: ClassVar[Collection[Type[Action]]] = ()
    constants: ClassVar[Collection[PDDLObject]] = ()

    def __init_subclass__(cls, /, types, predicates, actions, constants, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.types = tuple(sorted(types, key=operator.attrgetter("__name__")))
        cls.predicates = tuple(sorted(predicates, key=operator.attrgetter("__name__")))
        cls.actions = tuple(sorted(actions, key=operator.attrgetter("__name__")))
        cls.constants = tuple(sorted(constants))
        cls.static_predicates = find_static_predicates(cls.actions, cls.predicates)

    def __new__(cls, objects, goal, static_literals):
        return super().__new__(
            cls,
            (f"{cls.__name__}(objects={sorted(objects)}, goal={sorted(goal)}, "
             f"static_literals={sorted(static_literals)})"),
        )

    def __init__(self, objects, goal, static_literals):
        super().__init__()
        self.objectmap = TypeObjectMap(objects)
        self.goal = frozenset(goal)
        self.static_literals = frozenset(static_literals)

    @property
    def objects(self):
        return self.objectmap.objects

    @functools.cached_property
    def grounded_actions(self) -> Tuple[Action, ...]:
        return tuple(itertools.chain(*(a.ground(self) for a in self.actions)))

    def goal_satisfied(self, literals: AbstractSet[Predicate]) -> bool:
        return self.goal <= literals

    def valid_actions(self, literals: AbstractSet[Predicate]) -> Tuple[Action, ...]:
        return tuple(a for a in self.grounded_actions if a.applicable(literals))

    @classmethod
    def from_pyperplan_problem(cls: Type[P], problem: pddl.Problem) -> P:
        objects = itertools.chain(
            (parsing.parse_pyperplan_type(t)(name) for name, t in problem.objects.items()),
            cls.constants,
        )
        init_literals = (
            parsing.parse_pyperplan_grounded_predicate(p)
            for p in problem.initial_state
        )
        return cls(
            frozenset(objects),
            tuple(parsing.parse_pyperplan_grounded_predicate(p) for p in problem.goal),
            tuple(lit for lit in init_literals if type(lit) in cls.static_predicates),
        )


@dataclasses.dataclass(frozen=True)
class TypeObjectMap:
    objects: Tuple[PDDLObject]
    typemap: Dict[Type[PDDLObject], Set[PDDLObject]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set),
        init=False,
        repr=False,
        compare=False,
    )
    _subtypes: Dict[Type[PDDLObject], Set[Type[PDDLObject]]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set),
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, objects: Iterable[PDDLObject]):
        super().__setattr__("objects", tuple(sorted(objects)))
        super().__setattr__("typemap", collections.defaultdict(set))
        super().__setattr__("_subtypes", collections.defaultdict(set))
        for o in objects:
            self._add_to_subtypes(type(o))
            self.typemap[type(o)].add(o)

    def _add_to_subtypes(self, pddltype: Type[PDDLObject]):
        for parent_type in pddltype.__bases__:  # type: Type[PDDLObject]
            if issubclass(parent_type, PDDLObject):
                self._subtypes[parent_type].add(pddltype)
                self._add_to_subtypes(parent_type)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self,
                    key: Union[Type[PDDLObject], Iterable[Type[PDDLObject]]]
                    ) -> Tuple[PDDLObject, ...]:
        if isinstance(key, Iterable):
            objects = itertools.chain(*(self[t] for t in key))
        elif issubclass(key, PDDLObject):
            objects = itertools.chain(
                self.typemap.get(key, {}),
                *(self[t] for t in self._subtypes[key])
            )
        else:
            raise TypeError
        return tuple(sorted(frozenset(objects)))


def define_object_type(name, parent=None) -> Type[PDDLObject]:
    if name == "object":
        assert parent is None
        return PDDLObject
    parent = parent or PDDLObject
    return pytypes.new_class(name, (parent,))


def predicate(name: str,
              types: Sequence[Tuple[Type[PDDLObject], ...]],
              strict_type_check: bool = False) -> Type[Predicate]:
    predicate_cls = _predicate_cache.get(name, None)
    if predicate_cls is None:
        predicate_cls = pytypes.new_class(name, (Predicate,), {"types": types})
        _predicate_cache[name] = predicate_cls
    else:
        for i, (pred_var_types, var_types) in enumerate(zip(predicate_cls.types, types)):
            valid_signature = True
            if strict_type_check:
                valid_signature = set(pred_var_types) != set(var_types)
            else:
                for t in var_types:
                    valid_signature = issubclass(t, pred_var_types)
                    if not valid_signature:
                        break
            if not valid_signature:
                raise TypeError(
                    "A predicate already exists with the same name but with a incompatible type "
                    f"signature. Types at index {i} differ.\n"
                    f"Existing predicate types: {pred_var_types}\n"
                    f"Given types: {var_types}\n"
                )

    return predicate_cls


def define_action(name: str,
                  variables: Sequence[Tuple[str, Tuple[Type[PDDLObject], ...]]],
                  preconditions: Collection[Tuple[Type[Predicate], Tuple[str, ...]]],
                  add_effects: Collection[Tuple[Type[Predicate], Tuple[str, ...]]],
                  del_effects: Collection[Tuple[Type[Predicate], Tuple[str, ...]]]):
    return pytypes.new_class(
        name,
        (Action,),
        {"variables": variables,
         "preconditions": preconditions,
         "add_effects": add_effects,
         "del_effects": del_effects},
    )


def define_problem(name: str,
                   types: Collection[Type[PDDLObject]],
                   predicates: Collection[Type[Predicate]],
                   actions: Collection[Type[Action]],
                   constants: Collection[PDDLObject]) -> Type["Problem"]:
    return pytypes.new_class(
        name,
        (Problem,),
        {"types": types,
         "predicates": predicates,
         "actions": actions,
         "constants": constants},
    )


def _pyperplan_type_to_tuple(pddltype: pddl.Type) -> _TypeTuple:
    parent_type = None
    if pddltype.parent is not None:
        parent_type = _pyperplan_type_to_tuple(pddltype.parent)
    return pddltype.name, parent_type


@functools.lru_cache(maxsize=None)
def _object_type_from_type_tuple(pddltype) -> Type[PDDLObject]:
    parent = pddltype[1]
    if parent is not None:
        parent = _object_type_from_type_tuple(parent)
    return define_object_type(pddltype[0], parent)


def variable_predicate_map(signature: Sequence[Tuple[str, Tuple[Type[PDDLObject], ...]]],
                           predicates: Sequence[Tuple[Type[Predicate], Tuple[str, ...]]]):
    indices = {name: idx for idx, (name, _) in enumerate(signature)}
    pred_var_index = tuple(
        (p, tuple(indices[name] for name in var_names))
        for p, var_names in predicates
    )
    return pred_var_index
