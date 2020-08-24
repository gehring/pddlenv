import itertools
import functools
import operator
import dataclasses
import collections
import types as pytypes
from typing import Collection, ClassVar, Dict, FrozenSet, Iterable, Mapping, NamedTuple, Optional, Union, Set, Sequence, Tuple, Type, TypeVar

from pddl import pddl

_TypeTuple = Tuple[str, Optional["_TypeTuple"]]  # type: ignore
_predicate_cache: Dict[str, Type] = {}
A = TypeVar("A", bound="Action")
T = TypeVar("T", bound="TypeObjectMap")
P = TypeVar("P", bound="Problem")


class InvalidAssignment(ValueError):
    pass


class PDDLObject(NamedTuple):
    name: str


class Predicate(str):
    types: ClassVar[Tuple[Tuple[Type[PDDLObject], ...], ...]] = ()

    def __init_subclass__(cls, /, types):
        cls.types = tuple(tuple(var_types) for var_types in types)

    def __new__(cls, *objects, problem: "Problem" = None):
        if len(objects) != len(cls.types):
            raise ValueError(
                f"Predicate '{cls.__name__}' expects {len(cls.types)} objects but "
                f"{len(objects)} were given."
            )
        pred_str = super().__new__(cls, f"({cls.__name__} {' '.join((o.name for o in objects))})")
        if (problem is not None
                and cls in problem.static_predicates
                and pred_str not in problem.static_literals):
            raise InvalidAssignment
        return pred_str


class Action(str):
    __slots__ = ("objects", "preconditions", "add_effects", "del_effects")
    objects: Tuple[PDDLObject, ...]
    preconditions: FrozenSet[Predicate]
    add_effects: FrozenSet[Predicate]
    del_effects: FrozenSet[Predicate]

    types: ClassVar[Tuple[Type[PDDLObject], ...]] = ()
    pre_predicates: ClassVar[Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]] = ()
    add_predicates: ClassVar[Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]] = ()
    del_predicates: ClassVar[Tuple[Tuple[Type[Predicate], Tuple[int, ...]], ...]] = ()

    def __init_subclass__(cls, /, variables, preconditions, add_effects, del_effects, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.types = tuple(tuple(var_types) for _, var_types in variables)
        cls.pre_predicates = tuple(variable_predicate_map(variables, preconditions))
        cls.add_predicates = tuple(variable_predicate_map(variables, add_effects))
        cls.del_predicates = tuple(variable_predicate_map(variables, del_effects))

    def __new__(cls, *objects, problem):
        if len(cls.types) != len(objects):
            raise ValueError(
                f"Action '{cls.__name__}' expects {len(cls.types)} objects but "
                f"{len(objects)} were given."
            )
        return super().__new__(
            cls, f"({cls.__name__} - action {' '.join((o.name for o in objects))})")

    def __init__(self, *objects, problem: "Problem" = None):
        super().__init__()
        self.objects = objects
        preconditions = set(
            pred(*(objects[i] for i in var_idx))
            for pred, var_idx in self.pre_predicates
        )
        add_effects = set(
            pred(*(objects[i] for i in var_idx))
            for pred, var_idx in self.add_predicates
        )
        del_effects = set(
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

    @classmethod
    def ground(cls: Type[A], problem: "Problem") -> Iterable[A]:
        possible_values = (problem.objectmap[var_types] for var_types in cls.types)
        for assignment in itertools.product(*possible_values):
            try:
                yield cls(*assignment, problem=problem)
            except InvalidAssignment:
                continue


def find_static_predicates(actions: Collection[Type[Action]],
                           predicates: Collection[Type[Predicate]]) -> Collection[Type[Predicate]]:
    add_preds = itertools.chain(*(tuple(zip(*a.add_predicates))[0] for a in actions))
    del_preds = itertools.chain(*(tuple(zip(*a.del_predicates))[0] for a in actions))
    effect_preds = frozenset(itertools.chain(add_preds, del_preds))
    return frozenset(predicates) - effect_preds


class Problem(str):
    objectmap: "TypeObjectMap"
    goal: Collection[Predicate]
    static_literals: Collection[Predicate]

    types: ClassVar[Collection[Type[PDDLObject]]] = ()
    predicates: ClassVar[Collection[Type[Predicate]]] = ()
    static_predicates: ClassVar[Collection[Type[Predicate]]] = ()
    actions: ClassVar[Collection[Type[Action]]] = ()
    constants: ClassVar[Collection[PDDLObject]] = ()

    def __init_subclass__(cls, /, types, predicates, actions, constants, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.types = types
        cls.predicates = predicates
        cls.actions = actions
        cls.constants = constants
        cls.static_predicates = find_static_predicates(actions, predicates)

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

    @classmethod
    def from_pyperplan_problem(cls: Type[P], problem: pddl.Problem) -> P:
        objects = itertools.chain(
            (parse_pyperplan_type(t)(name) for name, t in problem.objects.items()),
            cls.constants,
        )
        init_literals = (parse_pyperplan_grounded_predicate(p) for p in problem.initial_state)
        return cls(
            frozenset(objects),
            tuple(parse_pyperplan_grounded_predicate(p) for p in problem.goal),
            tuple(lit for lit in init_literals if type(lit) in cls.static_predicates),
        )


@dataclasses.dataclass(frozen=True)
class TypeObjectMap:
    objects: Tuple[PDDLObject]
    typemap: Dict[Type[PDDLObject], Collection[PDDLObject]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set),
        init=False,
        repr=False,
        compare=False,
    )
    _subtypes: Dict[Type[PDDLObject], Collection[Type[PDDLObject]]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set),
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, objects: Iterable[PDDLObject]):
        super().__setattr__("objects", tuple(sorted(objects, key=operator.attrgetter("name"))))
        super().__setattr__("typemap", collections.defaultdict(set))
        super().__setattr__("_subtypes", collections.defaultdict(set))
        for o in objects:
            self._add_to_subtypes(type(o))
            self.typemap[type(o)].add(o)

    def _add_to_subtypes(self, pddltype: Type[PDDLObject]):
        for parent_type in pddltype.__bases__:  # type: Type[PDDLObject]
            if isinstance(parent_type, PDDLObject):
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

    @classmethod
    def from_pyperplan_dict(cls: Type[T], objects: Dict[str, pddl.Type]) -> T:
        return cls(parse_pyperplan_type(t)(name) for name, t in objects.items())


def define_object_type(name, parent=None) -> Type[PDDLObject]:
    parent = parent or PDDLObject
    return type(name, (parent,), {})  # type: Type[PDDLObject]


def predicate(name: str, types: Tuple[Type[PDDLObject]], strict_type_check=False) -> Type[Predicate]:
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
                  preconditions: Collection[Tuple[Type[Predicate], Tuple[str]]],
                  add_effects: Collection[Tuple[Type[Predicate], Tuple[str]]],
                  del_effects: Collection[Tuple[Type[Predicate], Tuple[str]]]):
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
    return pytypes.new_class(  # type: Type["Problem"]
        name,
        (Problem,),
        {"types": types,
         "predicates": predicates,
         "actions": actions,
         "constants": constants},
    )


def _pyperplan_type_to_tuple(pddltype: pddl.Type) -> _TypeTuple:
    parent_type = None
    if pddltype.parent is not None and pddltype.parent.name != "object":
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


def parse_pyperplan_type(pddltype: pddl.Type) -> Type[PDDLObject]:
    return _object_type_from_type_tuple(_pyperplan_type_to_tuple(pddltype))


def parse_pyperplan_predicate(pred: pddl.Predicate) -> Type[Predicate]:
    types = [
        tuple(parse_pyperplan_type(pddltype) for pddltype in var_types)
        for var_name, var_types in pred.signature
    ]
    return predicate(pred.name, types)


def parse_pyperplan_lifted_predicate(predicate: pddl.Predicate
                                     ) -> Tuple[Type[Predicate], Tuple[str, ...]]:
    pred = parse_pyperplan_predicate(predicate)
    var_names, _ = zip(*predicate.signature)
    return pred, var_names


def parse_pyperplan_grounded_predicate(pred: pddl.Predicate) -> Predicate:
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
    pred_cls = predicate(pred.name, tuple((t,) for t in object_types))
    return pred_cls(*objects)


def parse_pyperplan_action(action: pddl.Action) -> Type["Action"]:
    variables = [
        (var_name, tuple(parse_pyperplan_type(t) for t in types))
        for var_name, types in action.signature
    ]

    return define_action(  # type: Type[Action]
        action.name,
        variables,
        [parse_pyperplan_lifted_predicate(pred) for pred in action.precondition],
        [parse_pyperplan_lifted_predicate(pred) for pred in action.effect.addlist],
        [parse_pyperplan_lifted_predicate(pred) for pred in action.effect.dellist],
    )


def parse_pyperplan_domain(domain: pddl.Domain) -> Type["Problem"]:
    types = tuple(sorted(
        (parse_pyperplan_type(t) for t in domain.types.values()),
        key=operator.attrgetter("__name__"),
    ))
    predicates = tuple(sorted(
        (parse_pyperplan_predicate(p) for p in domain.predicates.values()),
        key=operator.attrgetter("__name__"),
    ))
    actions = tuple(sorted(
        (parse_pyperplan_action(a) for a in domain.actions.values()),
        key=operator.attrgetter("__name__"),
    ))
    constants = tuple(sorted(parse_pyperplan_type(t)(name) for name, t in domain.constants.items()))
    return define_problem(domain.name, types, predicates, actions, constants)
