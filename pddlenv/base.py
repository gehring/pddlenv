import collections
import dataclasses
import functools
import itertools
from typing import AbstractSet, Dict, FrozenSet, Iterable, Optional, Set, Tuple, TypeVar, Union

from pddl import pddl

T = TypeVar("T")
_TypeTuple = Tuple[str, Optional["_TypeTuple"]]  # type: ignore


def _pyperplan_type_to_tuple(type: pddl.Type) -> _TypeTuple:
    parent_type = None
    if type.parent is not None:
        parent_type = _pyperplan_type_to_tuple(type.parent)
    return type.name, parent_type


@dataclasses.dataclass(frozen=True, order=True)
class PDDLType:
    name: str
    parent: Optional["PDDLType"] = dataclasses.field(default=None, repr=False)

    def isinstanceof(self, pddltype: "PDDLType"):
        return pddltype in self.type_hierarchy

    @functools.cached_property
    def type_hierarchy(self) -> FrozenSet["PDDLType"]:
        types = {self}
        current_type = self
        while current_type.parent is not None:
            current_type = current_type.parent
            types.add(current_type)
        return frozenset(types)

    @classmethod
    def from_pyperplan(cls, pddltype: pddl.Type) -> "PDDLType":
        return cls._from_type_tuple(_pyperplan_type_to_tuple(pddltype))

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _from_type_tuple(cls, pddltype: _TypeTuple) -> "PDDLType":
        parent = pddltype[1]
        if parent is not None:
            parent = cls._from_type_tuple(parent)
        return cls(pddltype[0], parent)


@dataclasses.dataclass(frozen=True, order=True)
class PDDLObject:
    name: str
    type: PDDLType

    @classmethod
    def from_pyperplan(cls, pddlobject: Tuple[str, Union[pddl.Type, Tuple[pddl.Type]]]
                       ) -> "PDDLObject":
        pddltype = pddlobject[1]
        if isinstance(pddltype, tuple):
            assert len(pddltype) == 1
            pddltype = pddltype[0]
        return cls(pddlobject[0], PDDLType.from_pyperplan(pddltype))


@dataclasses.dataclass(frozen=True, order=True)
class PDDLVariable:
    name: str
    types: FrozenSet[PDDLType]

    def isvalid(self, pddlobject: PDDLObject) -> bool:
        return any(pddlobject.type.isinstanceof(t) for t in self.types)

    @classmethod
    def from_pyperplan(cls, variable: Tuple[str, Iterable[pddl.Type]]) -> "PDDLVariable":
        name = variable[0]
        types = frozenset({PDDLType.from_pyperplan(t) for t in variable[1]})
        return cls(name, types)


@dataclasses.dataclass(frozen=True, order=True)
class Predicate:
    name: str
    types: Tuple[FrozenSet[PDDLType], ...] = dataclasses.field(repr=False)

    def __call__(self, objects: Tuple[PDDLObject, ...]) -> "Literal":
        return Literal(self, objects)

    @classmethod
    def from_pyperplan(cls, predicate: pddl.Predicate) -> "Predicate":
        name: str = predicate.name
        types = tuple(PDDLVariable.from_pyperplan(v).types for v in predicate.signature)
        return cls(name, types)


@dataclasses.dataclass(frozen=True, eq=False)  # we want the default string equality and hash
class Literal(str):
    predicate: Predicate
    objects: Tuple[PDDLObject, ...]

    def __new__(cls, predicate: Predicate, objects: Tuple[PDDLObject, ...]):
        return super().__new__(cls, f"({predicate.name} {' '.join((o.name for o in objects))})")  # type: ignore  # noqa: E501


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
    preconditions: FrozenSet[Literal]
    add_effects: FrozenSet[Literal]
    del_effects: FrozenSet[Literal]

    def applicable(self, literals: AbstractSet[Literal]) -> bool:
        return self.preconditions <= literals

    def apply(self, literals: AbstractSet[Literal]) -> AbstractSet[Literal]:
        return (literals - self.del_effects) | self.add_effects

    def literals(self):
        return itertools.chain(self.preconditions, self.add_effects, self.del_effects)


@dataclasses.dataclass(frozen=True)
class TypeObjectMap:
    objects: FrozenSet[PDDLObject]
    typemap: Dict[PDDLType, Set[PDDLObject]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set),
        init=False,
        repr=False,
        compare=False,
    )
    _subtypes: Dict[PDDLType, Set[PDDLType]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(set),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        for o in self.objects:
            self._add_to_subtypes(o.type)
            self.typemap[o.type].add(o)

    def _add_to_subtypes(self, pddltype: PDDLType):
        if pddltype.parent is not None:
            self._subtypes[pddltype.parent].add(pddltype)
            self._add_to_subtypes(pddltype.parent)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, key: Union[PDDLType, Iterable[PDDLType]]) -> Tuple[PDDLObject, ...]:
        if isinstance(key, PDDLType):
            objects = frozenset(itertools.chain(
                self.typemap.get(key, {}), *(self[t] for t in self._subtypes[key])))
        elif isinstance(key, Iterable):
            objects = frozenset(itertools.chain(*(self[t] for t in key)))
        else:
            raise TypeError
        return tuple(sorted(objects))

    @classmethod
    def from_dict(cls, objects: Dict[str, pddl.Type]) -> "TypeObjectMap":
        return cls(
            frozenset((PDDLObject.from_pyperplan(o) for o in objects.items())))
