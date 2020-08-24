import pytest

from pddlenv import base


@pytest.fixture(scope="session", name="problem")
def make_problem():
    Cat = base.define_object_type("Cat")
    Hat = base.define_object_type("Hat")
    House = base.define_object_type("House")

    In = base.predicate("In", ((Cat,), (Hat,)))
    Out = base.predicate("Out", ((Cat,), (House,)))
    Clean = base.predicate("Clean", ((House,),))
    Dirty = base.predicate("Dirty", ((House,),))
    ParentsHappy = base.predicate("ParentsHappy", ((House,),))

    CleanHouse = base.define_action(
        "CleanHouse",
        (("?c", (Cat,)), ("?h", (House,))),
        [(Dirty, ("?h",)), (Out, ("?c", "?h"))],
        [(ParentsHappy, ("?h",)), (Clean, ("?h",))],
        [(Dirty, ("?h",))],
    )
    LetIn = base.define_action(
        "LetIn",
        (("?c", (Cat,)), ("?hat", (Hat,)), ("?h", (House,))),
        [(Out, ("?c", "?h")), (In, ("?c", "?hat"))],
        [(Dirty, ("?h",))],
        [(ParentsHappy, ("?h",)), (Clean, ("?h",)), (Out, ("?c", "?h"))],
    )

    Domain = base.define_problem(
        "DummyDomain",
        types=[Cat, Hat, House],
        predicates=[In, Out, Clean, Dirty, ParentsHappy],
        actions=[CleanHouse, LetIn],
        constants=[],
    )

    objects = [Cat("cat"), Hat("hat"), House("house")]
    goal = [Clean(House("house"))]
    static_literals = [In(Cat("cat"), Hat("hat"))]
    return Domain(objects, goal, static_literals)
