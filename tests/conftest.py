import csv
import glob
import os
from typing import AbstractSet, NamedTuple

import pytest

import pddlenv
from pddlenv import base

PDDL_ROOT_DIR = os.path.join(os.path.dirname(__file__), "pddl")


class PDDLTestCase(NamedTuple):
    domain_filename: str
    problem_filename: str
    init_literals: AbstractSet[base.Predicate]
    problem: base.Problem
    path_length: int


@pytest.fixture
def domain_directory(request):
    return os.path.join(PDDL_ROOT_DIR, request.param)


def pytest_generate_tests(metafunc):
    if "pddl_test_case" in metafunc.fixturenames:
        test_case_info = []
        domain_paths = glob.glob(os.path.join(PDDL_ROOT_DIR, "*/*.pddl"))
        for domain_path in domain_paths:
            print(f"domain_path: {domain_path}")
            dir_path = os.path.dirname(domain_path)
            problem_dir = os.path.join(dir_path, "problems")
            solution_path = os.path.join(problem_dir, "solutions.csv")

            with open(solution_path, newline="") as f:
                for solution in csv.DictReader(f):
                    print(f"solution: {solution}")
                    problem_path = os.path.join(problem_dir, f"{solution['problem']}.pddl")
                    assert os.path.exists(problem_path)
                    length = int(solution["length"])
                    test_case_info.append((domain_path, problem_path, length))

        metafunc.parametrize("pddl_test_case", test_case_info, indirect=True)


@pytest.fixture
def pddl_test_case(request):
    domain_path, problem_path, length = request.param
    init_literals, problem = pddlenv.parse_pddl_problem(domain_path, problem_path)
    return PDDLTestCase(domain_path, problem_path, init_literals, problem, length)


@pytest.fixture(scope="session", name="dummy_problem")
def make_dummy_problem():
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
    RemoveCat = base.define_action(
        "RemoveCat",
        (("?c", (Cat,)), ("?hat", (Hat,))),
        [(In, ("?c", "?hat"))],
        [],
        [(In, ("?c", "?hat"))],
    )

    Domain = base.define_problem(
        "DummyDomain",
        types=[Cat, Hat, House],
        predicates=[In, Out, Clean, Dirty, ParentsHappy],
        actions=[CleanHouse, LetIn, RemoveCat],
        constants=[],
    )

    objects = [Cat("cat"), Hat("hat"), House("house")]
    goal = [Clean(House("house"))]
    static_literals = [In(Cat("cat"), Hat("hat"))]
    return Domain(objects, goal, static_literals)
