# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import pytest
from numpy import allclose
import sys
import pandas as pd

from brainatlas_api.structures.structure_tree import StructureTree

if sys.version_info > (3,):
    long = int


@pytest.fixture
def nodes():
    list = [
        {
            "acronym": "root",
            "id": 997,
            "name": "root",
            "structure_id_path": [997],
            "rgb_triplet": [255, 255, 255],
        },
        {
            "acronym": "grey",
            "id": 8,
            "name": "Basic cell groups and regions",
            "structure_id_path": [997, 8],
            "rgb_triplet": [191, 218, 227],
        },
        {
            "acronym": "CH",
            "id": 567,
            "name": "Cerebrum",
            "structure_id_path": [997, 8, 567],
            "rgb_triplet": [176, 240, 255],
        },
    ]

    return list


@pytest.fixture
def tree(nodes):
    return StructureTree(nodes)


def test_get_structures_by_id(tree):

    obtained = tree.get_structures_by_id([1, 2])
    assert len(obtained) == 2


def test_get_structures_by_name(tree):

    obtained = tree.get_structures_by_name(["root"])
    assert len(obtained) == 1


def test_get_structures_by_acronym(tree):

    obtained = tree.get_structures_by_acronym(["root", "grey", "CH"])
    assert len(obtained) == 3


def test_get_colormap(tree):

    obtained = tree.get_colormap()
    assert allclose(obtained[997], [255, 255, 255])
    assert allclose(obtained[567], [176, 240, 255])


def test_get_name_map(tree):

    obtained = tree.get_name_map()
    assert obtained[997] == "root"
    assert obtained[567] == "Cerebrum"


def test_get_id_acronym_map(tree):

    obtained = tree.get_id_acronym_map()
    assert obtained["root"] == 997


def test_get_ancestor_id_map(tree):

    obtained = tree.get_ancestor_id_map()
    assert set(obtained[567]) == {567, 8, 997}


def test_structure_descends_from(tree):

    assert tree.structure_descends_from(567, 8)
    assert not tree.structure_descends_from(8, 567)


def test_has_overlaps(tree):

    obtained = tree.has_overlaps([567, 8])
    assert obtained == {8}


@pytest.mark.parametrize(
    "inp,out",
    [
        ("990099", [153, 0, 153]),
        ("#990099", [153, 0, 153]),
        ([153, 0, 153], [153, 0, 153]),
        ((153.0, 0.0, 153.0), [153, 0, 153]),
        ([long(153), long(0), long(153)], [153, 0, 153]),
    ],
)
def test_hex_to_rgb(inp, out):
    obt = StructureTree.hex_to_rgb(inp)
    assert allclose(obt, out)


@pytest.mark.parametrize(
    "inp,out",
    [
        ("/1/2/3/", [1, 2, 3]),
        ("1/2/3/", [1, 2, 3]),
        ("/1/2/3", [1, 2, 3]),
        ("1/2/3", [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        ([1.0, long(2), 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        ("", []),
    ],
)
def test_path_to_list(inp, out):
    obt = StructureTree.path_to_list(inp)
    assert allclose(obt, out)


def test_export_label_description(tree):
    exp = pd.DataFrame(
        {
            "IDX": [997, 8, 567],
            "-R-": [255, 191, 176],
            "-G-": [255, 218, 240],
            "-B-": [255, 227, 255],
            "-A-": [1.0, 1.0, 1.0],
            "VIS": [1, 1, 1],
            "MSH": [1, 1, 1],
            "LABEL": ["root", "grey", "CH"],
        }
    ).loc[:, ("IDX", "-R-", "-G-", "-B-", "-A-", "VIS", "MSH", "LABEL")]

    obt = tree.export_label_description()
    pd.testing.assert_frame_equal(obt, exp)


def test_print_structures(tree):
    tree.print_structures()


def test_print_structures_tree(tree):
    tree.print_structures_tree()
