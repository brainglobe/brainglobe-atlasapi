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
from __future__ import division, print_function, absolute_import
import operator as op
from six import string_types
import functools

import numpy as np
import pandas as pd
from treelib import Tree

from .simple_tree import SimpleTree


class StructureTree(SimpleTree):
    def __init__(self, nodes):
        """A tree whose nodes are brain structures and whose edges indicate 
        physical containment.
        
        Parameters
        ----------
        nodes : list of dict
            Each specifies a structure. Fields are:
            
            'acronym' : str
                Abbreviated name for the structure.
            'rgb_triplet' : str
                Canonical RGB uint8 color assigned to this structure
            'graph_id' : int
                Specifies the structure graph containing this structure.
            'graph_order' : int
                Canonical position in the flattened structure graph.
            'id': int
                Unique structure specifier.
            'name' : str
                Full name of structure.
            'structure_id_path' : list of int
                This structure's ancestors (inclusive) from the root of the 
                tree.
        
        """

        super(StructureTree, self).__init__(
            nodes,
            lambda s: int(s["id"]),
            lambda s: s["structure_id_path"][-2]
            if len(s["structure_id_path"]) > 1
            and s["structure_id_path"] is not None
            and np.isfinite(s["structure_id_path"][-2])
            else None,
        )

    def get_structures_by_id(self, structure_ids):
        """Obtain a list of brain structures from their structure ids
        
        Parameters
        ----------
        structure_ids : list of int
            Get structures corresponding to these ids.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
        
        """
        if isinstance(structure_ids, list):
            return self.nodes(structure_ids)
        else:
            return self.nodes([structure_ids])[0]

    def get_structures_by_name(self, names):
        """Obtain a list of brain structures from their names,
        
        Parameters
        ----------
        names : list of str
            Get structures corresponding to these names.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
            
        """
        if isinstance(names, list):
            return self.nodes_by_property("name", names)
        else:
            return self.nodes_by_property("name", [names])[0]

    def get_structures_by_acronym(self, acronyms):
        """Obtain a list of brain structures from their acronyms
        
        Parameters
        ----------
        names : list of str
            Get structures corresponding to these acronyms.
            
        Returns
        -------
        list of dict : 
            Each item describes a structure.
            
        """
        if isinstance(acronyms, list):
            return self.nodes_by_property("acronym", acronyms)
        else:
            return self.nodes_by_property("acronym", [acronyms])[0]

    def get_colormap(self):
        """Get a dictionary mapping structure ids to colors across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are RGB lists of integers.
        
        """

        return self.value_map(lambda x: x["id"], lambda y: y["rgb_triplet"])

    def get_name_map(self):
        """Get a dictionary mapping structure ids to names across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are structure name strings.
        
        """

        return self.value_map(lambda x: x["id"], lambda y: y["name"])

    def get_id_acronym_map(self):
        """Get a dictionary mapping structure acronyms to ids across all nodes.
        
        Returns
        -------
        dict : 
            Keys are structure acronyms. Values are structure ids.
        
        """

        return self.value_map(lambda x: x["acronym"], lambda y: y["id"])

    def get_ancestor_id_map(self):
        """Get a dictionary mapping structure ids to ancestor ids across all 
        nodes. 
        
        Returns
        -------
        dict : 
            Keys are structure ids. Values are lists of ancestor ids.
        
        """

        return self.value_map(
            lambda x: x["id"], lambda y: self.ancestor_ids([y["id"]])[0]
        )

    def structure_descends_from(self, child_id, parent_id):
        """Tests whether one structure descends from another. 
        
        Parameters
        ----------
        child_id : int
            Id of the putative child structure.
        parent_id : int
            Id of the putative parent structure.
            
        Returns
        -------
        bool :
            True if the structure specified by child_id is a descendant of 
            the one specified by parent_id. Otherwise False.
        
        """

        return parent_id in self.ancestor_ids([child_id])[0]

    def has_overlaps(self, structure_ids):
        """Determine if a list of structures contains structures along with 
        their ancestors
        
        Parameters
        ----------
        structure_ids : list of int
            Check this set of structures for overlaps
            
        Returns
        -------
        set : 
            Ids of structures that are the ancestors of other structures in 
            the supplied set.
        
        """

        ancestor_ids = functools.reduce(
            op.add, map(lambda x: x[1:], self.ancestor_ids(structure_ids))
        )
        return set(ancestor_ids) & set(structure_ids)

    def export_label_description(
        self,
        alphas=None,
        exclude_label_vis=None,
        exclude_mesh_vis=None,
        label_key="acronym",
    ):
        """Produces an itksnap label_description table from this structure tree

        Parameters
        ----------
        alphas : dict, optional
            Maps structure ids to alpha levels. Optional - will only use provided ids.
        exclude_label_vis : list, optional
            The structures denoted by these ids will not be visible in ITKSnap.
        exclude_mesh_vis : list, optional
            The structures denoted by these ids will not have visible meshes in ITKSnap.
        label_key: str, optional
            Use this column for display labels.

        Returns
        -------
        pd.DataFrame : 
            Contains data needed for loading as an ITKSnap label description file.

        """

        if alphas is None:
            alphas = {}
        if exclude_label_vis is None:
            exclude_label_vis = set([])
        if exclude_mesh_vis is None:
            exclude_mesh_vis = set([])

        df = pd.DataFrame(
            [
                {
                    "IDX": node["id"],
                    "-R-": node["rgb_triplet"][0],
                    "-G-": node["rgb_triplet"][1],
                    "-B-": node["rgb_triplet"][2],
                    "-A-": alphas.get(node["id"], 1.0),
                    "VIS": 1 if node["id"] not in exclude_label_vis else 0,
                    "MSH": 1 if node["id"] not in exclude_mesh_vis else 0,
                    "LABEL": node[label_key],
                }
                for node in self.nodes()
            ]
        ).loc[:, ("IDX", "-R-", "-G-", "-B-", "-A-", "VIS", "MSH", "LABEL")]

        return df

    @staticmethod
    def clean_structures(
        structures, whitelist=None, data_transforms=None, renames=None
    ):
        """Convert structures_with_sets query results into a form that can be 
        used to construct a StructureTree
        
        Parameters
        ----------
        structures : list of dict
            Each element describes a structure. Should have a structure id path 
            field (str values) and a structure_sets field (list of dict).
        whitelist : list of str, optional
            Only these fields will be included in the final structure record. Default is 
            the output of StructureTree.whitelist.
        data_transforms : dict, optional
            Keys are str field names. Values are functions which will be applied to the 
            data associated with those fields. Default is to map colors from hex to rgb and 
            convert the structure id path to a list of int.
        renames : dict, optional
            Controls the field names that appear in the output structure records. Default is 
            to map 'color_hex_triplet' to 'rgb_triplet'.
            
        Returns
        -------
        list of dict : 
            structures, after conversion of structure_id_path and structure_sets 
        
        """

        if whitelist is None:
            whitelist = StructureTree.whitelist()

        if data_transforms is None:
            data_transforms = StructureTree.data_transforms()

        if renames is None:
            renames = StructureTree.renames()
            whitelist.extend(renames.values())

        for ii, st in enumerate(structures):

            StructureTree.collect_sets(st)
            record = {}

            for name in whitelist:

                if name not in st:
                    continue
                data = st[name]

                if name in data_transforms:
                    data = data_transforms[name](data)

                if name in renames:
                    name = renames[name]

                record[name] = data

            structures[ii] = record

        return structures

    @staticmethod
    def data_transforms():
        return {
            "color_hex_triplet": StructureTree.hex_to_rgb,
            "structure_id_path": StructureTree.path_to_list,
        }

    @staticmethod
    def renames():
        return {"color_hex_triplet": "rgb_triplet"}

    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert a hexadecimal color string to a uint8 triplet
        
        Parameters
        ----------
        hex_color : string 
            Must be 6 characters long, unless it is 7 long and the first 
            character is #. If hex_color is a triplet of int, it will be 
            returned unchanged.
            
        Returns
        -------
        list of int : 
            3 characters long - 1 per two characters in the input string.
        
        """

        if not isinstance(hex_color, string_types):
            return list(hex_color)

        if hex_color[0] == "#":
            hex_color = hex_color[1:]

        return [int(hex_color[a * 2 : a * 2 + 2], 16) for a in range(3)]

    @staticmethod
    def path_to_list(path):
        """Structure id paths are sometimes formatted as "/"-seperated strings.
        This method converts them to a list of integers, if needed.
        """

        if not isinstance(path, string_types):
            return list(path)

        return [int(stid) for stid in path.split("/") if stid != ""]

    def print_structures(self, to_file=False, save_filepath=None):
        """ 
        Prints the name of every structure in the structure tree to the console.
        :param to_file: bool, default False. If True the tree structure is saved to 
            a file (at save_filepath) instead of printd to REPL
        :param save_filepath: str, if to_file = True, pass the path to a .txt file 
            where the tree structure will be saved.
        """
        names = [n["name"] for n in self.nodes()]
        acronyms = [n["acronym"] for n in self.nodes()]

        sort_idx = np.argsort(acronyms)
        acronyms, names = (
            np.array(acronyms)[sort_idx],
            np.array(names)[sort_idx],
        )

        if not to_file:
            [print("({}) - {}".format(a, n)) for a, n in zip(acronyms, names)]
        else:
            if save_filepath is None:
                raise ValueError(
                    "If setting to_file as True, you need to pass the path to \
                                            a .txt file where the tree will be saved"
                )
            elif not save_filepath.endswith(".txt"):
                raise ValueError(
                    f"save_filepath should point to a .txt file, not: {save_filepath}"
                )

            with open(save_filepath, "w") as out:
                for a, n in zip(acronyms, names):
                    out.write("({}) - {}\n".format(a, n))

    def get_structures_tree(self):
        """
            Creates a 'tree' graph with the hierarchical organisation of all structures
        """

        def add_descendants_to_tree(
            self, id_to_acronym_map, tree, structure_id, parent_id
        ):
            """
                Recursively goes through all the the descendants of a region and adds them to the tree
            """
            tree.create_node(
                tag=id_to_acronym_map[structure_id],
                identifier=structure_id,
                parent=parent_id,
            )
            descendants = self.child_ids([structure_id])[0]

            if len(descendants):
                for child in descendants:
                    add_descendants_to_tree(
                        self, id_to_acronym_map, tree, child, structure_id
                    )

        # Create a Tree structure and initialise with root
        acronym_to_id_map = self.get_id_acronym_map()
        id_to_acronym_map = {v: k for k, v in acronym_to_id_map.items()}

        root = acronym_to_id_map["root"]
        tree = Tree()
        tree.create_node(tag="root", identifier=root)

        # Recursively iterate through hierarchy#
        for child in self.child_ids([root])[0]:
            add_descendants_to_tree(self, id_to_acronym_map, tree, child, root)

        return tree

    def print_structures_tree(self, to_file=False, save_filepath=None):
        """
            Prints a 'tree' graph with the hierarchical organisation of all structures

            :param to_file: bool, default False. If True the tree structure is saved to 
                a file (at save_filepath) instead of printd to REPL
            :param save_filepath: str, if to_file = True, pass the path to a .txt file 
                where the tree structure will be saved.
        """

        tree = self.get_structures_tree()

        if not to_file:
            tree.show()
        else:
            if save_filepath is None:
                raise ValueError(
                    "If setting to_file as True, you need to pass the path to \
                                            a .txt file where the tree will be saved"
                )
            elif not save_filepath.endswith(".txt"):
                raise ValueError(
                    f"save_filepath should point to a .txt file, not: {save_filepath}"
                )

            tree.save2file(save_filepath)
        return tree
